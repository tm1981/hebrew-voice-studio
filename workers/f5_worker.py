"""F5-TTS Hebrew worker (conda env 'f5'): Yzamari/f5tts-hebrew-v2, F5-TTS v1 Base fine-tuned on
158 h of Hebrew whose text was pointed with Phonikud (stress marks included), so input text gets
the same Phonikud niqqud. Typed niqqud is kept per word (niqqud.py).

Voices: cloned from the recordings in voices/ only. The model card mentions 58 built-in voices,
but they lived in the author's GitHub project, which is gone. F5 needs the reference's exact
words: F5's own preprocessing cuts the clip to <12 s at a pause and transcribes it with Whisper
(large-v3-turbo); both are cached as voices/.cache/<name>.f5.wav and .f5.txt.

Whisper check (2 cloned voices x 15 sentences x 4 runs, words added/dropped/garbled): 4 of 120,
all in female1; without the reference's silence tail (see add_voice) 3 of 60.
About 2.5x real time on an RTX 5090.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from worker_common import serve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model_files import ensure as ensure_model  # noqa: E402
from niqqud import diacritize_respecting_typed  # noqa: E402

MODEL = "Yzamari/f5tts-hebrew-v2"
REF_TAIL_SECONDS = 0.3  # silence after the reference clip, see add_voice


def _load_audio(path, *args, **kwargs):
    """torchaudio 2.9 reads files through torchcodec, which needs FFmpeg DLLs this PC lacks."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _load_audio


class F5Backend:
    sample_rate = 24000
    native_speed = True
    custom_voices = True

    def __init__(self):
        self.voices: dict = {}
        self.builtin_voices: set = set()

    def load(self):
        from importlib.resources import files

        from f5_tts.infer.utils_infer import load_model, load_vocoder
        from f5_tts.model import DiT
        from huggingface_hub import snapshot_download
        from omegaconf import OmegaConf
        from phonikud_onnx import Phonikud

        print("Loading F5-TTS Hebrew...", flush=True)
        self.phonikud = Phonikud(str(ensure_model("phonikud", "kokoro-hebrew/phonikud-1.0.int8.onnx")))
        weights = Path(snapshot_download(MODEL))
        cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
        # vocab_file sets the text embedding size to this vocabulary (the card's "override text_num_embeds")
        self.model = load_model(DiT, OmegaConf.to_container(cfg.model.arch), str(weights / "model.safetensors"),
                                vocab_file=str(weights / "vocab.txt"), device="cuda")
        self.vocoder = load_vocoder(device="cuda")

    def _point(self, text: str) -> str:
        return diacritize_respecting_typed(text, self.phonikud.add_diacritics).replace("|", "")  # '|' marks prefixes

    def add_voice(self, name: str, path: Path):
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text

        clip, words = path.with_suffix(".f5.wav"), path.with_suffix(".f5.txt")
        if not (words.exists() and words.stat().st_mtime >= path.stat().st_mtime):
            # a placeholder transcript makes F5 only cut the clip; Whisper then hears the cut clip
            tmp_clip, _ = preprocess_ref_audio_text(str(path), "-", show_info=lambda *a: None)
            shutil.copyfile(tmp_clip, clip)
            words.write_text(self._transcribe(clip), encoding="utf-8")
        # The clip ends right on its last word (voices/.cache clips are trimmed tight, and F5 adds
        # only 50 ms), and then F5 often starts its sentence with the reference's last words:
        # 4 of 16 takes of one sentence ("ראיתי הרבה זמן הרכבת..."). 0.3 s of silence after the
        # reference: 0 of 16 with the same seeds. F5's preprocessing would trim the silence
        # again, so the padded clip goes straight to inference.
        audio, sr = sf.read(str(clip), dtype="float32")
        padded = path.with_suffix(".f5pad.wav")
        sf.write(padded, np.concatenate([audio, np.zeros(int(sr * REF_TAIL_SECONDS), dtype=np.float32)]), sr)
        text = self._point(words.read_text(encoding="utf-8")).rstrip()
        self.voices[name] = (str(padded), text + (" " if text.endswith(".") else ". "))  # F5's own ending

    @staticmethod
    def _transcribe(path: Path) -> str:
        """Whisper large-v3-turbo, loaded for this and freed after. The audio goes in as an array:
        given a path, transformers decodes it with torchcodec (see _load_audio)."""
        import gc

        from transformers import pipeline

        audio, sr = sf.read(str(path), dtype="float32")
        asr = pipeline("automatic-speech-recognition", "openai/whisper-large-v3-turbo",
                       torch_dtype=torch.float16, device="cuda")
        text = asr({"raw": audio, "sampling_rate": sr}, generate_kwargs={"language": "he", "task": "transcribe"})["text"]
        del asr
        gc.collect()
        torch.cuda.empty_cache()
        return text.strip()

    def remove_voice(self, name: str):
        self.voices.pop(name, None)

    def synthesize(self, text: str, voice: str, speed: float, steps=32, guidance=2.0, sway=-1.0, seed=None, **_):
        from f5_tts.infer.utils_infer import infer_process

        if seed is not None:
            torch.manual_seed(seed)
        ref_audio, ref_text = self.voices[voice]
        wav, _, _ = infer_process(ref_audio, ref_text, self._point(text), self.model, self.vocoder,
                                  nfe_step=steps, cfg_strength=guidance, sway_sampling_coef=sway,
                                  speed=speed, device="cuda", show_info=lambda *a: None)
        return wav


if __name__ == "__main__":
    serve(F5Backend(), "F5-TTS Hebrew worker")
