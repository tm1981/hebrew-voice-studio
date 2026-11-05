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

The fine-tune never learned Latin letters, so a sentence with no Hebrew in it goes to the base
F5-TTS v1 model, loaded on the first such sentence (CC BY-NC weights, about 1.4 GB). It has no
Hebrew letters either, so a Hebrew reference's words are spelled out in Latin letters (romanize).
The reference has to end on a finished sentence: see _english_voice. A sentence that mixes the
languages stays with the fine-tune, which garbles the English; joining the two models inside a
sentence broke short Hebrew fragments.
Whisper check (3 references x 8 sentences x 2 runs, hand-spelled references): no words added or
dropped in 48, 1 misheard.
"""
import re
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
BASE_MODEL = ("SWivid/F5-TTS", "F5TTS_v1_Base/model_1250000.safetensors", "84e5a410d9cead4de2f847e7c9369a6440bdfaca")
REF_TAIL_SECONDS = 0.3  # silence after the reference clip, see add_voice
ENGLISH_REF_TAIL_SECONDS = 0.5
MIN_ENGLISH_REF_SECONDS = 4
HEBREW = re.compile("[\u0590-\u05ff]")
LATIN = re.compile("[A-Za-z]")

CONSONANTS = dict(zip("אבגדהוזחטיכךלמםנןסעפףצץקרת",
                      ["", "v", "g", "d", "h", "v", "z", "ch", "t", "y", "ch", "ch", "l", "m", "m", "n", "n", "s",
                       "", "f", "f", "tz", "tz", "k", "r", "t"]))
VOWELS = {"\u05b0": "", "\u05b1": "e", "\u05b2": "a", "\u05b3": "o", "\u05b4": "i", "\u05b5": "e", "\u05b6": "e",
          "\u05b7": "a", "\u05b8": "a", "\u05b9": "o", "\u05ba": "o", "\u05bb": "u", "\u05c7": "o"}
DAGESH, METEG, SIN_DOT = "\u05bc", "\u05bd", "\u05c2"  # Phonikud's meteg: a spoken shva


def romanize(pointed: str) -> str:
    """Pointed Hebrew in Latin letters, close enough for the base model to line up with the speech."""
    parts = re.findall("([\u05d0-\u05ea])([\u0591-\u05c7]*)|([^\u05d0-\u05ea\u0591-\u05c7]+)", pointed)
    out, previous = [], ""  # previous: the vowel before this letter
    for i, (letter, marks, other) in enumerate(parts):
        if not letter:
            out.append(other)
            previous = ""
            continue
        vowel = next((VOWELS[m] for m in marks if m in VOWELS), None)
        last = i + 1 == len(parts) or not parts[i + 1][0]
        sound = CONSONANTS.get(letter, "")
        if letter in "בכפ" and DAGESH in marks:
            sound = {"ב": "b", "כ": "k", "פ": "p"}[letter]
        elif letter == "ש":
            sound = "s" if SIN_DOT in marks else "sh"
        if letter == "ו" and "\u05b9" in marks and not previous:
            out.append("o")
        elif letter == "ו" and DAGESH in marks and vowel is None and not previous:
            out.append("u")
        elif letter == "ו" and vowel is None and previous in ("o", "u"):
            pass  # a vav carrying the letter before's vowel
        elif letter == "י" and vowel is None and previous in ("i", "e"):
            pass  # a silent yod carrying the vowel
        elif letter == "ה" and last and vowel is None and DAGESH not in marks:
            pass  # silent at the end of a word
        else:
            out.append(sound + ("e" if vowel == "" and METEG in marks else vowel or ""))
        previous = vowel or ""
    return "".join(out)


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
        self.recordings: dict = {}  # name -> recording, for _english_voice
        self.english_voices: dict = {}
        self.base_model = None

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

        self.recordings[name] = path
        self.english_voices.pop(name, None)
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

    def _english_voice(self, name: str):
        """The reference for English: F5's clip cut after its last finished sentence, or the base
        model starts its reading with the words the clip stops on. A clip that stopped on "ככה"
        put "Kasha" before 9 of 16 readings; cut at the sentence before: 0 of 16."""
        if name in self.english_voices:
            return self.english_voices[name]
        path = self.recordings[name]
        clip, words = path.with_suffix(".f5en.wav"), path.with_suffix(".f5en.txt")
        if not (words.exists() and words.stat().st_mtime >= path.stat().st_mtime):
            audio, sr = sf.read(str(path.with_suffix(".f5.wav")), dtype="float32")
            chunks = self._transcribe_words(audio, sr)
            ends = [i for i, c in enumerate(chunks[:-1]) if c["text"].rstrip().endswith((".", "?", "!"))
                    and c["timestamp"][1] >= MIN_ENGLISH_REF_SECONDS]
            last = ends[-1] if ends else len(chunks) - 1
            if ends:
                audio = audio[:int((chunks[last]["timestamp"][1] + 0.06) * sr)]
            tail = np.zeros(int(sr * ENGLISH_REF_TAIL_SECONDS), dtype=np.float32)
            sf.write(clip, np.concatenate([audio, tail]), sr)
            words.write_text("".join(c["text"] for c in chunks[:last + 1]).strip(), encoding="utf-8")
        text = words.read_text(encoding="utf-8")
        text = romanize(self._point(text)).rstrip() if HEBREW.search(text) else text.rstrip()
        self.english_voices[name] = (str(clip), text + (" " if text.endswith(".") else ". "))
        return self.english_voices[name]

    def _english_model(self):
        if self.base_model is None:
            from importlib.resources import files

            from f5_tts.infer.utils_infer import load_model
            from f5_tts.model import DiT
            from huggingface_hub import hf_hub_download
            from omegaconf import OmegaConf

            print("Loading the F5-TTS base model for English...", flush=True)
            repo, file, revision = BASE_MODEL
            cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
            self.base_model = load_model(DiT, OmegaConf.to_container(cfg.model.arch),
                                         hf_hub_download(repo, file, revision=revision), device="cuda")
        return self.base_model

    @staticmethod
    def _transcribe_words(audio, sr) -> list:
        """Whisper's words with their times, in the language it hears: a voice may be recorded in English."""
        import gc

        from transformers import pipeline

        asr = pipeline("automatic-speech-recognition", "openai/whisper-large-v3-turbo",
                       torch_dtype=torch.float16, device="cuda")
        chunks = asr({"raw": audio, "sampling_rate": sr}, return_timestamps="word",
                     generate_kwargs={"task": "transcribe"})["chunks"]
        del asr
        gc.collect()
        torch.cuda.empty_cache()
        return chunks

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
        self.recordings.pop(name, None)
        self.english_voices.pop(name, None)

    def synthesize(self, text: str, voice: str, speed: float, steps=32, guidance=2.0, sway=-1.0, seed=None, **_):
        from f5_tts.infer.utils_infer import infer_process

        if seed is not None:
            torch.manual_seed(seed)
        if LATIN.search(text) and not HEBREW.search(text):
            (ref_audio, ref_text), model, words = self._english_voice(voice), self._english_model(), text
        else:
            (ref_audio, ref_text), model, words = self.voices[voice], self.model, self._point(text)
        wav, _, _ = infer_process(ref_audio, ref_text, words, model, self.vocoder,
                                  nfe_step=steps, cfg_strength=guidance, sway_sampling_coef=sway,
                                  speed=speed, device="cuda", show_info=lambda *a: None)
        return wav


if __name__ == "__main__":
    serve(F5Backend(), "F5-TTS Hebrew worker")
