"""Magpie-TTS Hebrew worker (conda env 'magpie': nemo_toolkit 3.0.0).

notmax123's Hebrew fine-tune of NVIDIA's magpie_tts_multilingual_357m. Like QwenTTS-he it was
trained on stressed IPA, not Hebrew script, so text goes through RenikudPlus first (typed
niqqud decides a word's reading).

No voice cloning: NVIDIA's release has the context (reference-audio) encoder removed, so the
voice is a baked speaker slot. The fine-tune added 15 Hebrew voices (slots 5-19, names from its
speaker_index.json); NVIDIA's 5 English voices (slots 0-4) are left out.

Two Windows workarounds:
  - pyopenjtalk (the Japanese G2P) has no Windows build. Every tokenizer has to be built so
    the Hebrew token ids keep their offsets, but the Japanese one only needs its phoneme list
    to do that, so an empty module stands in for pyopenjtalk.
  - NeMo maps languages to tokenizers in a table without Hebrew, and falls back to English
    phonemes, which drop most Hebrew sounds. 'he' is added to that table.
"""
import json
import re
import sys
import types
from pathlib import Path

sys.modules.setdefault("pyopenjtalk", types.ModuleType("pyopenjtalk"))

import torch
from huggingface_hub import snapshot_download

from worker_common import serve

ROOT = Path(__file__).resolve().parent.parent
MODEL = "notmax123/magpie_tts_multilingual_357m-he"
CODEC = "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps"
sys.path.insert(0, str(ROOT))
from model_files import ensure as ensure_model  # noqa: E402

RENIKUD = "renikud-plus/model.onnx"  # in models/, downloaded on first use
LATIN_RUN = re.compile(r"[A-Za-z][\w'’\-]*(?:[ \t,:\-]+[A-Za-z][\w'’\-]*)*")  # as engines.py


class MagpieBackend:
    native_speed = False
    custom_voices = False

    def __init__(self):
        self.voices: dict = {}
        self.builtin_voices: set[str] = set()
        self.sample_rate = 22050

    def load(self):
        from nemo.collections.tts.modules.magpietts_inference.utils import ModelLoadConfig, load_magpie_model
        from nemo.collections.tts.parts.utils.tts_dataset_utils import LANGUAGE_TOKENIZER_MAP
        from renikud_onnx import G2P

        print("Loading Magpie-TTS Hebrew...", flush=True)
        LANGUAGE_TOKENIZER_MAP["he"] = ["hebrew_chartokenizer"]
        self.g2p = G2P(str(ensure_model("renikud-plus", RENIKUD)), niqqud="use")
        model_dir = Path(snapshot_download(MODEL))
        codec = next(Path(snapshot_download(CODEC)).glob("*.nemo"))
        # hparams.yaml names its tokenizer files as "assets/...", which NeMo would resolve
        # against the working directory; written out with absolute paths to models/magpie/.
        hparams = ROOT / "models" / "magpie" / "hparams.yaml"
        hparams.parent.mkdir(exist_ok=True)
        text = (model_dir / "hparams.yaml").read_text(encoding="utf-8")
        hparams.write_text(text.replace("assets/", (model_dir / "assets").as_posix() + "/"), encoding="utf-8")
        self.model, _ = load_magpie_model(ModelLoadConfig(
            hparams_file=str(hparams), checkpoint_file=str(model_dir / "magpie_hebrew.ckpt"),
            codecmodel_path=str(codec),
        ))
        self.model.eval()
        self.sample_rate = self.model.sample_rate
        speakers = json.loads((model_dir / "speaker_index.json").read_text(encoding="utf-8"))
        self.voices = {name.replace("_hebrew", "-he"): index for name, index in speakers.items()}
        self.builtin_voices = set(self.voices)
        self.synthesize("שלום, מה שלומך היום?", next(iter(self.voices)), 1.0)  # warm-up

    def synthesize(self, text: str, voice: str, speed: float, temperature=0.6, top_k=80, guidance=2.5, seed=None, **_):
        params = self.model.inference_parameters
        params.temperature, params.topk, params.cfg_scale = temperature, top_k, guidance
        if seed is not None:
            torch.manual_seed(seed)
        # Without a closing mark the model has no cue to end the sentence. Whisper, 2 voices x 20
        # sentences, one word misheard: 7/40 -> 4/40.
        text = text.rstrip()
        if text and text[-1] not in ".?!…:;,":
            text += "."
        if LATIN_RUN.search(text):
            return self._synthesize_mixed(text, self.voices[voice])
        with torch.inference_mode():
            audio, length = self.model.do_tts(self.g2p.phonemize(text), language="he", speaker_index=self.voices[voice])
        return audio[0, : length[0]].float().cpu().numpy()

    def _synthesize_mixed(self, text: str, speaker: int):
        """English inside Hebrew. RenikudPlus leaves Latin letters as they are and the Hebrew
        tokenizer has no use for them (Whisper heard "ב-WhatsApp אתמול" as "בעצפתמול"), so
        English runs go through the base model's English phoneme tokenizer instead, in the same
        token sequence: one pass, no seams. Whisper then heard the English words and
        English-only sentences right. do_tts takes one language, so this is its single-chunk
        path with our own tokens."""
        tok, ids, pos = self.model.tokenizer, [], 0
        for run in [*LATIN_RUN.finditer(text), None]:
            hebrew = text[pos : run.start() if run else len(text)]
            if hebrew.strip():
                ids += tok.encode(text=self.g2p.phonemize(hebrew), tokenizer_name="hebrew_chartokenizer")
            if run:
                ids += tok.encode(text=f" {run.group(0)} ", tokenizer_name="english_phoneme")
                pos = run.end()
        ids.append(self.model.eos_id)
        device = self.model.device
        with torch.inference_mode():
            output = self.model.generate_speech(
                {"text": torch.tensor([ids], dtype=torch.int32, device=device),
                 "text_lens": torch.tensor([len(ids)], device=device), "speaker_indices": speaker},
                chunk_state=self.model.create_chunk_state(batch_size=1), end_of_text=[True],
                beginning_of_text=True, use_cfg=True, use_local_transformer_for_inference=True,
            )
            n = output.predicted_codes_lens[0]
            codes = output.predicted_codes[0, :, :n].unsqueeze(0)
            audio, length, _ = self.model._codec_helper.codes_to_audio(codes, torch.tensor([n], device=device))
        return audio[0, : length[0]].float().cpu().numpy()


if __name__ == "__main__":
    serve(MagpieBackend(), "Magpie-TTS Hebrew worker")
