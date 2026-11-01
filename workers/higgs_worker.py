"""Higgs Audio v3 worker (conda env 'higgs'). Boson AI's weights (bosonai/higgs-tts-3-4b; the port's
model.safetensors is the same file) run through the plain-transformers port
multimodalart/higgs-audio-v3-tts-4b-transformers, whose modeling code transformers loads from the
Hugging Face cache (models/hf, see HF_HOME in server.py) at a pinned revision. Voice presets come
from models/higgs-voices.

Every voice is a reference recording: Higgs without one was unusably slow on Hebrew
(199s for one sentence in testing). Built-in voices are only the Studio's generic
language-named presets; its presets of real, identifiable people are deliberately left out.
No reference transcript is passed: in testing (15 sentences, 2 voices, 2 runs) it didn't change
Whisper's error count.
"""
import os
from pathlib import Path

import soundfile as sf
import torch
import torch._dynamo
from transformers import AutoModelForCausalLM, AutoTokenizer

from worker_common import serve

ROOT = Path(__file__).resolve().parent.parent
PRESETS_DIR = ROOT / "models" / "higgs-voices"
PRESETS = ["English_Female", "English_Female2", "French_Female", "Arabic_Female", "German_Male", "Spanish_Male"]
REPO = "multimodalart/higgs-audio-v3-tts-4b-transformers"
REVISION = "30f01593ee6a12efa586c92455afe4b76e45095d"  # the remote code runs here, so pin it
SR = 24000
REF_MAX_SEC = 30  # longer references slow the prefill down a lot


class HiggsBackend:
    native_speed = False  # time-stretched by the main server
    custom_voices = True
    sample_rate = SR

    def __init__(self):
        self.voices: dict[str, str] = {}
        self.builtin_voices: set[str] = set()
        self.codes: dict[str, torch.Tensor] = {}  # voice -> encoded reference

    def load(self):
        print("Loading Higgs Audio v3...", flush=True)
        # trust_remote_code: otherwise it stops to ask on the console (a container has none to answer)
        self.tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(REPO, revision=REVISION, trust_remote_code=True,
                                                          dtype=torch.bfloat16).to("cuda").eval()
        self.model.get_audio_codec()
        torch.set_float32_matmul_precision("high")
        if os.environ.get("HIGGS_NO_COMPILE", "").lower() not in ("1", "true", "yes"):
            # about 2x; generation runs on the main thread (see worker_common), which compile needs
            torch._dynamo.config.suppress_errors = True  # fall back to eager instead of crashing
            self.model.model = torch.compile(self.model.model, dynamic=True)
        for name in PRESETS:
            wav = PRESETS_DIR / f"{name}.wav"
            if wav.exists():
                self.add_voice(name, wav)
                self.builtin_voices.add(name)
        if self.voices:
            self.synthesize("שלום, מה שלומך היום?", next(iter(self.voices)), 1.0)  # warm-up (compiles)

    def add_voice(self, name: str, path: Path):
        self.voices[name] = str(path)
        self.codes.pop(name, None)  # encoded on first use

    def remove_voice(self, name: str):
        self.voices.pop(name, None)
        self.codes.pop(name, None)

    def _reference(self, voice: str) -> torch.Tensor:
        if voice not in self.codes:
            wav, sr = sf.read(self.voices[voice], dtype="float32", always_2d=True)
            wav = torch.from_numpy(wav).mean(dim=1)[:int(sr * REF_MAX_SEC)]
            self.codes[voice] = self.model._encode_reference(wav, sr).cpu()
        return self.codes[voice]

    def synthesize(self, text: str, voice: str, speed: float, temperature=1.0, top_p=0.95, top_k=50, seed=None, **_):
        if seed is not None:
            torch.manual_seed(int(seed))
        kw = dict(reference_codes=self._reference(voice), max_new_tokens=2048, temperature=max(0.05, float(temperature)),
                  top_p=float(top_p) if top_p < 1.0 else None, top_k=int(top_k) if top_k > 0 else None)
        # A runaway (audio far too long for the text) is retried, unless a fixed seed would repeat it
        limit = 0.13 * len(text) + 3.0
        for _ in range(1 if seed is not None else 3):
            audio = self.model.generate_speech(text, self.tok, **kw)
            if audio.shape[-1] / SR <= limit:
                break
            print(f"Higgs runaway ({audio.shape[-1] / SR:.1f}s for {len(text)} chars), retrying", flush=True)
        return audio.numpy()


if __name__ == "__main__":
    serve(HiggsBackend(), "Higgs worker")
