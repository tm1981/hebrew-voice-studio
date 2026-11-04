"""VibeVoice worker (conda env 'omnivoice', whose transformers has VibeVoice built in; it adds
diffusers for the noise scheduler). Microsoft's long-form English TTS, as two engines:
`--size 1.5b` (vibevoice/VibeVoice-1.5B-hf, about 7 GB of GPU memory) and `--size 7b`
(vibevoice/VibeVoice-7B-hf, about 19 GB). Both MIT. English and Chinese only: Hebrew text
comes out as nonsense.

Voices:
  alice, carter, frank, maya, samuel - Microsoft's sample recordings (bezzam/vibevoice_samples)
  <custom>                           - cloned from a recording in voices/, no transcript needed

Without a seed every take is a little different; the same seed gives the same reading.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from worker_common import serve

MODELS = {
    "1.5b": ("vibevoice/VibeVoice-1.5B-hf", "d4cd82eb4371bf3e3d088e1ea7fa4eefb60449bf"),
    "7b": ("vibevoice/VibeVoice-7B-hf", "aae5684b9755da90bb1c75417a9bca4454dab418"),
}
SAMPLES, SAMPLES_REVISION = "bezzam/vibevoice_samples", "0abf6eee9bdf9fe0f11775166134223a39008442"
BUILTIN = {"alice": "en-Alice_woman", "carter": "en-Carter_man", "frank": "en-Frank_man",
           "maya": "en-Maya_woman", "samuel": "in-Samuel_man"}
REF_MAX_SECONDS = 30  # a longer reference only makes the prompt longer


class VibeVoiceBackend:
    sample_rate = 24000
    native_speed = False  # the server stretches the audio
    custom_voices = True

    def __init__(self, size: str):
        self.repo, self.revision = MODELS[size]
        self.voices: dict = {}
        self.builtin_voices = set(BUILTIN)

    def load(self):
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForTextToWaveform, AutoProcessor

        print(f"Loading {self.repo}...", flush=True)
        self.processor = AutoProcessor.from_pretrained(self.repo, revision=self.revision)
        self.model = AutoModelForTextToWaveform.from_pretrained(self.repo, revision=self.revision,
                                                                dtype=torch.bfloat16, device_map="cuda:0")
        samples = Path(snapshot_download(SAMPLES, repo_type="dataset", revision=SAMPLES_REVISION,
                                         allow_patterns=[f"voices/{f}.wav" for f in BUILTIN.values()]))
        for name, file in BUILTIN.items():
            self.voices[name] = self._reference(samples / "voices" / f"{file}.wav")
        self.synthesize("Hello, how are you today?", "alice", 1.0)  # warm-up

    def _reference(self, path: Path) -> np.ndarray:
        import librosa

        audio, _ = librosa.load(str(path), sr=self.sample_rate, mono=True, duration=REF_MAX_SECONDS)
        return audio

    def add_voice(self, name: str, path: Path):
        self.voices[name] = self._reference(path)

    def remove_voice(self, name: str):
        self.voices.pop(name, None)

    def synthesize(self, text: str, voice: str, speed: float, guidance=1.3, steps=10, seed=None, **_):
        if seed is not None:
            torch.manual_seed(seed)
        conversation = [{"role": "0", "content": [{"type": "text", "text": text},
                                                  {"type": "audio", "audio": self.voices[voice]}]}]
        inputs = self.processor.apply_chat_template(conversation, return_dict=True, tokenize=True,
                                                    add_generation_prompt=True).to(self.model.device, self.model.dtype)
        audio = self.model.generate(**inputs, guidance_scale=guidance, num_diffusion_steps=steps)
        audio = audio[0] if isinstance(audio, (list, tuple)) else audio
        return audio.float().cpu().numpy().reshape(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=MODELS, default="1.5b")
    size, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]  # worker_common reads --port
    serve(VibeVoiceBackend(size.size), f"VibeVoice {size.size.upper()} worker")
