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
import random
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

MAX_GUIDANCE = 1.4
MAX_TAKES = 3
CLEAN_FLOOR_DB = -35  # sound between words against speech; above this a bed of sound is audible


def floor_db(audio: np.ndarray, sample_rate: int) -> float:
    """Level of the quietest moments against speech."""
    frame = int(sample_rate * 0.03)
    rms = np.sqrt(np.convolve(audio ** 2, np.ones(frame) / frame, "same"))[:: frame // 2] + 1e-7
    return float(20 * np.log10(np.percentile(rms, 10) / np.percentile(rms, 95)))


class VibeVoiceBackend:
    sample_rate = 24000
    native_speed = False  # the server stretches the audio
    custom_voices = True

    def __init__(self, size: str):
        self.repo, self.revision = MODELS[size]
        self.voices: dict = {}
        self.builtin_voices = set(BUILTIN)

    def load(self):
        from diffusers import DPMSolverMultistepScheduler
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForTextToWaveform, AutoProcessor

        print(f"Loading {self.repo}...", flush=True)
        self.processor = AutoProcessor.from_pretrained(self.repo, revision=self.revision)
        self.model = AutoModelForTextToWaveform.from_pretrained(self.repo, revision=self.revision,
                                                                dtype=torch.bfloat16, device_map="cuda:0")
        # Microsoft's demo uses the SDE solver, Transformers' default isn't it, and that default
        # adds a bed of sound. Takes with a raised floor between words, same text and seeds:
        # 1.5B 29/36 -> 12/36, 7B 28/36 -> 9/36. Same speed.
        scheduler = self.model.generation_config.to_dict().get("noise_scheduler_config") or {}
        self.scheduler = DPMSolverMultistepScheduler(**{**scheduler, "algorithm_type": "sde-dpmsolver++"})
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

    def synthesize(self, text: str, voice: str, speed: float, guidance=1.3, steps=10, seed=None, retake=True, **_):
        # The model adds a bed of sound to some takes and not others, and the seed decides which:
        # the same text ran from -18 dB to -62 dB between words. Keep the first clean take: takes with
        # a raised floor 1.5B 12/36 -> 6/36, 7B 9/36 -> 5/36, for about twice the time.
        seed = random.randrange(2 ** 31) if seed is None else int(seed)
        best, best_floor = None, 0.0
        for attempt in range(MAX_TAKES if retake else 1):
            audio = self._take(text, voice, guidance, steps, seed + attempt)
            floor = floor_db(audio, self.sample_rate)
            if best is None or floor < best_floor:
                best, best_floor = audio, floor
            if floor <= CLEAN_FLOOR_DB:
                break
        return best

    def _take(self, text, voice, guidance, steps, seed):
        # Guidance feeds the choice to stop, and on Hebrew the model can end a reading almost at once.
        # 7B at 2.0: no sound for 9 of 40 sentences, on every seed. 1.4: 2 of 40 cut to 0.1 s.
        # 1.5B at the default: one sentence cut to 0.9 s. Guidance 1.0 read all of them in full.
        audio = self._read(text, voice, min(float(guidance), MAX_GUIDANCE), steps, seed)
        expected = sum(0.06 if "֐" <= c <= "׿" else 0.03 for c in text if c.isalpha())
        if audio is None or len(audio) / self.sample_rate < expected:
            retry = self._read(text, voice, 1.0, steps, seed)
            if retry is not None and (audio is None or len(retry) > len(audio)):
                audio = retry
        if audio is None:
            raise RuntimeError("VibeVoice ended the reading without any sound")
        return audio

    def _read(self, text, voice, guidance, steps, seed):
        if seed is not None:
            torch.manual_seed(seed)
        conversation = [{"role": "0", "content": [{"type": "text", "text": text},
                                                  {"type": "audio", "audio": self.voices[voice]}]}]
        inputs = self.processor.apply_chat_template(conversation, return_dict=True, tokenize=True,
                                                    add_generation_prompt=True).to(self.model.device, self.model.dtype)
        audio = self.model.generate(**inputs, guidance_scale=guidance, num_diffusion_steps=steps,
                                    noise_scheduler=self.scheduler)
        audio = audio[0] if isinstance(audio, (list, tuple)) else audio
        return None if audio is None else audio.float().cpu().numpy().reshape(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=MODELS, default="1.5b")
    size, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]  # worker_common reads --port
    serve(VibeVoiceBackend(size.size), f"VibeVoice {size.size.upper()} worker")
