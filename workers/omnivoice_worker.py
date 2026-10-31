"""OmniVoice worker (conda env 'omnivoice': it needs a newer transformers than Chatterbox).

Voices:
  female, male  - designed from a text description (no recording); a fixed seed per
                  voice keeps them the same speaker across requests.
  <custom>      - cloned from a recording in voices/; the clone prompt (audio tokens plus
                  a Whisper transcript) is cached as voices/<name>.omnivoice.pt.
"""
from pathlib import Path

import torch

from worker_common import serve

DESIGNED_VOICES = {
    "female": ("female", 1234),
    "male": ("male, low pitch", 4321),
}


class OmniVoiceBackend:
    sample_rate = 24000
    native_speed = True
    custom_voices = True

    def __init__(self):
        self.voices: dict = {}
        self.builtin_voices = set(DESIGNED_VOICES)

    def load(self):
        from omnivoice import OmniVoice

        print("Loading OmniVoice...", flush=True)
        self.model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
        self.voices.update(DESIGNED_VOICES)
        self.synthesize("שלום, מה שלומך היום?", "female", 1.0)  # warm-up

    def add_voice(self, name: str, path: Path):
        from omnivoice.models.omnivoice import VoiceClonePrompt

        cache = path.with_suffix(".omnivoice.pt")
        if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
            data = torch.load(cache, map_location="cuda:0", weights_only=True)
            prompt = VoiceClonePrompt(**data)
        else:
            prompt = self.model.create_voice_clone_prompt(str(path))  # transcribes with Whisper
            torch.save(
                {"ref_audio_tokens": prompt.ref_audio_tokens, "ref_text": prompt.ref_text, "ref_rms": prompt.ref_rms},
                cache,
            )
        self.voices[name] = prompt

    def remove_voice(self, name: str):
        self.voices.pop(name, None)

    def synthesize(self, text: str, voice: str, speed: float, steps=32, guidance=2.0, seed=None,
                   design_age="", design_pitch="", design_whisper=False, design_accent="", **_):
        from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

        entry = self.voices[voice]
        kwargs = {"text": text, "language": "he", "speed": speed,
                  "generation_config": OmniVoiceGenerationConfig(num_step=steps, guidance_scale=guidance)}
        if voice in DESIGNED_VOICES:
            instruct, voice_seed = entry
            # extra design words replace the voice's own pitch word (one per category)
            words = [w for w in instruct.split(", ") if not (design_pitch and w.endswith("pitch"))]
            words += [w for w in (design_age, design_pitch, design_accent) if w] + (["whisper"] if design_whisper else [])
            kwargs["instruct"] = ", ".join(words)
            torch.manual_seed(voice_seed if seed is None else seed)
        else:
            kwargs["voice_clone_prompt"] = entry
            if seed is not None:
                torch.manual_seed(seed)
        try:
            return self.model.generate(**kwargs)[0]
        except ValueError:
            # Very short text (one word) can get a zero-length duration estimate; give it a floor.
            return self.model.generate(**kwargs, duration=max(1.0, 0.09 * len(text)))[0]


if __name__ == "__main__":
    serve(OmniVoiceBackend(), "OmniVoice worker")
