"""OmniVoice worker (conda env 'omnivoice': it needs a newer transformers than Chatterbox).

Voices:
  female, male  - designed from a text description (no recording); a fixed seed per
                  voice keeps them the same speaker across requests.
  <custom>      - cloned from a recording in voices/; the clone prompt (audio tokens plus
                  a Whisper transcript) is cached as voices/<name>.omnivoice.pt.

The clone prompt is built here rather than by OmniVoice alone: it cuts a long recording at
a pause and then transcribes it with English-default Whisper, and a transcript that doesn't
match the audio leaks into the speech. With one 27s recording, its misheard last words came
out as an extra word ("איתן,") in front of all 8 test takes (4 sentences, 2 seeds; 13 extra
words). Here the Hebrew ivrit-ai Whisper transcribes it and the cut falls at the end of a
whole phrase: 7 of 8 takes clean, 3 extra words in one.
"""
from pathlib import Path

import torch

from worker_common import serve

DESIGNED_VOICES = {
    "female": ("female", 1234),
    "male": ("male, low pitch", 4321),
}
PROMPT_VERSION = 3  # cached prompts from before (OmniVoice's own cut and transcript) are rebuilt
REF_MAX_SECONDS = 12.0  # OmniVoice recommends 3-10s; cut at the end of the last phrase before this
REF_TAIL_SECONDS = 0.2  # kept after the phrase's last word
ASR_MODEL, ASR_REVISION = "ivrit-ai/whisper-large-v3-turbo", "f33172a8c3c6efbc040a7200e00835257cac0447"  # as stt.py
# OmniVoice scales a clone down to its recording's loudness when that is below 0.1 RMS (0.1
# turns that off), and its own level is still quieter than the other engines: one home
# recording came out at -25 dBFS, Chatterbox with it at -18. Clones are leveled to CLONE_RMS.
MIN_REF_RMS = 0.1
CLONE_RMS, PEAK_LIMIT = 0.12, 0.95  # about -18 dBFS


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
        data = None
        if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
            data = torch.load(cache, map_location="cuda:0", weights_only=True)
            if data.pop("version", 1) != PROMPT_VERSION:
                data = None
        if data is None:
            prompt = self._clone_prompt(path)
            data = {"ref_audio_tokens": prompt.ref_audio_tokens, "ref_text": prompt.ref_text, "ref_rms": prompt.ref_rms}
            torch.save({**data, "version": PROMPT_VERSION}, cache)
        prompt = VoiceClonePrompt(**data)
        prompt.ref_rms = max(prompt.ref_rms, MIN_REF_RMS)
        self.voices[name] = prompt

    def _clone_prompt(self, path: Path):
        from omnivoice.utils.audio import load_audio

        sr = self.model.sampling_rate
        wav = load_audio(str(path), sr)
        phrases = self._transcribe(wav[0], sr)
        # Whole phrases up to REF_MAX_SECONDS (at least the first), cut where the last one ends:
        # a cut mid-word leaves a broken last word that the clone starts its sentence with.
        keep = [p for p in phrases if p[1] <= REF_MAX_SECONDS] or phrases[:1]
        end = min(wav.shape[-1], int((keep[-1][1] + REF_TAIL_SECONDS) * sr))
        text = " ".join(p[2] for p in keep)
        # with ref_text given, OmniVoice doesn't cut the audio again, so the two stay matched
        return self.model.create_voice_clone_prompt((torch.from_numpy(wav[:, :end]), sr), ref_text=text)

    @staticmethod
    def _transcribe(audio, sr: int) -> list[tuple[float, float, str]]:
        """Hebrew Whisper's phrases as (start, end, text); loaded for this and freed after."""
        import gc

        from transformers import pipeline

        asr = pipeline("automatic-speech-recognition", ASR_MODEL, revision=ASR_REVISION,
                       dtype=torch.float16, device="cuda:0")
        out = asr({"raw": audio, "sampling_rate": sr}, return_timestamps=True,
                  generate_kwargs={"language": "he", "task": "transcribe"})
        del asr
        gc.collect()
        torch.cuda.empty_cache()
        duration = len(audio) / sr
        return [(c["timestamp"][0], c["timestamp"][1] or duration, c["text"].strip())
                for c in out["chunks"] if c["text"].strip()]

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
            audio = self.model.generate(**kwargs)[0]
        except ValueError:
            # Very short text (one word) can get a zero-length duration estimate; give it a floor.
            audio = self.model.generate(**kwargs, duration=max(1.0, 0.09 * len(text)))[0]
        return audio if voice in DESIGNED_VOICES else self._level(audio)

    @staticmethod
    def _level(audio):
        """A clone at CLONE_RMS, about Chatterbox's level, with peaks kept under PEAK_LIMIT."""
        import numpy as np

        rms, peak = float(np.sqrt(np.mean(audio**2))), float(np.abs(audio).max())
        if rms < 1e-4:
            return audio
        return audio * min(CLONE_RMS / rms, PEAK_LIMIT / peak)


if __name__ == "__main__":
    serve(OmniVoiceBackend(), "OmniVoice worker")
