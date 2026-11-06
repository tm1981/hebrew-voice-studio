"""Ming-3.0-tts 0.5B worker (conda env 'ming'; code in vendor/Ming-3.0-tts, tools/fetch_vendor.py).
inclusionAI's Qwen2.5-0.5B plus flow-matching speech model, MIT. English in this app: Chinese,
German, Spanish, French, Korean and Japanese also exist, Hebrew doesn't. Measured on Hebrew: the
Hebrew transcript gave 0.3 s of audio, a Latin one gave gibberish (Whisper, 3 of 3 sentences).

Voices:
  emma, george, charlotte - the model's built-in English voices (US_Emma, UK_George, UK_Charlotte)
  <custom>                - cloned from a recording in voices/; its first reading cuts the clip and
                            transcribes it with Whisper (reference.py), then both are cached

Without a seed every take is a little different; the same seed gives the same reading.

Measured on an RTX 5090, 16 readings per voice, Whisper word check: no wrong words for the built-in
voice or a clone made from Hebrew speech, 2 of 16 for a second clone. About 1.1 s to generate each
second of audio, about 3.7 GB of GPU memory, 44.1 kHz. transformers has to stay at 4.46.3: on 4.57.3
the model's stop signal never fires and every reading runs to about 28 s.
"""
import random
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from worker_common import serve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor" / "Ming-3.0-tts"))  # its modules import each other at top level

REPO, REVISION = "inclusionAI/Ming-3.0-tts-0.5B", "7247f4c94a5b2e987404ed28a5cfec41b1a4ca61"
BUILTIN = {"emma": "US_Emma", "george": "UK_George", "charlotte": "UK_Charlotte"}
QUIET_TAIL = 10 ** (-32 / 20)  # against the loudest moment of the clip
MIN_SECONDS_PER_LETTER = 0.04  # normal reading here is about 0.07


def _read_clip(path, **_):
    """Replaces torchaudio.load: torchaudio 2.9's needs torchcodec, which the app's envs leave out.
    Cuts a quiet tail, because the model takes one as the reading being over. One test clip with
    0.7 s of room tone: 7 of 16 readings lost words or stopped at 0.3 s; cut: 0 of 16. Silence
    added to it: 9 of 16 at 0.1 s, 16 of 16 at 0.3 s. Whisper's last word time lands in that tail."""
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    frame = int(sample_rate * 0.05)
    mono = audio.mean(axis=1)
    levels = np.sqrt((mono[:len(mono) // frame * frame].reshape(-1, frame) ** 2).mean(axis=1))
    loud = np.nonzero(levels > levels.max() * QUIET_TAIL)[0]
    return torch.from_numpy(audio[:(loud[-1] + 1) * frame].T.copy()), sample_rate


class MingBackend:
    native_speed = False  # only slow/medium/fast, so the server stretches the audio
    custom_voices = True
    sample_rate = 44100

    def __init__(self):
        self.voices: dict = dict(BUILTIN)
        self.builtin_voices = set(BUILTIN)

    def load(self):
        from huggingface_hub import snapshot_download
        from ming_tts import MingTTS

        torchaudio.load = _read_clip
        print(f"Loading {REPO}...", flush=True)
        # The prompt's encoding draws random noise: cached, a seed would read differently the
        # first time a voice is used than afterwards.
        self.model = MingTTS(snapshot_download(REPO, revision=REVISION), cache_prompts=False)
        self.sample_rate = self.model.sample_rate
        self.synthesize("Hello, how are you today?", "emma", 1.0)  # warm-up: the text normalizer takes about 15 s to wake

    def add_voice(self, name: str, path: Path):
        self.voices[name] = Path(path)

    def remove_voice(self, name: str):
        self.voices.pop(name, None)

    def synthesize(self, text: str, voice: str, speed: float, seed=None, guidance=2.0, temperature=0.0, direction="", **_):
        seed = random.randrange(2 ** 31) if seed is None else int(seed)
        source = self.voices[voice]
        if isinstance(source, str):  # the model refuses direction on a built-in voice
            return self._read(text, seed, guidance, temperature, task="IP", controls={"ip": source}, use_zero_spk_emb=True)
        from reference import reference

        clip, words = reference(source)
        voicing = dict(prompt_wav_path=str(clip), use_spk_emb=True)
        if direction.strip():
            # With the transcript as well, 3 of 6 directed readings stopped at 0.3 s; without it, 0 of 6.
            return self._read(text, seed, guidance, temperature, task="FREEFORM", instruction=direction.strip(), **voicing)
        audio = self._read(text, seed, guidance, temperature, task="TTS", prompt_text=words, **voicing)
        letters = sum(c.isalpha() for c in re.sub(r"\[[^\]]*\]", "", text))  # [Laughter] and the like aren't read
        if len(audio) / self.sample_rate < letters * MIN_SECONDS_PER_LETTER:
            # Without the transcript no reading stopped early (0 of 32), but the voice is a little
            # further off (0.73 against 0.79 on one clone) and it brings more room sound.
            audio = self._read(text, seed, guidance, temperature, task="TTS", **voicing)
        return audio

    def _read(self, text, seed, guidance, temperature, **voicing):
        random.seed(seed)
        np.random.seed(seed % 2 ** 32)
        torch.manual_seed(seed)
        audio = self.model.synthesize(text=text, language_id="English", cfg=float(guidance),
                                      temperature=float(temperature), **voicing)
        return audio.squeeze().numpy()


if __name__ == "__main__":
    serve(MingBackend(), "Ming-3.0-tts worker")
