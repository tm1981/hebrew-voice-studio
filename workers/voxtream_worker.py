"""voxtream2-he worker (env 'voxtream'; in the Docker image).

flowty1/voxtream2-he (GPL-3.0): a Hebrew fine-tune of the streaming model Voxtream 2, 0.5B
parameters. It only clones: every voice is a recording in voices/. Text goes through the
library's own Hebrew phonemizer (ReNikud, with espeak-ng for English words).

Measured on an RTX 5090: about 4x real time, about 2.5 GB of GPU memory. Its card warns that
"outputs are not always stable", and testing agreed. What is done about it here:

- The model keeps state from one reading to the next, in two places. Its streaming state
  (reset_streaming_state, off by default) is reset for every reading. The other the library
  never resets: its "sink attention" keeps the first reading's prompt and a growing list of the
  phonemes and audio of every reading since (SinkAttention.reset() exists; generate_stream()
  doesn't call it). It comes into play once a reading is long, and from then on everything is
  read against leftovers: after one reading of 35 s, 8 of the next 10 sentences were wrong and
  stayed wrong (0 to 2 of 10 with the reset). That is the "fine for a few sentences, then it
  falls apart" this engine showed. The sink is reset before every take.
- Readings can still start with seconds of silence (one test voice: 12 of 15, up to 22 s).
  Silence is cut from both ends of each reading.
- "clean" runs Sidon, the library's prompt enhancer, on the recording first. It removes the
  recording's background noise from the clones; for one test voice it also brought the long
  silences. It is a setting, on by default.

- A reading that drags is read again (up to twice): its worst failures are long ones, babble
  or silence in the middle, at under 3 letters a second where good readings run at 5 to 19
  (median 12). That catches the failures heard most, about a quarter of all wrong readings;
  a dropped or swapped word in a reading of normal length is not caught. A take is also cut
  off once it is too long to be a good reading of its text, instead of running up to a minute.

Tried and not better (tools/container-test/voxtream_variants.py, before the sink reset): a
shorter clip that ends on a finished sentence, the same uncleaned, and more guidance.
"""
from pathlib import Path

import numpy as np
import torch

from worker_common import serve

SILENCE_DB = 35  # below the reading's loudest 20 ms by this much counts as silence
KEEP_BEFORE, KEEP_AFTER = 0.05, 0.12  # seconds of the trimmed silence left at each end
MIN_PACE = 4.5   # letters per second: a reading slower than this has gone wrong
MIN_LETTERS = 12  # shorter texts are too short to judge by pace
RETAKES = 2


def trim_silence(audio: np.ndarray, rate: int) -> np.ndarray:
    hop = int(rate * 0.02)
    if len(audio) < 2 * hop:
        return audio
    frames = audio[: len(audio) // hop * hop].reshape(-1, hop)
    level = np.sqrt((frames**2).mean(axis=1))
    loud = np.flatnonzero(level > level.max() * 10 ** (-SILENCE_DB / 20))
    if not len(loud):
        return audio
    start = max(loud[0] * hop - int(rate * KEEP_BEFORE), 0)
    return audio[start: (loud[-1] + 1) * hop + int(rate * KEEP_AFTER)]


class VoxtreamBackend:
    native_speed = False
    custom_voices = True
    sample_rate = 24000

    def __init__(self):
        self.voices: dict = {}   # name -> the recording's path
        self.prompts: dict = {}  # (name, clean) -> the prepared prompt, built when first used
        self.builtin_voices: set = set()

    def load(self):
        import soundfile as sf
        import torchaudio

        def load_with_soundfile(path, *_, **__):
            audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
            return torch.from_numpy(audio.T.copy()), rate

        # torchaudio 2.9 reads files through torchcodec, whose CUDA build can't load in this
        # image (it wants CUDA libraries on the system path). The voices are WAV files, which
        # soundfile reads; the readings scored the same either way.
        torchaudio.load = load_with_soundfile

        from voxtream2_he import SpeechGenerator

        print("Loading voxtream2-he...", flush=True)
        self.generator = SpeechGenerator(reset_streaming_state=True)
        self.phonemizer = self.generator.make_phonemizer()
        self.sample_rate = self.generator.config.mimi_sr

    def add_voice(self, name: str, path: Path):
        self.remove_voice(name)
        self.voices[name] = str(path)

    def remove_voice(self, name: str):
        self.voices.pop(name, None)
        for key in [k for k in self.prompts if k[0] == name]:
            del self.prompts[key]

    def synthesize(self, text: str, voice: str, speed: float, clean=True, seed=None, **_):
        key = (voice, bool(clean))
        if key not in self.prompts:
            self.prompts[key] = self.generator.prepare_prompt_from_path(self.voices[voice], enhance_with_sidon=bool(clean))
        phonemes = self.phonemizer.phonemize(text, language="he")
        letters = sum(c.isalpha() for c in text)
        best = np.zeros(0, dtype=np.float32)
        # No good reading of this text is longer than this; a take that gets there has gone wrong
        self.generator.config.max_audio_length_ms = int(min(60, max(letters, MIN_LETTERS) / MIN_PACE + 3) * 1000)
        for take in range(1 + RETAKES):
            self.generator.model.sink_attention.reset()  # see the top of this file
            if seed is not None:
                torch.manual_seed(seed + take)  # the same seed gives the same takes
            frames = [np.asarray(frame, dtype=np.float32) for frame, _ in self.generator.generate_stream(
                text=phonemes, prepared_prompt=self.prompts[key], language="he", normalize_text=False, input_is_phonemes=True)]
            audio = trim_silence(np.concatenate(frames), self.sample_rate) if frames else best[:0]
            if len(audio) and (not len(best) or len(audio) < len(best)):
                best = audio  # of takes that all drag, the shortest has the least babble
            if letters < MIN_LETTERS or (len(audio) and letters / (len(audio) / self.sample_rate) >= MIN_PACE):
                return audio
            print(f"voxtream2-he: take {take + 1} dragged ({len(audio) / self.sample_rate:.1f} s for {letters} letters)", flush=True)
        return best


if __name__ == "__main__":
    serve(VoxtreamBackend(), "voxtream2-he worker")
