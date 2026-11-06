"""A voice's reference clip with its exact words, for workers whose model needs a transcript
(Ming, Drama 3, Breeze 2). Their envs need torch, transformers, numpy and soundfile.

The clip is the start of the prepared recording, cut after the last finished sentence within
MAX_SECONDS, so it ends on a finished thought in speech rather than on silence or mid-sentence.
Models start their reading with the last words of a clip that doesn't: F5's base model did in
9 of 16 readings, and 0 of 16 with the clip cut like this. Ming returned 0.3 s of audio for a
clip ending in digital silence. Cached as voices/.cache/<name>.ref.wav and .ref.txt.
"""
import gc
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

MAX_SECONDS = 12
MIN_SECONDS = 4
SENTENCE_END = (".", "?", "!")


def _words(audio: np.ndarray, sample_rate: int) -> list:
    """Whisper's words with their times, in the language it hears."""
    import transformers
    from transformers import pipeline

    # Ming has to stay on transformers 4.46, which only knows the old name; 5.x only the new one
    version = tuple(int(n) for n in transformers.__version__.split(".")[:2])
    half = {"dtype" if version >= (4, 56) else "torch_dtype": torch.float16}
    asr = pipeline("automatic-speech-recognition", "openai/whisper-large-v3-turbo", device="cuda", **half)
    chunks = asr({"raw": audio, "sampling_rate": sample_rate}, return_timestamps="word",
                 generate_kwargs={"task": "transcribe"})["chunks"]
    del asr
    gc.collect()
    torch.cuda.empty_cache()
    return chunks


def reference(path: Path) -> tuple[Path, str]:
    """(clip, its words) for the prepared recording at `path`, transcribed once and cached."""
    path = Path(path)
    clip, text = path.with_suffix(".ref.wav"), path.with_suffix(".ref.txt")
    if text.exists() and clip.exists() and text.stat().st_mtime >= path.stat().st_mtime:
        return clip, text.read_text(encoding="utf-8").strip()
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sample_rate != 16000:
        hearing = np.interp(np.arange(0, len(audio), sample_rate / 16000), np.arange(len(audio)), audio)
    else:
        hearing = audio
    chunks = _words(hearing.astype(np.float32), 16000)
    if not chunks:
        raise RuntimeError(f"No speech heard in {path.name}")
    ends = [i for i, c in enumerate(chunks) if c["text"].rstrip().endswith(SENTENCE_END)
            and MIN_SECONDS <= (c["timestamp"][1] or 0) <= MAX_SECONDS]
    # The last word Whisper hears always gets a full stop, so it only counts as a sentence end
    # when there is nothing better.
    inner = [i for i in ends if i < len(chunks) - 1]
    if inner or ends:
        last = (inner or ends)[-1]
    else:  # one long sentence: the last word that fits
        last = max((i for i, c in enumerate(chunks) if (c["timestamp"][1] or 0) <= MAX_SECONDS), default=len(chunks) - 1)
    end = chunks[last]["timestamp"][1] or len(audio) / sample_rate
    sf.write(clip, audio[:int((end + 0.06) * sample_rate)], sample_rate)
    words = "".join(c["text"] for c in chunks[:last + 1]).strip()
    text.write_text(words, encoding="utf-8")
    return clip, words
