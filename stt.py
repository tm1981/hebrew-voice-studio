"""Speech-to-text, for Home Assistant's voice assistant (over Wyoming, see wyoming_service.py)
and POST /v1/audio/transcriptions. The default model is ivrit-ai's Hebrew fine-tune of Whisper
large-v3-turbo (Apache-2.0, about 1.6 GB, downloaded into models/hf on first use); --stt-model
or STT_MODEL picks another Whisper model from Hugging Face, "off" turns it off.

Like the GPU engines it loads on its first request, unloads after the idle time set in the web
UI, and can be kept loaded (Server status), so a voice command doesn't wait for it to load.

On audio without speech (a false wake word, room noise) Whisper makes up text, usually
"תודה רבה", and this fine-tune never reports "no speech", so windows are checked with Silero VAD
(on the CPU, about 10 ms) first. Hint words (names of rooms, devices, people) are given to
Whisper as a prompt, so it spells them that way.
"""
import gc
import gzip
import re
import threading
import time

import numpy as np

from engines import OnDemand

DEFAULT_MODEL = "ivrit-ai/whisper-large-v3-turbo"
DEFAULT_REVISION = "f33172a8c3c6efbc040a7200e00835257cac0447"
SAMPLE_RATE = 16000  # what Whisper takes
CHUNK_SAMPLES = 30 * SAMPLE_RATE  # Whisper's window; longer audio is read window by window
SILENCE_RMS = 0.003  # a window quieter than this (RMS) is skipped without asking the VAD
MIN_SPEECH_SECONDS = 0.2  # a window with less speech than this (per the VAD) is skipped
# Text that compresses this well is a loop ("לכן, נכנס לספר את זה" again and again), which
# Whisper produces on loud noise; that window is dropped. OpenAI's Whisper uses the same test.
MAX_COMPRESSION_RATIO = 2.4
MAX_PROMPT_TOKENS = 200  # Whisper's limit is 223 for the prompt
MAX_SECONDS = 15 * 60  # longer uploads are cut there
# Direction marks Whisper sometimes puts in Hebrew text; invisible, but they stop Home
# Assistant from matching the sentence to a command.
BIDI_MARKS = re.compile("[‎‏‪-‮⁦-⁩]")


class SpeechToText(OnDemand):
    name = "stt"
    label = "Speech-to-text (Whisper)"

    def __init__(self, device: str, model: str = DEFAULT_MODEL):
        self.device = device
        self.model_id = model
        self.revision = DEFAULT_REVISION if model == DEFAULT_MODEL else None
        self.lock = threading.RLock()
        self.voices = {}  # listed with the engines in Server status, which count voices

    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration
        from transformers.utils import logging

        logging.get_logger("transformers.generation.utils").setLevel(logging.ERROR)  # per-request notices

        print(f"Loading speech-to-text ({self.model_id}) on {self.device}...", flush=True)
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id, revision=self.revision, dtype=self.dtype).to(self.device).eval()
        from silero_vad import load_silero_vad

        self.vad = load_silero_vad(onnx=True)
        self._generate(np.random.default_rng(0).normal(0, 0.05, SAMPLE_RATE).astype(np.float32), "he", None)  # warm-up

    def _unload(self) -> None:
        import torch

        del self.model, self.processor, self.vad
        gc.collect()
        torch.cuda.empty_cache()

    def transcribe(self, audio: np.ndarray, language: str | None = "he", hints: str = "") -> str:
        """Mono float32 audio at 16 kHz -> text. language: "he", "he-IL", "en"...; None = detect.
        hints: words to spell as given ("סלון, נאנוליף, יואב")."""
        language = (language or "").split("-")[0].split("_")[0].lower() or None
        with self.lock:
            self.ensure_loaded()
            prompt = self._prompt_ids(hints)
            texts = []
            for start in range(0, max(len(audio), 1), CHUNK_SAMPLES):
                chunk = audio[start:start + CHUNK_SAMPLES]
                if len(chunk) < SAMPLE_RATE // 10 or np.sqrt(np.mean(chunk ** 2)) < SILENCE_RMS:
                    continue  # a sliver at the end, or silence
                if self._speech_seconds(chunk) < MIN_SPEECH_SECONDS:
                    continue  # noise, music, a hum
                text = self._generate(chunk, language, prompt)
                if text and len(gzip.compress(text.encode())) * MAX_COMPRESSION_RATIO < len(text.encode()):
                    print(f"[stt] dropped a repeating transcript ({len(text)} characters)")
                    continue
                texts.append(text)
            self.last_used = time.monotonic()
        return " ".join(t for t in texts if t)

    def _speech_seconds(self, chunk: np.ndarray) -> float:
        import torch
        from silero_vad import get_speech_timestamps

        spans = get_speech_timestamps(torch.from_numpy(np.ascontiguousarray(chunk)), self.vad, sampling_rate=SAMPLE_RATE)
        return sum(s["end"] - s["start"] for s in spans) / SAMPLE_RATE

    def _prompt_ids(self, hints: str):
        import torch

        hints = " ".join(hints.split())
        if not hints:
            return None
        ids = self.processor.get_prompt_ids(hints)  # <|startofprev|> then the words' tokens
        return torch.tensor(ids[:MAX_PROMPT_TOKENS + 1], device=self.device)

    def _generate(self, chunk: np.ndarray, language: str | None, prompt) -> str:
        import torch

        features = self.processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        extra = {} if prompt is None else {"prompt_ids": prompt}
        with torch.inference_mode():
            ids = self.model.generate(features.to(self.device, self.dtype), language=language, task="transcribe", **extra)
        return BIDI_MARKS.sub("", self.processor.batch_decode(ids, skip_special_tokens=True)[0]).strip()


def decode_audio(data: bytes) -> np.ndarray:
    """An uploaded recording (wav, mp3, flac, ogg) -> mono float32 at 16 kHz, at most
    MAX_SECONDS. Only what libsndfile reads: no ffmpeg, which would follow playlists and URLs
    inside a crafted file."""
    import io

    import librosa
    import soundfile as sf

    try:
        with sf.SoundFile(io.BytesIO(data)) as f:
            if not (1 <= f.channels <= 2 and 8000 <= f.samplerate <= 48000):
                raise ValueError
            # Block by block, each mixed to mono, until MAX_SECONDS: memory stays bounded even
            # when the header lies about the length.
            rate, left, parts = f.samplerate, MAX_SECONDS * f.samplerate, []
            for block in f.blocks(blocksize=rate * 10, dtype="float32", always_2d=True):
                parts.append(block[:left].mean(axis=1))
                left -= len(parts[-1])
                if left <= 0:
                    break
            audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    except Exception:
        raise ValueError("Couldn't read that audio file (wav, mp3, flac and ogg work).")
    return librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE) if rate != SAMPLE_RATE else audio
