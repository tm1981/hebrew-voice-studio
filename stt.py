"""Speech-to-text, for Home Assistant's voice assistant (over Wyoming, see wyoming_service.py)
and POST /v1/audio/transcriptions. The default model is ivrit-ai's Hebrew fine-tune of Whisper
large-v3-turbo (Apache-2.0, about 1.6 GB, downloaded into models/hf on first use); --stt-model
or STT_MODEL picks another Whisper model from Hugging Face, "off" turns it off.

Like the GPU engines it loads on its first request, unloads after the idle time set in the web
UI, and can be kept loaded (Server status), so a voice command doesn't wait for it to load.
"""
import gc
import threading
import time

import numpy as np

from engines import OnDemand

DEFAULT_MODEL = "ivrit-ai/whisper-large-v3-turbo"
DEFAULT_REVISION = "f33172a8c3c6efbc040a7200e00835257cac0447"
SAMPLE_RATE = 16000  # what Whisper takes
CHUNK_SAMPLES = 30 * SAMPLE_RATE  # Whisper's window; longer audio is read window by window
# A window quieter than this (RMS) is skipped: on silence Whisper makes up text ("תודה רבה").
SILENCE_RMS = 0.003
MAX_SECONDS = 15 * 60  # longer uploads are cut there


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
        self._generate(np.random.default_rng(0).normal(0, 0.05, SAMPLE_RATE).astype(np.float32), "he")  # warm-up

    def _unload(self) -> None:
        import torch

        del self.model, self.processor
        gc.collect()
        torch.cuda.empty_cache()

    def transcribe(self, audio: np.ndarray, language: str | None = "he") -> str:
        """Mono float32 audio at 16 kHz -> text. language: "he", "he-IL", "en"...; None = detect."""
        language = (language or "").split("-")[0].split("_")[0].lower() or None
        with self.lock:
            self.ensure_loaded()
            texts = []
            for start in range(0, max(len(audio), 1), CHUNK_SAMPLES):
                chunk = audio[start:start + CHUNK_SAMPLES]
                if len(chunk) < SAMPLE_RATE // 10 or np.sqrt(np.mean(chunk ** 2)) < SILENCE_RMS:
                    continue  # a sliver at the end, or silence
                texts.append(self._generate(chunk, language))
            self.last_used = time.monotonic()
        return " ".join(t for t in texts if t)

    def _generate(self, chunk: np.ndarray, language: str | None) -> str:
        import torch

        features = self.processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        with torch.inference_mode():
            ids = self.model.generate(features.to(self.device, self.dtype), language=language, task="transcribe")
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def decode_audio(data: bytes) -> np.ndarray:
    """An uploaded recording (wav, mp3, flac, ogg) -> mono float32 at 16 kHz, at most
    MAX_SECONDS. Only what libsndfile reads: no ffmpeg, which would follow playlists and URLs
    inside a crafted file."""
    import io

    import librosa
    import soundfile as sf

    try:
        with sf.SoundFile(io.BytesIO(data)) as f:
            if not (1 <= f.channels <= 8 and 1000 <= f.samplerate <= 192000):
                raise ValueError
            # frames= caps the decode even when the header lies about the length
            audio = f.read(frames=int(MAX_SECONDS * f.samplerate), dtype="float32", always_2d=True).mean(axis=1)
            rate = f.samplerate
    except Exception:
        raise ValueError("Couldn't read that audio file (wav, mp3, flac and ogg work).")
    return librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE) if rate != SAMPLE_RATE else audio
