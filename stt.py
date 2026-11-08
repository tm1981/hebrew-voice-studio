"""Speech-to-text, for Home Assistant's voice assistant (over Wyoming, see wyoming_service.py)
and POST /v1/audio/transcriptions. The default model is ivrit-ai's Hebrew fine-tune of Whisper
large-v3-turbo (Apache-2.0, about 1.6 GB, downloaded into models/hf on first use). Settings in the
web UI switch to OpenAI's all-language large-v3-turbo (MIT, same size); --stt-model or STT_MODEL
picks any Whisper model from Hugging Face, "off" turns it off.

Like the GPU engines it loads on its first request, unloads after the idle time set in the web
UI, and can be kept loaded (Settings), so a voice command doesn't wait for it to load.

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
# The models Settings offers: id -> (languages, as told to Home Assistant; label).
BUILTIN_MODELS = {
    DEFAULT_MODEL: (["he"], "Hebrew: ivrit-ai Whisper large-v3 turbo"),
    "openai/whisper-large-v3-turbo": (
        "af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo fr gl gu ha haw he hi hr ht "
        "hu hy id is it ja jw ka kk km kn ko la lb ln lo lt lv mg mi mk ml mn mr ms mt my ne nl nn no oc pa pl "
        "ps pt ro ru sa sd si sk sl sn so sq sr su sv sw ta te tg th tk tl tr tt uk ur uz vi yi yo yue zh".split(),
        "All languages: OpenAI Whisper large-v3 turbo"),
}
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
        self.lock = threading.RLock()
        self.set_model(model)
        self.voices = {}  # listed with the engines in Settings, which count voices

    def set_model(self, model: str) -> None:
        """Switches to another Whisper model: the current one is unloaded (after a running
        transcription), the new one loads on the next request."""
        with self.lock:
            self.unload()
            self.model_id = model
            self.revision = DEFAULT_REVISION if model == DEFAULT_MODEL else None

    @property
    def languages(self) -> list[str]:
        return BUILTIN_MODELS.get(self.model_id, (["he"],))[0]

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

    def word_times(self, audio: np.ndarray, sample_rate: int, text: str) -> list[list[float]] | None:
        """When each word of `text` (split at spaces) is spoken in `audio`: [[start, end]] in
        seconds, for marking words as a reading plays. The words are known, so nothing is
        recognized: Whisper reads the text against the audio and its alignment heads (the
        cross-attention heads that follow the audio, listed in the model's generation config)
        give, through dynamic time warping, the frame where each token is heard. That's
        OpenAI's word-timestamp method, with the text given instead of guessed. None if the
        audio is too long for one window or the model has no alignment heads."""
        import torch

        words = text.split()
        heads = getattr(self.model_generation_config(), "alignment_heads", None)
        audio = resample_16k(audio, sample_rate)
        if not words or not heads or len(audio) > CHUNK_SAMPLES - SAMPLE_RATE // 2:
            return None
        with self.lock:
            self.ensure_loaded()
            tok = self.processor.tokenizer
            ids, owner = [], []
            for i, word in enumerate(words):
                pieces = tok.encode(" " + word, add_special_tokens=False)
                ids += pieces
                owner += [i] * len(pieces)
            language = script_language(text)
            prefix = tok.convert_tokens_to_ids(["<|startoftranscript|>", f"<|{language}|>", "<|transcribe|>", "<|notimestamps|>"])
            sequence = torch.tensor([prefix + ids + [tok.convert_tokens_to_ids("<|endoftext|>")]], device=self.device)
            features = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
            self.model.set_attn_implementation("eager")  # the fused kernels don't return attention weights
            try:
                with torch.inference_mode():
                    out = self.model(input_features=features.to(self.device, self.dtype), decoder_input_ids=sequence,
                                     output_attentions=True)
            finally:
                self.model.set_attn_implementation("sdpa")
            frames = max(1, len(audio) // FRAME_SAMPLES)
            weights = torch.stack([out.cross_attentions[layer][0, head] for layer, head in heads]).float()[:, :, :frames]
            weights = (weights - weights.mean(-2, keepdim=True)) / (weights.std(-2, keepdim=True) + 1e-6)
            # row k predicts token k+1: the rows from <|notimestamps|> on are the text tokens, then the end
            matrix = weights.mean(0)[len(prefix) - 1 : -1].cpu().numpy()
            self.last_used = time.monotonic()
        token_start = token_starts(matrix) / (SAMPLE_RATE / FRAME_SAMPLES)
        silences = silent_spans(audio)
        times = []
        for i in range(len(words)):
            first, last = owner.index(i), len(owner) - owner[::-1].index(i)
            start, end = float(token_start[first]), float(token_start[last])
            for s, e in silences:  # a pause in the first half of a word's span comes before the word
                if s < (start + end) / 2 and start < e < end:
                    start = e
            times.append([round(start, 2), round(max(end, start + 0.05), 2)])
        return times

    def model_generation_config(self):
        with self.lock:
            self.ensure_loaded()
            return self.model.generation_config

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


FRAME_SAMPLES = 320  # Whisper's encoder: one frame per 20 ms
_SCRIPT_HEBREW, _SCRIPT_ARABIC, _SCRIPT_LATIN = re.compile("[א-ת]"), re.compile("[ء-ي]"), re.compile("[A-Za-z]")


def script_language(text: str) -> str:
    """The language token for a sentence, by its letters: Hebrew unless it has none."""
    if _SCRIPT_HEBREW.search(text):
        return "he"
    if _SCRIPT_ARABIC.search(text):
        return "ar"
    return "en" if _SCRIPT_LATIN.search(text) else "he"


def resample_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate == SAMPLE_RATE:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE).astype(np.float32)


def token_starts(matrix: np.ndarray) -> np.ndarray:
    """Dynamic time warping through [tokens + end, frames] attention (high = heard there): the
    first frame of each row on the best monotonic path."""
    cost = -matrix
    rows, cols = cost.shape
    total = np.full((rows + 1, cols + 1), np.inf)
    total[0, 0] = 0
    step = np.zeros((rows + 1, cols + 1), dtype=np.int8)  # 0 diagonal, 1 from the row above, 2 from the left
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            options = (total[i - 1, j - 1], total[i - 1, j], total[i, j - 1])
            k = int(np.argmin(options))
            total[i, j], step[i, j] = cost[i - 1, j - 1] + options[k], k
    starts = np.zeros(rows)
    i, j = rows, cols
    while i > 0 and j > 0:
        starts[i - 1] = j - 1  # walking back, so the last write per row is its first frame
        k = step[i, j]
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return starts


def silent_spans(audio: np.ndarray, min_seconds: float = 0.12) -> list[tuple[float, float]]:
    """Stretches quieter than 2% of the loudest 20 ms frame, in seconds."""
    frames = len(audio) // FRAME_SAMPLES
    if frames == 0:
        return []
    rms = np.sqrt(np.mean(audio[: frames * FRAME_SAMPLES].reshape(frames, FRAME_SAMPLES) ** 2, axis=1))
    quiet = rms < rms.max() * 0.02
    spans, start = [], None
    for i, q in enumerate(np.append(quiet, False)):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * FRAME_SAMPLES / SAMPLE_RATE >= min_seconds:
                spans.append((start * FRAME_SAMPLES / SAMPLE_RATE, i * FRAME_SAMPLES / SAMPLE_RATE))
            start = None
    return spans


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
