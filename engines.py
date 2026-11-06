"""TTS engines behind one interface.

In-process (conda env 'voice'):  LightBlue, Blue 2.5, Kokoro (CPU, ONNX) and Chatterbox (GPU).
Workers in their own env:          OmniVoice, Higgs, QwenTTS-he, Magpie, F5-TTS and VibeVoice (English),
                                   see workers/. Their
                                   dependencies conflict with Chatterbox, so each runs as a
                                   small HTTP server started by RemoteEngine.
GPU engines (Chatterbox and the workers) start unloaded and can be unloaded again; see OnDemand.
On another machine (optional):     Zonos-Hebrew (Docker, Gradio API at ZONOS_URL / --zonos-url).

Every engine exposes:
    name, label, sample_rate, voices (name -> voice data), builtin_voices
    options: request settings it uses (beyond text, voice, speed)
    custom_voices: whether it can build a voice from a recording
    synthesize(text, voice, speed, **options) -> float32 mono numpy array
    diacritize(text) -> text with niqqud, as this engine will read it (uses_niqqud=False:
                        the engine reads plain text and this returns it unchanged)
    add_voice(name, audio_path) / remove_voice(name)

Custom voices live in voices/, as one recording (voices/<name>.wav) or a folder of a few
recordings of the same speaker (voices/<name>/*.wav). Each is prepared once into
voices/.cache/<name>.wav (silence trimmed, clips joined), which is what engines clone from;
LightBlue also averages a style per clip. Engines cache what they build next to it
(<name>.pt Chatterbox, <name>.lightblue.json LightBlue, <name>.omnivoice.pt OmniVoice).
"""
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
import soundfile as sf
import torch

from niqqud import diacritize_respecting_typed, strip_niqqud

from model_files import ensure as ensure_model
from paths import DATA_DIR, ROOT

VOICES_DIR = DATA_DIR / "voices"
VOICE_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
PAUSE_SECONDS = 0.25
HEBREW_LETTER = re.compile(r"[א-ת]")
LANG_TAG = re.compile(r"<[a-z]{2}>")
# A run of Latin-script words ("Claude Code", "rock'n'roll", "e-mail"), joined by spaces,
# commas, colons or hyphens. Digits alone stay Hebrew; text_prep has already written numbers
# in English sentences as English words, so they join the run.
LATIN_RUN = re.compile(r"[A-Za-z][\w'’\-]*(?:[ \t,:\-]+[A-Za-z][\w'’\-]*)*")
# Each worker's Python env: <VOICE_ENVS>/<env>, conda's envs folder by default (the Docker
# image sets VOICE_ENVS=/opt/envs).
ENVS_DIR = Path(os.environ.get("VOICE_ENVS") or Path.home() / ".conda" / "envs")


PREPARED_DIR = VOICES_DIR / ".cache"
PREPARED_SAMPLE_RATE = 44100
CLIP_GAP_SECONDS = 0.3
TRIM_TOP_DB = 35  # quieter than peak by this much counts as silence
# Clips are leveled before cloning: a quiet home recording made quiet clones (OmniVoice copies
# the reference's level), and clips of one voice recorded at different levels now match.
CLIP_RMS, CLIP_PEAK = 0.1, 0.95  # about -20 dBFS, peaks kept under -0.5 dB


def is_audio(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VOICE_EXTENSIONS


def voice_clips(name: str) -> list[Path]:
    """The recordings behind a custom voice: voices/<name>/* or the single voices/<name>.<ext>."""
    folder = VOICES_DIR / name
    if folder.is_dir():
        return sorted(p for p in folder.iterdir() if is_audio(p))
    return sorted(p for p in VOICES_DIR.glob(f"{name}.*") if is_audio(p))[:1]


def custom_voice_names() -> list[str]:
    VOICES_DIR.mkdir(exist_ok=True)
    names = {p.stem for p in VOICES_DIR.iterdir() if is_audio(p)}
    names |= {p.name for p in VOICES_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")}
    return sorted(n for n in names if voice_clips(n))


def load_trimmed(path: Path, sample_rate: int = PREPARED_SAMPLE_RATE) -> np.ndarray:
    """Mono audio with leading/trailing silence cut (zero-shot engines read the start of a
    clip, so silence there is wasted reference), leveled to CLIP_RMS."""
    import librosa

    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    trimmed, _ = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)
    return level(trimmed if trimmed.size else audio, CLIP_RMS, CLIP_PEAK)


def level(audio: np.ndarray, rms: float, peak: float) -> np.ndarray:
    """audio scaled to the given RMS, or less where that would push a peak over `peak`."""
    now, top = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0, float(np.abs(audio).max()) if audio.size else 0.0
    if now < 1e-4:
        return audio
    return (audio * min(rms / now, peak / top)).astype(np.float32)


def is_fresh(cache: Path, *sources: Path) -> bool:
    return cache.exists() and all(cache.stat().st_mtime >= s.stat().st_mtime for s in sources)


def prepared_clip(name: str) -> Path:
    """voices/.cache/<name>.wav: the voice's clips trimmed and joined, rebuilt when a clip changes."""
    clips = voice_clips(name)
    if not clips:
        raise FileNotFoundError(f"No recordings for voice '{name}'")
    out = PREPARED_DIR / f"{name}.wav"
    if not is_fresh(out, *clips):
        PREPARED_DIR.mkdir(parents=True, exist_ok=True)
        gap = np.zeros(int(PREPARED_SAMPLE_RATE * CLIP_GAP_SECONDS), dtype=np.float32)
        parts = []
        for clip in clips:
            parts += [load_trimmed(clip), gap]
        sf.write(out, np.concatenate(parts[:-1]), PREPARED_SAMPLE_RATE)
    return out


def check_recording(path: Path) -> dict:
    """Rough quality check of a voice recording: length, clipping, level and background noise
    (speech level vs. the quietest frames). Warnings are in Hebrew for the UI."""
    import librosa

    audio, sr = librosa.load(str(path), sr=16000, mono=True)
    seconds = len(audio) / sr
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    clipped = float(np.mean(np.abs(audio) > 0.98)) if audio.size else 0.0
    rms = librosa.feature.rms(y=audio, frame_length=480, hop_length=240)[0] if audio.size else np.zeros(1)
    noise, speech = np.percentile(rms, 10), np.percentile(rms, 90)
    snr = float(20 * np.log10(max(speech, 1e-6) / max(noise, 1e-6)))
    warnings, codes = [], []  # codes let the UI show the warnings in its own language
    if seconds < 6:
        codes.append("short")
        warnings.append(f"קצר מדי ({seconds:.0f} שניות): עדיף 10 עד 15 שניות של דיבור.")
    elif seconds > 30:
        codes.append("long")
        warnings.append(f"ארוך ({seconds:.0f} שניות): רוב המנועים משתמשים רק בתחילת ההקלטה.")
    if clipped > 0.001:
        codes.append("clipped")
        warnings.append("יש עיוות (הקלטה חזקה מדי). הרחיקו מעט את המיקרופון או הנמיכו את עוצמת הקלט.")
    if peak < 0.1:
        codes.append("quiet")
        warnings.append("ההקלטה שקטה מאוד. התקרבו למיקרופון.")
    if snr < 20:
        codes.append("noisy")
        warnings.append("יש רעש רקע משמעותי. הקליטו בחדר שקט.")
    return {"seconds": round(seconds, 1), "peak": round(peak, 3), "clipped_percent": round(clipped * 100, 2),
            "snr_db": round(snr, 1), "warnings": warnings, "codes": codes, "ok": not warnings}


def remove_voice_files(name: str) -> None:
    import shutil

    folder = VOICES_DIR / name
    if folder.is_dir():
        shutil.rmtree(folder)
    for path in [*VOICES_DIR.glob(f"{name}.*"), *PREPARED_DIR.glob(f"{name}.*")]:
        path.unlink()


def load_custom_voices(engine) -> None:
    """One unreadable recording skips that voice, not the whole engine."""
    for name in custom_voice_names():
        try:
            engine.add_voice(name, prepared_clip(name))
        except Exception as e:
            print(f"[WARN] {engine.label}: skipped voice '{name}': {e}")


def split_sentences(text: str, max_chars: int = 200) -> list[str]:
    """Sentence-sized chunks; autoregressive engines generate short inputs more reliably."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?؟…\n])\s+", text.strip()) if s.strip()]
    chunks: list[str] = []
    for sentence in sentences:
        while len(sentence) > max_chars:
            cut = max(sentence.rfind(",", 0, max_chars), sentence.rfind("،", 0, max_chars))
            if cut <= 0:
                cut = sentence.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            chunks.append(sentence[: cut + 1].strip())
            sentence = sentence[cut + 1 :].strip()
        if sentence:
            chunks.append(sentence)
    return chunks


def join_with_pauses(parts: list[np.ndarray], sample_rate: int) -> np.ndarray:
    pause = np.zeros(int(sample_rate * PAUSE_SECONDS), dtype=np.float32)
    out: list[np.ndarray] = []
    for part in parts:
        out += [part.astype(np.float32), pause]
    return np.concatenate(out[:-1]) if out else np.zeros(0, dtype=np.float32)


# Per-engine settings (the web UI's Advanced panel). Each engine lists specs; the UI draws a
# control per spec (its label and hint come from the UI's translations, "opt.<key>") and the
# server passes the cleaned values to engine.synthesize() as keyword arguments. The defaults
# are the engines' settings before these options existed, so an untouched panel changes nothing.
def rng(key: str, default: float, lo: float, hi: float, step: float) -> dict:
    return {"key": key, "type": "range", "default": default, "min": lo, "max": hi, "step": step}


def choice(key: str, choices: list[str], default: str = "") -> dict:
    return {"key": key, "type": "select", "default": default, "choices": choices}


def toggle(key: str, default: bool = False) -> dict:
    return {"key": key, "type": "toggle", "default": default}


def text(key: str, default: str = "", limit: int = 300) -> dict:
    return {"key": key, "type": "text", "default": default, "limit": limit}


SEED = {"key": "seed", "type": "seed", "default": None}  # None: random (or the engine's fixed seed)


def clean_options(specs: list[dict], values: dict) -> dict:
    """Every spec's value: from `values` when valid (clamped, cast), else the default."""
    out = {}
    for spec in specs:
        key, value, default = spec["key"], values.get(spec["key"]), spec["default"]
        try:
            if spec["type"] == "range":
                value = min(max(float(value), spec["min"]), spec["max"]) if value is not None else default
                if isinstance(spec["step"], int):
                    value = int(round(value))
            elif spec["type"] == "select":
                value = value if value in spec["choices"] else default
            elif spec["type"] == "toggle":
                value = bool(value) if value is not None else default
            elif spec["type"] == "text":
                value = str(value).strip()[:spec["limit"]] if value is not None else default
            elif spec["type"] == "seed":
                value = int(value) if value not in (None, "", -1) else None
                value = value if value is None or 0 <= value < 2**31 else None
        except (TypeError, ValueError):
            value = default
        out[key] = value
    return out


class OnDemand:
    """A GPU engine that starts unloaded and can be unloaded to free its memory; the next request
    loads it (20-60 s, the biggest about 3 minutes). Subclasses hold self.lock (an RLock) while they work, so unloading never cuts
    a running job, and implement _load() / _unload(). Voice names stay listed while unloaded."""

    gpu = True
    loaded = False
    last_used = 0.0

    def ensure_loaded(self) -> None:
        """Call with self.lock held."""
        if not self.loaded:
            self._load()
            self.loaded = True
        self.last_used = time.monotonic()

    def unload(self, wait: bool = True) -> bool:
        """Returns False when the engine is busy and wait is False."""
        if not self.lock.acquire(blocking=wait):
            return False
        try:
            if self.loaded:
                print(f"Unloading {self.label} to free GPU memory")
                self._unload()
                self.loaded = False
        finally:
            self.lock.release()
        return True


class ChatterboxEngine(OnDemand):
    name = "chatterbox"
    label = "Chatterbox"
    option_specs = [
        rng("exaggeration", 0.5, 0.25, 2.0, 0.05),  # below 0.25 is outside Chatterbox's own range
        rng("cfg_weight", 0.5, 0.0, 1.0, 0.05),
        rng("temperature", 0.4, 0.1, 1.5, 0.05),
        SEED,
    ]
    custom_voices = True
    uses_niqqud = True
    sample_rate = 24000  # the model's, needed before it loads

    def __init__(self, device: str):
        from chatterbox.mtl_tts import Conditionals

        from chatterbox_fixes import is_arabic, load_dicta, speech_token_ratio

        self._Conditionals = Conditionals
        self._speech_token_ratio = speech_token_ratio
        self._is_arabic = is_arabic
        self.device = device
        self.dicta = load_dicta()  # CPU; stays when the model is unloaded
        self.lock = threading.RLock()
        self.voices = dict.fromkeys(["default", *custom_voice_names()])
        self.builtin_voices = {"default"}

    def _load(self) -> None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        import chatterbox_fast
        from chatterbox_fixes import enable_speed_control, guard_speech_tokens, optimize_t3

        print(f"Loading Chatterbox Multilingual on {self.device}...")
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        guard_speech_tokens(self.model)
        chatterbox_fast.install(self.model.t3)  # before optimize_t3, which wraps it
        optimize_t3(self.model)
        enable_speed_control(self.model)
        self.sample_rate = self.model.sr
        self.voices = {"default": self.model.conds}
        self.loaded = True  # so the voices below are built now
        load_custom_voices(self)
        self.model.generate("שלום", language_id="he")  # warm-up so the first request isn't slow

    def _unload(self) -> None:
        import gc

        self.model = None
        self.voices = dict.fromkeys(self.voices)  # the conditionals are GPU tensors too
        gc.collect()
        torch.cuda.empty_cache()

    def add_voice(self, name: str, audio_path: Path) -> None:
        cache = audio_path.with_suffix(".pt")
        with self.lock:
            if not self.loaded:
                self.voices[name] = None  # built when the model loads again
                return
            if is_fresh(cache, audio_path):
                self.voices[name] = self._Conditionals.load(cache, map_location=self.device).to(self.device)
                return
            saved = self.model.conds
            try:
                self.model.prepare_conditionals(str(audio_path))
                conds = self.model.conds
            finally:
                self.model.conds = saved
            conds.save(cache)
            self.voices[name] = conds

    def remove_voice(self, name: str) -> None:
        with self.lock:
            self.voices.pop(name, None)

    def diacritize(self, text: str) -> str:
        return self.dicta.add_diacritics(text)

    def synthesize(self, text, voice, speed, language="he", exaggeration=0.5, cfg_weight=0.5, temperature=0.4,
                   seed=None, **_):
        parts: list[np.ndarray] = []
        with self.lock:
            self.ensure_loaded()
            self.model.conds = self.voices[voice]
            self.model.speech_speed = speed
            for chunk in split_sentences(text):
                self.model.t3.speech_ratio = self._speech_token_ratio(chunk)
                if seed is not None:
                    torch.manual_seed(seed)
                wav = self.model.generate(
                    chunk,
                    language_id="ar" if language == "he" and self._is_arabic(chunk) else language,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )
                parts.append(wav.squeeze(0).cpu().numpy())
            self.last_used = time.monotonic()  # idle time counts from the end of a long job
        return join_with_pauses(parts, self.sample_rate)


class LightBlueEngine:
    name = "lightblue"
    label = "LightBlue"
    custom_voices = True
    uses_niqqud = True
    onnx_dir = ROOT / "models" / "blue-onnx-v2"
    # Local path: RenikudPlus's auto-download also fetches a datastore.json that is
    # missing from its Hugging Face repo, and fails.
    renikud_model = ROOT / "models" / "renikud-plus" / "model.onnx"
    repo = ROOT / "vendor" / "BlueTTS"
    voices_dir = repo / "voices"
    total_steps = 5
    pace_blend = None  # the runtime's own choice
    steps_range = (2, 32)
    cfg_scale = 4.0

    @property
    def option_specs(self) -> list[dict]:
        return [rng("steps", self.total_steps, *self.steps_range, 1), rng("guidance", self.cfg_scale, 1.0, 8.0, 0.5), SEED]

    onnx_source = "blue-onnx-v2"  # model_files.SOURCES key for onnx_dir

    def _fetch_models(self) -> None:
        ensure_model(self.onnx_source, f"{self.onnx_dir.name}/tts.json")
        ensure_model("renikud-plus", "renikud-plus/model.onnx")

    def __init__(self):
        if not sys.flags.utf8_mode:
            raise RuntimeError("LightBlue reads its files as UTF-8; run Python with -X utf8 or PYTHONUTF8=1.")
        self._fetch_models()
        sys.path.insert(0, str(self.repo))
        from src.blue_onnx import Style, limit_peak, load_text_to_speech, load_voice_style
        from src.blue_onnx.style import export_voice_style, style_from_payload

        self._limit_peak = limit_peak
        self._export_voice_style = export_voice_style
        self._style_from_payload = style_from_payload
        self._style = Style
        self._load_text_to_speech = load_text_to_speech
        print(f"Loading {self.label} (CPU)...")
        self.tts = self._load_tts()
        # RenikudPlus ignores niqqud in its input by default; "use" makes typed niqqud
        # decide the reading (per word), and leaves unpointed text as before.
        from renikud_onnx import G2P

        self.tts.g2p.renikud = G2P(str(self.renikud_model), niqqud="use")
        self.sample_rate = self.tts.sample_rate
        self.lock = threading.Lock()
        self.voices = {p.stem: load_voice_style([str(p)]) for p in sorted(self.voices_dir.glob("*.json"))}
        self.builtin_voices = set(self.voices)
        if self.custom_voices:
            load_custom_voices(self)
        self.synthesize("שלום", next(iter(self.voices)), 1.0)  # warm-up; also loads the G2P model

    def _load_tts(self):
        return self._load_text_to_speech(
            onnx_dir=str(self.onnx_dir),
            config_path=str(self.onnx_dir / "tts.json"),
            renikud_path=str(self.renikud_model),
        )

    def add_voice(self, name: str, audio_path: Path) -> None:
        """LightBlue only reads the first few seconds of a clip, so joined clips would waste
        all but the first: build a style per clip and average them (its documented blending)."""
        cache = audio_path.with_suffix(".lightblue.json")
        if is_fresh(cache, audio_path):
            payload = json.loads(cache.read_text(encoding="utf-8"))
        else:
            clips = voice_clips(name)
            sources = [audio_path]
            if len(clips) > 1:
                sources = []
                for i, clip in enumerate(clips):
                    part = PREPARED_DIR / f"{name}.part{i}.wav"
                    sf.write(part, load_trimmed(clip), PREPARED_SAMPLE_RATE)
                    sources.append(part)
            payloads = [
                self._export_voice_style(str(s), onnx_dir=str(self.onnx_dir), config=str(self.onnx_dir / "tts.json"))
                for s in sources
            ]
            payload = {
                key: {
                    "data": np.mean([np.asarray(p[key]["data"], dtype=np.float32) for p in payloads], axis=0).tolist(),
                    "dims": payloads[0][key]["dims"],
                }
                for key in ("style_ttl", "style_dp")
            }
            cache.write_text(json.dumps(payload), encoding="utf-8")
        with self.lock:
            self.voices[name] = self._style_from_payload(payload)

    def remove_voice(self, name: str) -> None:
        with self.lock:
            self.voices.pop(name, None)

    def diacritize(self, text: str) -> str:
        return self.tts.g2p._load_renikud().vocalize(text)

    # Text with no Hebrew: LightBlue's lang="en" mode reads short sentences unevenly ("Good
    # night." at a third of the usual loudness, now and then almost silent), while the same
    # text marked <en> in Hebrew mode is steady and as clear (Whisper: 20/20 vs 19/20 exact).
    english_as_tagged_hebrew = True

    def tag_languages(self, text: str, language: str) -> tuple[str, str]:
        """LightBlue's Hebrew G2P can't read Latin script; English must be marked
        <en>...</en> so it goes through espeak."""
        if language != "he" or LANG_TAG.search(text):
            return text, language
        if not HEBREW_LETTER.search(text):
            return (f"<en>{text}</en>", language) if self.english_as_tagged_hebrew else (text, "en")
        return LATIN_RUN.sub(lambda m: f"<en>{m.group(0)}</en>", text), language

    @staticmethod
    def pad_ending(text: str) -> str:
        """LightBlue's duration predictor gives a sentence the same length for every voice, so a
        voice that speaks slower (common with clones) gets its last word squeezed or cut.
        A trailing "..." buys the ending room; the silence it adds is trimmed afterwards.
        Whisper check on cloned voices: last word wrong in 6 of 69 sentences before, 1 after."""
        text = text.rstrip()
        return text if text.endswith(("...", "…")) else text.rstrip(".") + "..."

    def trim_tail(self, audio: np.ndarray) -> np.ndarray:
        loud = np.flatnonzero(np.abs(audio) > 0.01 * (np.max(np.abs(audio)) or 1))
        return audio[: loud[-1] + int(self.sample_rate * 0.08)] if loud.size else audio

    def synthesize(self, text, voice, speed, language="he", steps=None, guidance=None, seed=None, **_):
        text, language = self.tag_languages(self.pad_ending(text), language)
        with self.lock:
            if seed is not None:
                np.random.seed(seed)  # the flow starts from np.random noise
            audio = self._generate(text, language, voice, speed, steps or self.total_steps, guidance or self.cfg_scale)
        audio = audio[0] if audio.ndim == 2 else audio
        return self._limit_peak(self.trim_tail(audio)).astype(np.float32)

    def _generate(self, text, language, voice, speed, steps, guidance) -> np.ndarray:
        """Called with the lock held."""
        audio, _ = self.tts(
            text,
            lang=language,
            style=self.voices[voice],
            total_step=steps,
            cfg_scale=guidance,
            speed=speed,
            pace_blend=self.pace_blend,
            silence_duration=PAUSE_SECONDS,
        )
        return audio


class BlueTTS25Engine(LightBlueEngine):
    """BlueTTS v2.5, the newer model from LightBlue's author (same 44.1 kHz), with he/en/de/it/es
    in the training mix (ru/yi need front ends we don't have). Same runtime code as LightBlue
    (vendor/BlueTTS), separate ONNX bundle.

    Cloning: the bundle has no audio encoder, so a recording goes through the BlueCodec
    encoder (pip package `bluecodec`, weights in models/blue-codec; the model the bundle was
    exported with). Its latents, normalized with the bundle's stats, feed reference_encoder
    for the voice and duration_predictor (the reference-audio form) for its pacing; there is
    no ONNX that turns a recording into the style_dp the shipped voices carry. Check: speech
    from each shipped voice, encoded back, is closest to its own style (cosine 0.66-0.86,
    others 0.01-0.44)."""

    name = "blue25"
    label = "Blue 2.5"
    english_as_tagged_hebrew = False  # trained on English too; its English voices read it natively
    codec_dir = ROOT / "models" / "blue-codec"
    onnx_dir = ROOT / "models" / "bluetts-2.5"
    onnx_source = "bluetts-2.5"
    voices_dir = onnx_dir / "voices"
    total_steps = 16  # the bundle's documented settings
    steps_range = (5, 16)  # as its own demo
    cfg_scale = 3.0
    max_ref_frames = 150  # what the runtime gives reference_encoder (~10 s)
    # The duration model gives a sentence nearly the same length for every voice (within 2%,
    # from style_dp or from the recording), and cloned voices speak slower, so their last word
    # got squeezed or dropped. Whisper, 4 cloned voices x 8 sentences: last word wrong 5/32 at
    # 1.0, 5/32 at 1.15, 0/32 at 1.25 (words heard 89-98% -> 98-100%). The tail trim was not it.
    clone_duration_scale = 1.25
    # As the bundle's own demo: long sentences rush without it. Whisper, 4 voices x 20 sentences:
    # words added or dropped in 3/80 -> 0/80, a word misheard in 4/80 -> 2/80.
    pace_blend = 0.30

    def __init__(self):
        self._fetch_models()  # the stats below come from the bundle
        self.z_refs: dict[str, np.ndarray] = {}  # cloned voice -> normalized codec latents
        self._codec = None
        stats = np.load(self.onnx_dir / "stats.npz")
        self._mean, self._std = stats["mean"], stats["std"]
        self._scale = float(stats["normalizer_scale"][0])
        self._ref_encoder = ort.InferenceSession(str(self.onnx_dir / "reference_encoder.onnx"), providers=["CPUExecutionProvider"])
        self._durations = _DurationRouter(
            ort.InferenceSession(str(self.onnx_dir / "duration_predictor.onnx"), providers=["CPUExecutionProvider"]),
            self.clone_duration_scale,
        )
        super().__init__()

    def _load_tts(self):
        tts = super()._load_tts()
        self._durations.style = tts.dp_ort
        tts.dp_ort = self._durations
        return tts

    def _latents(self, audio: np.ndarray) -> np.ndarray:
        """44.1 kHz mono -> normalized latents [1, 144, T], the tail trimmed as the runtime does."""
        import torch
        from bluecodec.utils import compress_latents

        if self._codec is None:
            from bluecodec import BlueCodec

            ensure_model("blue-codec", "blue-codec/model.safetensors")
            self._codec = BlueCodec.from_pretrained(str(self.codec_dir))
        with torch.no_grad():
            mel = self._codec.mel_transform(torch.from_numpy(audio.astype(np.float32))[None])
            mel = mel[..., : mel.shape[-1] // 6 * 6]
            z = compress_latents(self._codec.encoder(mel), factor=6).numpy()
        z = (z - self._mean) / self._std * self._scale
        t = z.shape[2]
        return z[:, :, : max(1, t - max(2, int(t * 0.05)))].astype(np.float32)

    def add_voice(self, name: str, audio_path: Path) -> None:
        """Each clip gives a style (averaged, as LightBlue does); the clips' latents, joined,
        give the pacing."""
        cache = audio_path.with_suffix(f".{self.name}.json")
        if is_fresh(cache, audio_path):
            payload = json.loads(cache.read_text(encoding="utf-8"))
        else:
            latents = [self._latents(load_trimmed(clip)) for clip in voice_clips(name) or [audio_path]]
            styles = [
                self._ref_encoder.run(None, {"z_ref": z[:, :, : self.max_ref_frames], "mask": np.ones((1, 1, min(z.shape[2], self.max_ref_frames)), np.float32)})[0]
                for z in latents
            ]
            style_ttl = np.mean(styles, axis=0)
            z_ref = np.concatenate(latents, axis=2)[:, :, : self.max_ref_frames]
            payload = {
                "style_ttl": {"data": style_ttl.tolist(), "dims": list(style_ttl.shape)},
                "z_ref": {"data": z_ref.tolist(), "dims": list(z_ref.shape)},
            }
            cache.write_text(json.dumps(payload), encoding="utf-8")
        style_ttl, z_ref = (np.asarray(payload[k]["data"], np.float32).reshape(payload[k]["dims"]) for k in ("style_ttl", "z_ref"))
        with self.lock:
            # style_dp is unused for these voices (_DurationRouter); the runtime only needs its shape.
            self.voices[name] = self._style(style_ttl, np.zeros((1, 8, 16), np.float32))
            self.z_refs[name] = z_ref

    def remove_voice(self, name: str) -> None:
        with self.lock:
            self.voices.pop(name, None)
            self.z_refs.pop(name, None)

    def _generate(self, text, language, voice, speed, steps, guidance) -> np.ndarray:
        self._durations.z_ref = self.z_refs.get(voice)
        return super()._generate(text, language, voice, speed, steps, guidance)


class _DurationRouter:
    """Stands in for Blue 2.5's duration model: shipped voices use the style_dp form, cloned
    voices (z_ref set) the reference-audio form."""

    def __init__(self, by_reference, clone_scale=1.0):
        self.by_reference = by_reference
        self.clone_scale = clone_scale
        self.style = None
        self.z_ref = None

    def run(self, outputs, feed):
        if self.z_ref is None:
            return self.style.run(outputs, feed)
        duration, *rest = self.by_reference.run(outputs, {
            "text_ids": feed["text_ids"],
            "z_ref": self.z_ref,
            "text_mask": feed["text_mask"],
            "ref_mask": np.ones((1, 1, self.z_ref.shape[2]), np.float32),
        })
        return [duration * self.clone_scale, *rest]


class KokoroEngine:
    """Kokoro-82M fine-tuned on SASPEECH (one male voice), with Phonikud for niqqud and
    phonemes. Non-commercial license (inherited from the dataset)."""

    name = "kokoro"
    label = "Kokoro"
    custom_voices = False
    uses_niqqud = True
    model_dir = ROOT / "models" / "kokoro-hebrew"
    # Kokoro's English voice styles have the same shape and read Hebrew clearly with this
    # model (Whisper heard 85-98% of words alone, 98-100% blended 50/50 with Shaul).
    # name -> (English voice, its share; the rest is Shaul). Needs voices-english.bin
    # (voices-v1.0.bin from the kokoro-onnx releases).
    EXTRA_VOICES = {
        "heart": ("af_heart", 1.0), "bella": ("af_bella", 1.0), "emma": ("bf_emma", 1.0), "michael": ("am_michael", 1.0),
        "heart_shaul": ("af_heart", 0.5), "bella_shaul": ("af_bella", 0.5), "emma_shaul": ("bf_emma", 0.5),
        "michael_shaul": ("am_michael", 0.5),
    }

    def __init__(self):
        import onnxruntime as rt
        from kokoro_onnx import Kokoro
        from phonikud import phonemize
        from phonikud_onnx import Phonikud

        print("Loading Kokoro (CPU)...")
        ensure_model("kokoro-hebrew", "kokoro-hebrew/kokoro.onnx")
        ensure_model("phonikud", "kokoro-hebrew/phonikud-1.0.int8.onnx")
        self._phonemize = phonemize
        self.phonikud = Phonikud(str(self.model_dir / "phonikud-1.0.int8.onnx"))
        # Explicit CPU session: kokoro-onnx otherwise tries every provider onnxruntime-gpu
        # lists, and TensorRT/cuDNN DLLs aren't installed here.
        session = rt.InferenceSession(str(self.model_dir / "kokoro.onnx"), providers=["CPUExecutionProvider"])
        self.kokoro = Kokoro.from_session(
            session, str(self.model_dir / "voices-hebrew.bin"), vocab_config=str(self.model_dir / "config.json")
        )
        self.sample_rate = 24000
        self.lock = threading.Lock()
        self.voices = {v.removeprefix("he_"): v for v in self.kokoro.get_voices()}
        self.voices.update(self._extra_voices())
        self.builtin_voices = set(self.voices)
        self.synthesize("שלום", next(iter(self.voices)), 1.0)  # warm-up

    def _extra_voices(self) -> dict:
        try:
            path = ensure_model("kokoro-english-voices", "kokoro-hebrew/voices-english.bin")
        except Exception as e:  # optional: without it Kokoro has Shaul only
            print(f"[WARN] Kokoro: English voice styles unavailable ({e}), only Shaul is available")
            return {}
        english, shaul = np.load(path), np.load(self.model_dir / "voices-hebrew.bin")["he_shaul"]
        return {
            name: (share * english[voice] + (1 - share) * shaul).astype(np.float32)
            for name, (voice, share) in self.EXTRA_VOICES.items()
        }

    def add_voice(self, name: str, audio_path: Path) -> None:
        raise NotImplementedError("Kokoro has fixed voices")

    def remove_voice(self, name: str) -> None:
        pass

    def _pointed(self, text: str) -> str:
        return diacritize_respecting_typed(text, self.phonikud.add_diacritics)

    def diacritize(self, text: str) -> str:
        return self._pointed(text).replace("|", "")  # '|' marks prefixes for the G2P

    def _phonemes(self, text: str) -> str:
        """Hebrew through Phonikud; English runs through espeak (en-us), which kokoro-onnx ships.
        Phonikud passes Latin letters through raw, so Kokoro mumbled English and the words
        around it (Whisper: "ב-WhatsApp אתמול" heard as "בעץ פתמול"; with espeak, right)."""
        out, pos = [], 0
        for m in LATIN_RUN.finditer(text):
            if text[pos : m.start()].strip():
                out.append(self._phonemize(self._pointed(text[pos : m.start()])))
            out.append(self.kokoro.tokenizer.phonemize(m.group(0), "en-us"))
            pos = m.end()
        if text[pos:].strip():
            out.append(self._phonemize(self._pointed(text[pos:])))
        return " ".join(out)

    option_specs = [rng("blend", 0, 0, 100, 5)]

    def _style(self, voice: str, blend: int):
        """blend: percent of Shaul (the Hebrew-trained voice) mixed into the chosen voice."""
        style = self.voices[voice]
        if not blend:
            return style
        if isinstance(style, str):
            style = self.kokoro.get_voice_style(style)
        shaul = self.kokoro.get_voice_style("he_shaul")
        return ((1 - blend / 100) * style + blend / 100 * shaul).astype(np.float32)

    def synthesize(self, text, voice, speed, blend=0, **_):
        parts = []
        with self.lock:
            style = self._style(voice, blend)
            for chunk in split_sentences(text, max_chars=300):
                phonemes = self._phonemes(chunk)
                samples, self.sample_rate = self.kokoro.create(
                    phonemes, voice=style, speed=speed, is_phonemes=True
                )
                parts.append(samples)
        return join_with_pauses(parts, self.sample_rate)


OMNIVOICE_AGES = ["", "child", "teenager", "young adult", "middle-aged", "elderly"]
OMNIVOICE_PITCHES = ["", "very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"]
OMNIVOICE_ACCENTS = ["", "american accent", "british accent", "australian accent", "canadian accent", "indian accent",
                     "russian accent", "chinese accent", "japanese accent", "korean accent", "portuguese accent"]
WORKER_OPTIONS = {
    # design_*: words OmniVoice accepts as its instruct, for designed voices and clones
    # steps below 16 cut sentence endings; F5 sway at -0.3 or above garbles speech (Whisper)
    "omnivoice": [rng("steps", 32, 16, 64, 4), rng("guidance", 2.0, 0.0, 4.0, 0.1), SEED,
                  toggle("denoise", True), toggle("postprocess", True),
                  choice("design_age", OMNIVOICE_AGES), choice("design_pitch", OMNIVOICE_PITCHES),
                  toggle("design_whisper"), choice("design_accent", OMNIVOICE_ACCENTS)],
    "higgs": [rng("temperature", 0.8, 0.1, 1.5, 0.05), rng("top_p", 0.95, 0.5, 1.0, 0.01),
              rng("top_k", 50, 0, 200, 5), SEED],
    # sampling off is the adapter card's greedy decoding (and the fast CUDA-graph path)
    # needs: only works with that toggle on (the UI greys it out otherwise)
    "qwen": [toggle("sampling"), {**rng("temperature", 0.9, 0.1, 1.5, 0.05), "needs": "sampling"},
             {**rng("top_p", 1.0, 0.5, 1.0, 0.01), "needs": "sampling"},
             rng("repetition_penalty", 1.0, 1.0, 1.5, 0.05), {**SEED, "needs": "sampling"}],
    # temperature: the checkpoint's own 0.6
    "magpie": [rng("temperature", 0.6, 0.1, 1.5, 0.05), rng("top_k", 80, 1, 200, 1),
               rng("guidance", 2.5, 1.0, 5.0, 0.1), SEED],
    "f5": [rng("steps", 32, 16, 64, 4), rng("guidance", 2.0, 1.0, 3.0, 0.1), rng("sway", -1.0, -1.0, -0.5, 0.1), SEED],
    "vibevoice": [rng("guidance", 1.3, 1.0, 1.4, 0.05), rng("steps", 10, 5, 30, 1), SEED, toggle("retake", True)],
    # Ming: direction only works with a recorded voice, and the effect is weak in testing
    "ming": [rng("guidance", 2.0, 1.0, 4.0, 0.1), rng("temperature", 0.0, 0.0, 1.0, 0.1), text("direction"), SEED],
    # Drama: top_k 0 is no cap; retake reads again when a take ends after the first clause
    "drama": [rng("temperature", 1.0, 0.5, 1.5, 0.05), rng("top_p", 0.9, 0.5, 1.0, 0.05),
              rng("top_k", 0, 0, 200, 1), SEED, toggle("retake", True)],
    # Breeze: temperature 1.1 raises the floor between words; guidance only works with an instruction
    "breeze": [rng("temperature", 0.9, 0.5, 1.1, 0.05), rng("top_k", 50, 10, 200, 10),
               rng("repetition_penalty", 1.1, 1.0, 1.3, 0.05), text("instruction"),
               {**rng("guidance", 4.0, 1.0, 6.0, 0.5), "needs": "instruction"}, SEED],
}
WORKER_OPTIONS["vibevoice7b"] = WORKER_OPTIONS["vibevoice"]
WORKER_OPTIONS["breezefast"] = WORKER_OPTIONS["breeze"]


class RemoteEngine(OnDemand):
    """An engine served by a worker process in its own conda env (see workers/).

    The worker protocol: GET /info, POST /synthesize {text, voice, speed} -> WAV,
    POST /voices {name, path}, DELETE /voices/{name}, POST /shutdown. Text is split into
    sentences here and sent one at a time; speed is applied here if the worker can't do it.
    The worker process starts on the first reading; unloading stops it, which frees all of its
    GPU memory."""

    uses_niqqud = False  # these models read the text as-is; there's no niqqud step to preview
    ONE_LANGUAGE = {"f5"}  # two models, Hebrew and English: a sentence mixing them comes out wrong
    ENGLISH_ONLY = {"vibevoice", "vibevoice7b"}  # trained on English and Chinese, Hebrew is accented
    NO_HEBREW = {"ming", "breeze", "breezefast"}  # Hebrew text comes out as nonsense
    # QwenTTS-he ends a reading by itself after about 13 s, with or without the fast decode loop.
    # All 24 readings of up to 150 characters finished; at 170 the clone stopped early on both test texts.
    MAX_CHARS = {"qwen": 120}

    def __init__(self, name: str, label: str, env: str, script: str, port: int, startup_timeout: float, info: dict):
        """info: what the worker's /info will say, to list the engine before it loads."""
        self.name, self.label = name, label
        self.one_language = name in self.ONE_LANGUAGE
        self.english_only = name in self.ENGLISH_ONLY
        self.no_hebrew = name in self.NO_HEBREW
        self.option_specs = WORKER_OPTIONS.get(name, [])
        self.env, self.script, self.port = env, script, port
        self.url = f"http://127.0.0.1:{port}"
        self.startup_timeout = startup_timeout
        self.process = None
        self.lock = threading.RLock()
        if not self._python().exists():
            raise RuntimeError(f"conda env '{self.env}' not found")
        self.sample_rate, self.native_speed, self.custom_voices = (
            info["sample_rate"], info["native_speed"], info["custom_voices"])
        self.builtin_voices = set(info["builtin_voices"])
        self.voices = dict.fromkeys([*info["builtin_voices"], *(custom_voice_names() if self.custom_voices else [])])

    def _start(self) -> None:
        if self._info() is None:  # not already running: start it
            python = self._python()
            if not python.exists():
                raise RuntimeError(f"conda env '{self.env}' not found")
            print(f"Starting {self.label} worker (env '{self.env}', port {self.port})...")
            self.process = subprocess.Popen(
                [str(python), "-X", "utf8", str(ROOT / "workers" / self.script.split()[0]), *self.script.split()[1:],
                 "--port", str(self.port)],
                cwd=str(ROOT),
            )

    def _python(self) -> Path:
        return ENVS_DIR / self.env / ("python.exe" if os.name == "nt" else "bin/python")

    def _load(self) -> None:
        self._start()
        self.wait_ready()

    def _unload(self) -> None:
        # /shutdown also stops a worker this server didn't start (one left running earlier).
        try:
            requests.post(f"{self.url}/shutdown", timeout=5)
        except requests.RequestException:
            pass
        if self.process and self.process.poll() is None:
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        deadline = time.monotonic() + 20
        while self._info() is not None and time.monotonic() < deadline:
            time.sleep(0.5)

    def _info(self):
        try:
            return requests.get(f"{self.url}/info", timeout=2).json()
        except (requests.RequestException, ValueError):
            return None

    def wait_ready(self) -> "RemoteEngine":
        deadline = time.monotonic() + self.startup_timeout
        while (info := self._info()) is None:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(f"{self.label} worker exited with code {self.process.returncode}")
            if time.monotonic() > deadline:
                raise RuntimeError(f"{self.label} worker didn't start within {self.startup_timeout:.0f}s")
            time.sleep(1)
        self.sample_rate = info["sample_rate"]
        self.voices = dict.fromkeys(info["voices"])
        self.builtin_voices = set(info["builtin_voices"])
        self.native_speed = info["native_speed"]
        self.custom_voices = info["custom_voices"]
        self.loaded = True  # so the voices below are registered now
        self.last_used = time.monotonic()
        if self.custom_voices:
            load_custom_voices(self)  # the worker knows no custom voices until we register them
        return self

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()

    def add_voice(self, name: str, audio_path: Path) -> None:
        with self.lock:
            if self.loaded:  # else it's registered when the worker starts again
                r = requests.post(f"{self.url}/voices", json={"name": name, "path": str(audio_path)}, timeout=600)
                if r.status_code != 200:
                    raise RuntimeError(r.json().get("detail", r.text))
            self.voices[name] = None

    def remove_voice(self, name: str) -> None:
        with self.lock:
            if self.loaded:
                requests.delete(f"{self.url}/voices/{name}", timeout=30)
            self.voices.pop(name, None)

    def diacritize(self, text: str) -> str:
        return text

    def synthesize(self, text, voice, speed, **options):
        options = {spec["key"]: options[spec["key"]] for spec in self.option_specs if spec["key"] in options}
        parts = []
        with self.lock:
            self.ensure_loaded()
            for chunk in split_sentences(text, self.MAX_CHARS.get(self.name, 200)):
                body = {"text": chunk, "voice": voice, "speed": speed if self.native_speed else 1.0, "options": options}
                r = requests.post(f"{self.url}/synthesize", json=body, timeout=600)
                if r.status_code != 200:
                    raise RuntimeError(f"{self.label}: {r.json().get('detail', r.text)}")
                audio, _ = sf.read(io.BytesIO(r.content), dtype="float32")
                parts.append(audio)
            self.last_used = time.monotonic()
        audio = join_with_pauses(parts, self.sample_rate)
        if speed != 1.0 and not self.native_speed:
            import librosa

            audio = librosa.effects.time_stretch(audio, rate=speed)
        return audio


class ZonosEngine:
    """Zonos-Hebrew running elsewhere (Docker on another machine), reached through its
    Gradio API. Set ZONOS_URL to point at it. License: check Zonos-Hebrew's (likely
    non-commercial) before commercial use."""

    name = "zonos"
    label = "Zonos"
    hebrew_only = True
    option_specs = [
        choice("emotion", ["neutral", "happy", "sad", "angry", "fear", "surprise"], "neutral"),
        rng("pitch_std", 45.0, 0.0, 300.0, 5.0),  # Zonos's own default and range
        rng("guidance", 2.0, 1.0, 5.0, 0.1),
        SEED,  # None: the fixed 420, which decides the default voice
    ]
    custom_voices = True
    uses_niqqud = False  # Zonos reads plain Hebrew text through its own phonemizer
    strips_niqqud = True  # and garbles pointed text, so niqqud is removed before sending
    sample_rate = 44100
    model_choice = "notmax123/Zonos-Hebrew"
    default_rate = 15.0  # Zonos's speaking_rate (phonemes/sec) at speed 1.0
    seed = 420  # fixed: without a reference recording, the seed decides the voice
    # Gradio inputs e1..e8 = happiness, sadness, disgust, fear, surprise, anger, other, neutral
    EMOTIONS = {
        "happy": [0.9, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1],
        "sad": [0.05, 0.9, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1],
        "angry": [0.05, 0.05, 0.05, 0.05, 0.05, 0.9, 0.1, 0.1],
        "fear": [0.05, 0.05, 0.05, 0.9, 0.05, 0.05, 0.1, 0.1],
        "surprise": [0.05, 0.05, 0.05, 0.05, 0.9, 0.05, 0.1, 0.1],
    }

    def __init__(self, url: str):
        from gradio_client import Client, handle_file

        print(f"Connecting to Zonos at {url}...")
        self._handle_file = handle_file
        self.client = Client(url, verbose=False)
        self.lock = threading.Lock()
        self.voices: dict[str, Path | None] = {"default": None}
        self.builtin_voices = {"default"}
        load_custom_voices(self)

    def add_voice(self, name: str, audio_path: Path) -> None:
        self.voices[name] = audio_path  # uploaded with each request; Zonos encodes it itself

    def remove_voice(self, name: str) -> None:
        self.voices.pop(name, None)

    def diacritize(self, text: str) -> str:
        return strip_niqqud(text)

    def _generate(self, text: str, voice: str, speed: float, emotion: str, pitch_std: float, guidance: float,
                  seed: int) -> np.ndarray:
        speaker = self.voices[voice]
        unconditional = [] if emotion in self.EMOTIONS else ["emotion"]
        if speaker is None:
            unconditional.append("speaker")
        e = self.EMOTIONS.get(emotion, [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2])
        path, _ = self.client.predict(
            model_choice=self.model_choice, text=text, language="he",
            speaker_audio=self._handle_file(str(speaker)) if speaker else None, prefix_audio=None,
            **{f"e{i}": v for i, v in enumerate(e, 1)},
            # Measured: audio length goes roughly with rate^-0.5 (rate x1.5 -> 18% shorter), and
            # rates far from 15 drag or stop helping. So the rate covers sqrt of the speed
            # change and synthesize() time-stretches the other sqrt.
            speaking_rate=float(np.clip(self.default_rate * speed, 5.0, 30.0)),
            pitch_std=pitch_std, cfg_scale=guidance,
            seed=seed, randomize_seed=False, unconditional_keys=unconditional,
            api_name="/generate_audio",
        )
        audio, self.sample_rate = sf.read(path, dtype="float32")
        return audio.mean(axis=1) if audio.ndim == 2 else audio

    def synthesize(self, text, voice, speed, emotion="neutral", pitch_std=45.0, guidance=2.0, seed=None, **_):
        text = strip_niqqud(text)  # Zonos's phonemizer falls apart on pointed text
        parts = []
        with self.lock:
            for chunk in split_sentences(text):
                try:
                    parts.append(self._generate(chunk, voice, speed, emotion, pitch_std, guidance,
                                                self.seed if seed is None else seed))
                except Exception as e:  # network or remote failure: report it like a worker
                    raise RuntimeError(f"Zonos: {e}") from e
        audio = join_with_pauses(parts, self.sample_rate)
        if speed != 1.0:
            import librosa

            audio = librosa.effects.time_stretch(audio, rate=speed**0.5)
        return audio


# Zonos-Hebrew is optional: off unless ZONOS_URL (or server.py --zonos-url) points at its
# Gradio server, e.g. http://<docker-host>:7860/ (docker/zonos-hebrew).
ZONOS_URL = os.environ.get("ZONOS_URL", "")

WORKERS = [
    # name, label, conda env, script (and its arguments), port offset from the server's port,
    # startup timeout (first run downloads/loads weights). Offsets keep each server instance's
    # workers apart: server 8010 -> workers 8031 to 8041.
    ("omnivoice", "OmniVoice", "omnivoice", "omnivoice_worker.py", 21, 600),
    ("higgs", "Higgs", "higgs", "higgs_worker.py", 22, 600),
    ("qwen", "QwenTTS-he", "qwentts", "qwen_worker.py", 23, 600),
    ("magpie", "Magpie-TTS Hebrew", "magpie", "magpie_worker.py", 24, 600),
    ("f5", "F5-TTS Hebrew", "f5", "f5_worker.py", 25, 600),
    # VibeVoice runs in OmniVoice's env (same transformers); the first start downloads 5 / 19 GB
    ("vibevoice", "VibeVoice 1.5B", "omnivoice", "vibevoice_worker.py --size 1.5b", 26, 1800),
    ("vibevoice7b", "VibeVoice 7B", "omnivoice", "vibevoice_worker.py --size 7b", 27, 1800),
    ("ming", "Ming-3.0-tts", "ming", "ming_worker.py", 28, 600),
    # The first start downloads about 40 GB and converts it, which needs about 32 GB of system memory
    ("drama", "Fish Audio Drama 3", "drama", "drama_worker.py", 29, 3600),
    # One process each: fast holds about twice the GPU memory for about 7 times the speed
    ("breeze", "Breeze TTS 2", "breeze", "breeze_worker.py", 30, 1800),
    ("breezefast", "Breeze TTS 2 Fast", "breeze", "breeze_worker.py --fast", 31, 1800),
]
# What each worker's /info says, for listing it while it's unloaded (see the worker's own voices).
HIGGS_PRESETS = ROOT / "models" / "higgs-voices"  # optional, see the README
VIBEVOICE_INFO = {"sample_rate": 24000, "native_speed": False, "custom_voices": True,
                  "builtin_voices": ["alice", "carter", "frank", "maya", "samuel"]}
BREEZE_INFO = {"sample_rate": 24000, "native_speed": False, "custom_voices": True, "builtin_voices": ["default"]}
WORKER_INFO = {
    "omnivoice": {"sample_rate": 24000, "native_speed": True, "custom_voices": True, "builtin_voices": ["female", "male"]},
    "higgs": {"sample_rate": 24000, "native_speed": False, "custom_voices": True,
              "builtin_voices": [p.stem for p in sorted(HIGGS_PRESETS.glob("*.wav"))]},
    "qwen": {"sample_rate": 24000, "native_speed": False, "custom_voices": True, "builtin_voices": ["female", "male"]},
    "magpie": {"sample_rate": 22050, "native_speed": False, "custom_voices": False,
               "builtin_voices": ["female1", "female1-he", "female2", "female3", "female4", "female5", "male1", "male1-he",
                                  "male2", "male3", "male4", "male5", "voice1", "voice2", "voice3"]},
    "f5": {"sample_rate": 24000, "native_speed": True, "custom_voices": True, "builtin_voices": []},
    "vibevoice": VIBEVOICE_INFO,
    "vibevoice7b": VIBEVOICE_INFO,
    "ming": {"sample_rate": 44100, "native_speed": False, "custom_voices": True,
             "builtin_voices": ["emma", "george", "charlotte"]},
    "drama": {"sample_rate": 44100, "native_speed": False, "custom_voices": True, "builtin_voices": ["default"]},
    "breeze": BREEZE_INFO,
    "breezefast": BREEZE_INFO,
}


def engine_enabled(name: str) -> bool:
    """VOICE_ENGINES, e.g. "lightblue,kokoro,qwen,magpie", limits which engines start (and so
    which models get downloaded); unset means all."""
    wanted = os.environ.get("VOICE_ENGINES", "").replace(" ", "")
    return not wanted or name in wanted.split(",")


def load_engines(device: str, server_port: int, zonos_url: str = ZONOS_URL) -> dict:
    """The CPU engines load here; GPU engines are only listed and load when first used.
    LightBlue comes first in the list: it's the fastest and the UI's default."""
    workers = []
    for name, label, env, script, offset, timeout in WORKERS:
        if not engine_enabled(name):
            continue
        try:
            workers.append(RemoteEngine(name, label, env, script, server_port + offset, timeout, WORKER_INFO[name]))
        except Exception as e:
            print(f"[WARN] {label} isn't available: {e}")

    engines = {}
    factories = [("lightblue", LightBlueEngine), ("blue25", BlueTTS25Engine), ("kokoro", KokoroEngine),
                 ("chatterbox", lambda: ChatterboxEngine(device))]
    if zonos_url:
        factories.append(("zonos", lambda: ZonosEngine(zonos_url)))
    for name, factory in factories:
        if not engine_enabled(name):
            continue
        try:
            engine = factory()
            engines[engine.name] = engine
        except Exception as e:  # keep serving with whatever engine did load
            print(f"[WARN] engine failed to load: {e}")

    engines.update((worker.name, worker) for worker in workers)
    if not engines:
        raise RuntimeError("No TTS engine could be loaded.")
    return engines


def stop_workers(engines: dict) -> None:
    for engine in engines.values():
        if isinstance(engine, RemoteEngine):
            engine.stop()


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
