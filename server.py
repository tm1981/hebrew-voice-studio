"""Local Hebrew TTS API over several engines: LightBlue, Blue 2.5 and Kokoro (fast, CPU),
Chatterbox (GPU), OmniVoice, Higgs, QwenTTS-he, Magpie and F5-TTS (GPU workers in their own conda envs, which
this server starts and stops itself), and optionally Zonos-Hebrew on another machine (its address is set on the web UI's
Server status screen, or with --zonos-url / ZONOS_URL; off when not set).

Run:
    conda activate voice
    python server.py                         # port 8010, Chatterbox on GPU 0
    set TTS_DEVICE=cuda:1 & python server.py --port 8011

Voices are addressed as "<engine>/<name>", e.g. "lightblue/noa" or "chatterbox/default".
Custom voices: drop a clean 10-15s recording of one speaker (wav/mp3/flac/ogg) into
voices/ and restart, or upload it from the web UI. It becomes a voice in every engine
that can clone (all but Kokoro and Magpie).

Access: requests from this PC need nothing. Anything else (LAN, phone, other tools) needs
the API key, sent as "Authorization: Bearer <key>", "X-API-Key: <key>" or ?api_key=<key>.
The key comes from --api-key or TTS_API_KEY, else it's generated once into api_key.txt.
--no-auth turns the check off.

Text: every request goes through the pronunciation dictionary (pronunciations.json) and,
unless "normalize": false, Hebrew normalization of numbers, dates, currency and
abbreviations (text_prep.py). [הפסקה 2] inserts a pause; with "dialogue": true, lines like
"נועה: ..." are read by the voice in "speakers" (script.py).

Library and cache: readings made through /tts/stream (the web UI) are saved in
outputs/library with their sentences, for replay, download and per-sentence editing, and
an identical request is replayed from there. /tts and /v1 use a memory cache instead.

Endpoints:
    GET    /                                             -> web UI
    GET    /health
    GET    /voices                                       -> [{"id", "engine", "name", "builtin", "options"}]
    GET    /options                                      -> {engine: [setting specs]}; a request's "options"
    POST   /voices  (multipart: name, files)             -> add a custom voice (1-5 clips) to every cloning engine
    DELETE /voices/{name}
    POST   /diacritize {"text": "...", "engine": "lightblue"} -> {"text": "<with niqqud>"}
    POST   /tts {"text": "...", "voice": "lightblue/noa", "speed": 1.0, "format": "wav"|"mp3"} -> audio
    POST   /tts/stream (same body)                       -> audio sentence by sentence (see tts_stream)
    GET    /tts?text=...&voice=...&speed=...             -> audio (handy for browsers)
    POST   /prepare {"text": "..."}                      -> the text as engines will receive it
    POST   /speakers {"text": "..."}                     -> speaker names for dialogue mode
    GET    /library, /library/{id}, /library/{id}/audio?format=mp3, /library/{id}/sentences/{n}
    POST   /library/{id}/sentences/{n} {"text"?}         -> regenerate (or edit) one sentence
    DELETE /library/{id}
    GET    /presets  /  PUT /presets {"name", "settings"}  /  DELETE /presets/{name}
    POST   /documents (multipart: file, settings)        -> background job; GET /jobs/{id}, /documents/{id}/zip
    POST   /voices/check (multipart: files)              -> recording quality warnings
    GET    /status                                       -> engines, GPU, library, jobs
    GET    /settings/zonos  /  PUT /settings/zonos {"url"} -> connect to / drop Zonos-Hebrew (saved in settings.json)
    POST   /engines/{name}/unload, /engines/{name}/load  -> free / take a GPU engine's memory (it also
                                                            loads by itself on its next request)
    POST   /gpu/free                                     -> unload every GPU engine that isn't busy
    GET    /settings/auto-unload  /  PUT {"minutes"}     -> unload GPU engines unused that long (0 = off)
    GET    /pronunciations                               -> {"word": "reading", ...}
    PUT    /pronunciations {"word", "reading"}  /  DELETE /pronunciations/{word}
    POST   /v1/audio/speech                              -> OpenAI-compatible TTS (see openai_speech)
    GET    /v1/models, /v1/audio/voices, /v1/audio/models
    GET    /docs                                         -> interactive Swagger UI
"""
import subprocess
import sys

# LightBlue opens its files without an encoding, which on Windows means the Hebrew ANSI
# codepage and a crash. Re-run in Python's UTF-8 mode rather than depend on PYTHONUTF8.
if not sys.flags.utf8_mode:
    try:
        sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))
    except KeyboardInterrupt:  # Ctrl+C reaches the child too; it shuts itself down
        sys.exit(130)

import os
from pathlib import Path

# Self-contained: model downloads (Hugging Face, torch hub) go into models/ in this folder, not
# the user's profile. Set before torch/transformers are imported; the workers inherit it.
for _var, _dir in (("HF_HOME", "hf"), ("TORCH_HOME", "torch")):
    os.environ.setdefault(_var, str(Path(__file__).resolve().parent / "models" / _dir))

import argparse
import atexit
import io
import ipaddress
import json
import re
import secrets
import threading
import time
from typing import Any

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from chatterbox_fixes import DEFAULT_TEMPERATURE
from paths import DATA_DIR
from engines import (
    ROOT,
    VOICE_EXTENSIONS,
    VOICES_DIR,
    clean_options,
    default_device,
    load_engines,
    prepared_clip,
    remove_voice_files,
    split_sentences,
    stop_workers,
)
from documents import MAX_DOCUMENT_CHARS, Jobs, extract_text, split_chapters, zip_chapters
from engines import check_recording
from library import SENTENCE_GAP, Library, MemoryCache, cache_key, to_rate
from script import detect_speakers, parse
from text_prep import Dictionary, prepare

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8010)  # 8000 is taken by a Windows service here
parser.add_argument("--api-key", default=os.environ.get("TTS_API_KEY"), help="key required off this PC")
parser.add_argument("--no-auth", action="store_true", help="accept requests from anywhere without a key")
parser.add_argument("--zonos-url", default=None,
                    help="Zonos-Hebrew Gradio server, e.g. http://<docker-host>:7860/ (\"\" turns it off). "
                         "Without it: the address saved in the web UI, else ZONOS_URL")
parser.add_argument("--lan-needs-key", action="store_true", help="also require the key from the home network")
args = parser.parse_args()

DEVICE = os.environ.get("TTS_DEVICE", default_device())
VOICE_NAME = re.compile(r"^[\w\- ]{1,40}$")  # \w covers Hebrew letters
DEFAULT_VOICE = "chatterbox/default"  # kept for API clients written before LightBlue
MAX_CLIPS = 5
API_KEY_FILE = DATA_DIR / "api_key.txt"
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


def trusted_client(host: str) -> bool:
    """This PC always; the home network (192.168.x.x, 10.x, 172.16-31.x) unless --lan-needs-key."""
    if host in LOCAL_HOSTS:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or (not args.lan_needs_key and ip.is_private and not ip.is_link_local)

OPEN_PATHS = {"/", "/favicon.ico", "/docs", "/openapi.json"}  # the UI page asks for the key itself


def load_api_key() -> str | None:
    if args.no_auth:
        return None
    if args.api_key:
        return args.api_key
    if not API_KEY_FILE.exists():
        API_KEY_FILE.write_text(secrets.token_urlsafe(24), encoding="utf-8")
    return API_KEY_FILE.read_text(encoding="utf-8").strip()


API_KEY = load_api_key()
dictionary = Dictionary()
library = Library()
api_cache = MemoryCache()
jobs = Jobs()
PRESETS_FILE = DATA_DIR / "presets.json"
SETTINGS_FILE = DATA_DIR / "settings.json"  # settings changed from the web UI


def load_settings() -> dict:
    return json.loads(SETTINGS_FILE.read_text(encoding="utf-8")) if SETTINGS_FILE.exists() else {}


# Zonos address: --zonos-url, else the one saved from the web UI, else ZONOS_URL; empty = off.
zonos_url = args.zonos_url if args.zonos_url is not None else load_settings().get("zonos_url", os.environ.get("ZONOS_URL", ""))
engines = load_engines(DEVICE, args.port, zonos_url)
zonos_state = {"url": zonos_url, "error": "Couldn't connect at startup." if zonos_url and "zonos" not in engines else None}
atexit.register(stop_workers, engines)
app = FastAPI(title="Hebrew TTS")
if API_KEY:
    print(f"API key for requests from outside the home network: {API_KEY}  (also in {API_KEY_FILE.name})")


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """Requests from this PC and the home network pass; others need the key. Compared in constant time."""
    path = request.url.path
    open_path = path in OPEN_PATHS or path.startswith("/fonts/")
    if API_KEY and request.client and not trusted_client(request.client.host) and not open_path:
        auth = request.headers.get("authorization", "")
        given = (
            request.headers.get("x-api-key")
            or (auth[7:] if auth.lower().startswith("bearer ") else "")
            or request.query_params.get("api_key", "")
        )
        if not secrets.compare_digest(given.encode(), API_KEY.encode()):
            return JSONResponse({"detail": "Missing or wrong API key"}, status_code=401)
    return await call_next(request)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = DEFAULT_VOICE
    language: str = "he"
    speed: float = Field(1.0, ge=0.5, le=2.0)
    format: str = Field("wav", pattern="^(wav|mp3|flac|opus|pcm)$")
    # Per-engine settings, {engine: {key: value}}; the keys each engine takes are in GET /options.
    # Missing keys keep the engine's defaults.
    options: dict[str, dict[str, Any]] = {}
    # Silence between sentences, seconds.
    sentence_pause: float = Field(SENTENCE_GAP, ge=0.0, le=2.0)
    # Older form of a few of those settings, still accepted: Chatterbox and Zonos.
    exaggeration: float = Field(0.5, ge=0.0, le=2.0)
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0)
    temperature: float = Field(DEFAULT_TEMPERATURE, ge=0.05, le=2.0)
    emotion: str = Field("neutral", pattern="^(neutral|happy|sad|angry|fear|surprise)$")
    # Numbers, dates, currency and abbreviations into words (the dictionary always applies).
    normalize: bool = True
    # "שם: טקסט" lines read by the voice mapped to each speaker (see script.py).
    dialogue: bool = False
    speakers: dict[str, str] = {}
    # Reuse an identical earlier reading (library or memory cache) instead of generating.
    use_cache: bool = True

    def settings(self) -> dict:
        """What's saved with a library item and reused when a sentence is regenerated. Settings
        left at their defaults are left out, so readings made before they existed keep their
        cache keys."""
        out = self.model_dump(exclude={"text", "format", "use_cache"})
        if not any(self.options.values()):
            out.pop("options")
        if self.sentence_pause == SENTENCE_GAP:
            out.pop("sentence_pause")
        return out

    def engine_options(self, engine) -> dict:
        """The settings for one engine, cleaned against its specs (older fields as defaults)."""
        legacy = {"exaggeration": self.exaggeration, "cfg_weight": self.cfg_weight,
                  "temperature": self.temperature, "emotion": self.emotion}
        specs = getattr(engine, "option_specs", [])
        given = self.options.get(engine.name, {})
        return clean_options(specs, {**{k: v for k, v in legacy.items() if engine.name in ("chatterbox", "zonos")}, **given})


class DiacritizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    engine: str = "chatterbox"
    normalize: bool = True


class PrepareRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    normalize: bool = True


class PronunciationRequest(BaseModel):
    word: str = Field(..., min_length=1, max_length=60)
    reading: str = Field(..., min_length=1, max_length=200)


def synthesize(engine, voice: str, text: str, req: TTSRequest) -> np.ndarray:
    """Run an engine with the request's settings, mapping failures to HTTP errors."""
    try:
        return engine.synthesize(text, voice, req.speed, language=req.language, **req.engine_options(engine))
    except ValueError as e:  # e.g. a language this engine doesn't support
        raise HTTPException(400, str(e))
    except RuntimeError as e:  # a worker or remote engine failed or went away
        raise HTTPException(502, str(e))


def resolve_voice(voice_id: str):
    """'lightblue/noa' -> (engine, 'noa'). A bare name means a Chatterbox voice."""
    engine_name, _, name = voice_id.rpartition("/")
    engine = engines.get(engine_name or "chatterbox")
    if engine is None or name not in engine.voices:
        raise HTTPException(400, f"No voice '{voice_id}'. Available: {', '.join(v['id'] for v in list_voices())}")
    return engine, name


def resample(audio: np.ndarray, sample_rate: int, target: int) -> np.ndarray:
    if sample_rate == target:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=sample_rate, target_sr=target)


def encode(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    """wav, mp3, flac; opus (Ogg, 48 kHz); pcm (raw 16-bit little-endian, 24 kHz, as OpenAI sends)."""
    if fmt == "pcm":
        pcm = np.clip(resample(audio, sample_rate, 24000), -1.0, 1.0)
        return (pcm * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    if fmt == "opus":
        sf.write(buf, resample(audio, sample_rate, 48000), 48000, format="OGG", subtype="OPUS")
    else:
        sf.write(buf, audio, sample_rate, format=fmt.upper())
    return buf.getvalue()


MEDIA_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "opus": "audio/ogg", "pcm": "audio/pcm"}


def plan(req: TTSRequest) -> list[tuple]:
    """The reading as [("speech", voice_id, sentence) | ("pause", seconds)]: dialogue and
    pause markers resolved, dictionary and normalization applied, split into sentences."""
    items = []
    for seg in parse(req.text, req.voice, req.dialogue, req.speakers):
        if seg[0] == "pause":
            items.append(seg)
            continue
        resolve_voice(seg[1])  # 400 now for an unknown voice, not halfway through
        items += [("speech", seg[1], s) for s in split_sentences(prepare(seg[2], dictionary, req.normalize))]
    if not any(i[0] == "speech" for i in items):
        raise HTTPException(400, "There's no text to read.")
    return items


def request_key(req: TTSRequest, items: list[tuple]) -> str:
    return cache_key(items=items, **req.settings())


def speak_item(item: tuple, req: TTSRequest) -> tuple[np.ndarray, int]:
    engine, voice = resolve_voice(item[1])
    return synthesize(engine, voice, item[2], req), engine.sample_rate


def assemble(results: list[tuple], gap: float = SENTENCE_GAP) -> tuple[np.ndarray, int]:
    """results: [(item, audio, sample_rate)] -> one track at the highest sample rate used
    (a dialogue can mix engines), with a short gap between sentences and pauses as given."""
    rate = max(sr for item, _, sr in results if item[0] == "speech")
    parts, previous_speech = [], False
    for item, audio, sr in results:
        if item[0] == "pause":
            parts.append(np.zeros(int(rate * item[1]), dtype=np.float32))
            previous_speech = False
            continue
        if previous_speech:
            parts.append(np.zeros(int(rate * gap), dtype=np.float32))
        parts.append(to_rate(audio, sr, rate).astype(np.float32))
        previous_speech = True
    return np.concatenate(parts), rate


def silence_wav(seconds: float) -> bytes:
    return encode(np.zeros(int(24000 * seconds), dtype=np.float32), 24000, "wav")


def item_summary(meta: dict) -> list[dict]:
    return [{k: i.get(k) for k in ("kind", "text", "voice", "seconds")} for i in meta["items"]]


@app.get("/", include_in_schema=False)
def web_ui():
    # no-cache: browsers revalidate, so an updated page shows on reload, not an old cached copy
    return FileResponse(ROOT / "web" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/fonts/{name}", include_in_schema=False)
def web_font(name: str):
    """The UI's default fonts, bundled so the page works without internet (tools/fetch_fonts.py)."""
    path = ROOT / "web" / "fonts" / name
    if not re.fullmatch(r"[\w.-]+\.(woff2|css)", name) or not path.is_file():
        raise HTTPException(404, "No such font file.")
    return FileResponse(path, headers={"Cache-Control": "max-age=604800"})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(DEVICE) if "cuda" in DEVICE else None,
        "engines": {e.name: {"label": e.label, "sample_rate": e.sample_rate} for e in engines.values()},
    }


@app.get("/voices")
def list_voices():
    return [
        {
            "id": f"{e.name}/{name}",
            "engine": e.name,
            "engine_label": e.label,
            "name": name,
            "builtin": name in e.builtin_voices,
            "options": [spec["key"] for spec in getattr(e, "option_specs", [])],
            "hebrew_only": getattr(e, "hebrew_only", False),  # English in the text comes out wrong
        }
        for e in engines.values()
        for name in e.voices
    ]


@app.get("/options")
def list_options():
    """Each engine's settings: [{key, type: range|select|toggle|seed, default, min, max, step, choices}]."""
    return {e.name: getattr(e, "option_specs", []) for e in engines.values()}


@app.post("/voices")
def add_voice(name: str = Form(...), files: list[UploadFile] = File(...)):
    """One recording, or up to MAX_CLIPS recordings of the same speaker (stored as a folder)."""
    name = name.strip()
    builtin = {n for e in engines.values() for n in e.builtin_voices}
    if not VOICE_NAME.match(name) or name in builtin:
        raise HTTPException(400, "Voice name must be 1-40 letters, digits, spaces, - or _, and not a built-in voice.")
    if len(files) > MAX_CLIPS:
        raise HTTPException(400, f"Upload at most {MAX_CLIPS} recordings per voice.")
    suffixes = [os.path.splitext(f.filename or "")[1].lower() for f in files]
    if any(s not in VOICE_EXTENSIONS for s in suffixes):
        raise HTTPException(400, f"Upload {', '.join(sorted(VOICE_EXTENSIONS))} files.")
    for engine in engines.values():
        engine.remove_voice(name)
    remove_voice_files(name)
    if len(files) == 1:
        (VOICES_DIR / f"{name}{suffixes[0]}").write_bytes(files[0].file.read())
    else:
        folder = VOICES_DIR / name
        folder.mkdir()
        for i, (f, suffix) in enumerate(zip(files, suffixes), 1):
            (folder / f"clip{i}{suffix}").write_bytes(f.file.read())
    errors = {}
    try:
        clip = prepared_clip(name)
    except Exception as e:
        remove_voice_files(name)
        raise HTTPException(400, f"Could not read that recording: {e}")
    for engine in engines.values():
        if engine.custom_voices:
            try:
                engine.add_voice(name, clip)
            except Exception as e:  # one engine failing shouldn't lose the voice everywhere
                errors[engine.label] = str(e)
    if not any(name in e.voices for e in engines.values()):
        remove_voice_files(name)
        raise HTTPException(400, f"Could not read that recording: {'; '.join(errors.values())}")
    for label, error in errors.items():
        print(f"[WARN] {label}: voice '{name}' not added: {error}")
    return list_voices()


@app.delete("/voices/{name}")
def delete_voice(name: str):
    if not any(name in e.voices and name not in e.builtin_voices for e in engines.values()):
        raise HTTPException(404, f"No custom voice named '{name}'.")
    for engine in engines.values():
        if name not in engine.builtin_voices:
            engine.remove_voice(name)
    remove_voice_files(name)
    return list_voices()


@app.post("/diacritize")
def diacritize(req: DiacritizeRequest):
    """Return the text with the niqqud the chosen engine will actually read. Words the
    user already pointed keep their niqqud. uses_niqqud=False: the engine reads plain
    text, and the text comes back unchanged."""
    engine = engines.get(req.engine)
    if engine is None:
        raise HTTPException(400, f"Unknown engine '{req.engine}'")
    text = prepare(req.text, dictionary, req.normalize)
    return {
        "text": engine.diacritize(text),
        "uses_niqqud": engine.uses_niqqud,
        "strips_niqqud": getattr(engine, "strips_niqqud", False),
    }


@app.post("/prepare")
def prepare_text(req: PrepareRequest):
    """The text after the dictionary and normalization, as every engine receives it."""
    return {"text": prepare(req.text, dictionary, req.normalize)}


@app.get("/pronunciations")
def list_pronunciations():
    return dict(sorted(dictionary.entries.items()))


@app.put("/pronunciations")
def set_pronunciation(req: PronunciationRequest):
    dictionary.set(req.word.strip(), req.reading.strip())
    return list_pronunciations()


@app.delete("/pronunciations/{word}")
def delete_pronunciation(word: str):
    if word not in dictionary.entries:
        raise HTTPException(404, f"'{word}' isn't in the dictionary.")
    dictionary.remove(word)
    return list_pronunciations()


@app.post("/tts")
def tts(req: TTSRequest):
    """One finished file. Not saved to the library (API callers would flood it); identical
    requests come from a memory cache, or from the library if the UI already made it."""
    items = plan(req)
    key = request_key(req, items)
    t0 = time.perf_counter()
    cached = api_cache.get(key) if req.use_cache else None
    if cached is None and req.use_cache and (meta := library.find(key)):
        cached = sf.read(library.full_path(meta["id"]), dtype="float32")
    if cached is not None:
        audio, rate = cached
    else:
        audio, rate = assemble([(i, *speak_item(i, req)) if i[0] == "speech" else (i, None, 0) for i in items],
                               req.sentence_pause)
        api_cache.put(key, audio, rate)
    return Response(
        encode(audio, rate, req.format),
        media_type=MEDIA_TYPES[req.format],
        headers={
            "X-Audio-Seconds": f"{len(audio) / rate:.2f}",
            "X-Generation-Seconds": f"{time.perf_counter() - t0:.2f}",
            "X-Engine": resolve_voice(req.voice)[0].name,
            "X-Cache": "hit" if cached is not None else "miss",
        },
    )


def frame(kind: bytes, payload: bytes) -> bytes:
    return kind + len(payload).to_bytes(4, "big") + payload


@app.post("/tts/stream")
def tts_stream(req: TTSRequest):
    """Sentence-by-sentence audio, so playback can start before long text is done. The
    result is saved to the library (sentences kept separately for editing); an identical
    earlier reading is replayed from there instead of generated again.

    Body: frames of 1-byte type + 4-byte big-endian length + payload:
      a  WAV (16-bit) of one sentence or pause, sent as soon as it's ready
      m  JSON {"id", "audio_seconds", "generation_seconds", "cached", "items"} at the end
      f  the whole result in the requested format (for download)
      e  JSON {"detail"} if generation failed part-way
    """
    items = plan(req)
    key = request_key(req, items)

    def replay(meta):
        for n, item in enumerate(meta["items"]):
            if item["kind"] == "pause":
                yield frame(b"a", silence_wav(item["seconds"]))
            else:
                audio, sr = library.sentence_audio(meta, n)
                yield frame(b"a", encode(audio, sr, "wav"))
        yield from finish(meta, cached=True)

    def finish(meta, cached):
        info = {"id": meta["id"], "audio_seconds": meta["audio_seconds"], "cached": cached,
                "generation_seconds": 0.0 if cached else meta["generation_seconds"], "items": item_summary(meta)}
        yield frame(b"m", json.dumps(info, ensure_ascii=False).encode())
        full, rate = sf.read(library.full_path(meta["id"]), dtype="float32")
        yield frame(b"f", encode(full, rate, req.format))

    def generate():
        t0 = time.perf_counter()
        results = []
        try:
            for item in items:
                if item[0] == "pause":
                    results.append({"kind": "pause", "seconds": item[1]})
                    yield frame(b"a", silence_wav(item[1]))
                    continue
                audio, rate = speak_item(item, req)
                results.append({"kind": "speech", "voice": item[1], "text": item[2], "audio": audio, "sample_rate": rate})
                yield frame(b"a", encode(audio, rate, "wav"))
        except HTTPException as e:
            yield frame(b"e", json.dumps({"detail": e.detail}).encode())
            return
        meta = library.save(
            title=req.text.strip().split("\n")[0][:60], kind="dialogue" if req.dialogue else "reading",
            text=req.text, voice=req.voice, settings=req.settings(), items=results,
            generation_seconds=time.perf_counter() - t0, key=key,
        )
        yield from finish(meta, cached=False)

    hit = library.find(key) if req.use_cache else None
    return StreamingResponse(replay(hit) if hit else generate(), media_type="application/octet-stream")


@app.post("/speakers")
def speakers(req: PrepareRequest):
    """Speaker names in "שם: טקסט" lines, for mapping each to a voice."""
    return {"speakers": detect_speakers(req.text)}


# --- Library: saved readings, playable, downloadable, editable sentence by sentence

class SentenceEdit(BaseModel):
    text: str | None = Field(None, max_length=1000)  # None: same text, a new take


def library_meta(item_id: str) -> dict:
    try:
        return library.get(item_id)
    except KeyError:
        raise HTTPException(404, f"No saved reading '{item_id}'.")


@app.get("/library")
def library_list():
    return library.list()


@app.get("/library/{item_id}")
def library_get(item_id: str):
    meta = library_meta(item_id)
    return {**{k: v for k, v in meta.items() if k != "items"}, "items": item_summary(meta)}


@app.get("/library/{item_id}/audio")
def library_audio(item_id: str, format: str = Query("wav", pattern="^(wav|mp3|flac|opus)$")):
    meta = library_meta(item_id)
    audio, rate = sf.read(library.full_path(meta["id"]), dtype="float32")
    return Response(encode(audio, rate, format), media_type=MEDIA_TYPES[format],
                    headers={"Content-Disposition": f'attachment; filename="{item_id}.{format}"'})


@app.get("/library/{item_id}/sentences/{index}")
def library_sentence(item_id: str, index: int):
    meta = library_meta(item_id)
    if not (0 <= index < len(meta["items"])) or meta["items"][index]["kind"] != "speech":
        raise HTTPException(404, "No such sentence.")
    audio, rate = library.sentence_audio(meta, index)
    return Response(encode(audio, rate, "wav"), media_type="audio/wav")


@app.post("/library/{item_id}/sentences/{index}")
def library_regenerate(item_id: str, index: int, edit: SentenceEdit):
    """Regenerate one sentence (optionally with new text) with the reading's original
    settings and voice, then rebuild the full file."""
    meta = library_meta(item_id)
    if not (0 <= index < len(meta["items"])) or meta["items"][index]["kind"] != "speech":
        raise HTTPException(404, "No such sentence.")
    item = meta["items"][index]
    req = TTSRequest(text="x", **meta["settings"])
    text = prepare(edit.text, dictionary, req.normalize) if edit.text else item["text"]
    audio, rate = speak_item(("speech", item["voice"], text), req)
    meta = library.replace_sentence(item_id, index, text, audio, rate)
    return library_get(meta["id"])


@app.delete("/library/{item_id}")
def library_delete(item_id: str):
    library_meta(item_id)
    library.delete(item_id)
    return {"ok": True}


# --- Presets: named settings, shared by every browser that uses this server

class Preset(BaseModel):
    name: str = Field(..., min_length=1, max_length=40)
    settings: dict


def load_presets() -> dict:
    return json.loads(PRESETS_FILE.read_text(encoding="utf-8")) if PRESETS_FILE.exists() else {}


@app.get("/presets")
def presets_list():
    return load_presets()


@app.put("/presets")
def presets_set(preset: Preset):
    presets = load_presets()
    presets[preset.name.strip()] = preset.settings
    PRESETS_FILE.write_text(json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8")
    return presets


@app.delete("/presets/{name}")
def presets_delete(name: str):
    presets = load_presets()
    presets.pop(name, None)
    PRESETS_FILE.write_text(json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8")
    return presets


# --- Settings from the web UI: the optional Zonos-Hebrew server

class ZonosSettings(BaseModel):
    url: str = Field("", max_length=300)


@app.get("/settings/zonos")
def zonos_settings():
    return {"url": zonos_state["url"], "connected": "zonos" in engines, "error": zonos_state["error"]}


@app.put("/settings/zonos")
def zonos_settings_set(body: ZonosSettings):
    """Connects to (or, with an empty url, drops) Zonos without a restart, and saves the
    address for the next start. A failed connection is reported but the address is kept."""
    from engines import ZonosEngine

    url = body.url.strip()
    if url and not re.match(r"^https?://", url):
        raise HTTPException(400, "The address must start with http:// or https://")
    engines.pop("zonos", None)
    zonos_state.update(url=url, error=None)
    if url:
        try:
            engines["zonos"] = ZonosEngine(url)
        except Exception as e:
            zonos_state["error"] = str(e) or type(e).__name__
    save_settings(zonos_url=url)
    return zonos_settings()


def save_settings(**values) -> None:
    SETTINGS_FILE.write_text(json.dumps({**load_settings(), **values}, indent=2), encoding="utf-8")


# --- GPU memory: GPU engines can be unloaded, and load again on their next request

def gpu_engine(name: str):
    from engines import OnDemand

    engine = engines.get(name)
    if not isinstance(engine, OnDemand):
        raise HTTPException(404, f"No GPU engine '{name}'")
    return engine


@app.post("/engines/{name}/unload")
def engine_unload(name: str):
    if not gpu_engine(name).unload(wait=False):
        raise HTTPException(409, "busy")  # reading right now; try again when it's done
    return status()


@app.post("/engines/{name}/load")
def engine_load(name: str):
    engine = gpu_engine(name)
    try:
        with engine.lock:
            engine.ensure_loaded()
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    return status()


@app.post("/gpu/free")
def gpu_free():
    """Unloads every GPU engine that isn't reading right now; busy ones are listed."""
    from engines import OnDemand

    busy = [e.label for e in engines.values() if isinstance(e, OnDemand) and not e.unload(wait=False)]
    return {**status(), "busy": busy}


class AutoUnloadSettings(BaseModel):
    minutes: int = Field(..., ge=0, le=24 * 60)  # 0 = off


auto_unload = {"minutes": int(load_settings().get("auto_unload_minutes", 30))}


@app.get("/settings/auto-unload")
def auto_unload_settings():
    return auto_unload


@app.put("/settings/auto-unload")
def auto_unload_settings_set(body: AutoUnloadSettings):
    auto_unload["minutes"] = body.minutes
    save_settings(auto_unload_minutes=body.minutes)
    return auto_unload


def auto_unload_loop() -> None:
    """Unloads a GPU engine that hasn't read anything for auto_unload minutes."""
    from engines import OnDemand

    while True:
        time.sleep(30)
        limit = auto_unload["minutes"] * 60
        for engine in list(engines.values()):
            if limit and isinstance(engine, OnDemand) and engine.loaded and time.monotonic() - engine.last_used > limit:
                try:
                    engine.unload(wait=False)
                except Exception as e:
                    print(f"[WARN] couldn't unload {engine.label}: {e}")


threading.Thread(target=auto_unload_loop, daemon=True, name="auto-unload").start()


# --- Long documents: read chapter by chapter in the background into the library

@app.post("/documents")
def documents_start(file: UploadFile = File(...), settings: str = Form("{}")):
    """TXT/MD/DOCX/PDF -> chapters -> one library item each. settings: the same JSON fields
    as /tts (voice, speed, normalize...). Poll GET /jobs/{id} for progress."""
    try:
        text = extract_text(file.filename or "", file.file.read())
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not text.strip():
        raise HTTPException(400, "No text found in that file (a scanned PDF has none to extract).")
    if len(text) > MAX_DOCUMENT_CHARS:
        raise HTTPException(400, f"That document is too long ({len(text):,} characters; the limit is {MAX_DOCUMENT_CHARS:,}).")
    base = TTSRequest(text="x", **json.loads(settings))
    resolve_voice(base.voice)
    title = os.path.splitext(file.filename or "document")[0]

    def read_chapter(chapter_title: str, chapter_text: str, group: str) -> dict:
        req = base.model_copy(update={"text": chapter_text})
        t0 = time.perf_counter()
        results = []
        for item in plan(req):
            if item[0] == "pause":
                results.append({"kind": "pause", "seconds": item[1]})
            else:
                audio, rate = speak_item(item, req)
                results.append({"kind": "speech", "voice": item[1], "text": item[2], "audio": audio, "sample_rate": rate})
        return library.save(title=f"{title}: {chapter_title}", kind="chapter", text=chapter_text, voice=req.voice,
                            settings=req.settings(), items=results, generation_seconds=time.perf_counter() - t0,
                            group=group)

    return jobs.start(title, split_chapters(text), read_chapter)


@app.get("/jobs")
def jobs_list():
    return sorted(jobs.jobs.values(), key=lambda j: j["started"], reverse=True)


@app.get("/jobs/{job_id}")
def jobs_get(job_id: str):
    if job_id not in jobs.jobs:
        raise HTTPException(404, "No such job.")
    return jobs.get(job_id)


@app.post("/jobs/{job_id}/cancel")
def jobs_cancel(job_id: str):
    if job_id not in jobs.jobs:
        raise HTTPException(404, "No such job.")
    return jobs.cancel(job_id)


@app.get("/documents/{group}/zip")
def documents_zip(group: str, format: str = Query("mp3", pattern="^(wav|mp3|flac|opus)$")):
    """Every chapter of one document, as numbered audio files in a ZIP."""
    chapters = sorted((m for m in library.list() if m.get("group") == group), key=lambda m: m["created"])
    if not chapters:
        raise HTTPException(404, "No chapters for that document yet.")
    files = []
    for n, m in enumerate(chapters, 1):
        audio, rate = sf.read(library.full_path(m["id"]), dtype="float32")
        safe = re.sub(r'[\\/:*?"<>|]', "", m["title"])[:60]
        files.append((f"{n:02d} - {safe}.{format}", encode(audio, rate, format)))
    return Response(zip_chapters(files), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="document-{group}.zip"'})


# --- Voice recording quality check (before saving a voice)

@app.post("/voices/check")
def voices_check(files: list[UploadFile] = File(...)):
    import tempfile

    results = []
    for f in files:
        suffix = os.path.splitext(f.filename or "")[1].lower() or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(f.file.read())
        try:
            results.append({"file": f.filename, **check_recording(Path(tmp.name))})
        except Exception as e:
            results.append({"file": f.filename, "ok": False, "warnings": [f"לא ניתן לקרוא את הקובץ: {e}"],
                            "codes": ["unreadable"], "error": str(e)})
        finally:
            os.unlink(tmp.name)
    return results


# --- Status of everything, for the UI's status panel

@app.get("/status")
def status():
    from engines import OnDemand, RemoteEngine, ZonosEngine

    rows = []
    for e in engines.values():
        kind = "worker" if isinstance(e, RemoteEngine) else "remote" if isinstance(e, ZonosEngine) else "local"
        gpu = isinstance(e, OnDemand)
        loaded = e.loaded if gpu else True
        alive = e._info() is not None if isinstance(e, RemoteEngine) and loaded else loaded
        idle = round((time.monotonic() - e.last_used) / 60, 1) if gpu and loaded else None
        rows.append({"name": e.name, "label": e.label, "kind": kind, "voices": len(e.voices), "alive": alive,
                     "gpu": gpu, "loaded": loaded, "idle_minutes": idle})
    gpu = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu = {"name": torch.cuda.get_device_name(0), "used_gb": round((total - free) / 1e9, 1), "total_gb": round(total / 1e9, 1)}
    return {
        "engines": rows, "gpu": gpu, "library_items": len(library.list()), "memory_cache": len(api_cache.items),
        "jobs_running": sum(1 for j in jobs.jobs.values() if j["status"] == "running"),
        "auth": bool(API_KEY), "auto_unload_minutes": auto_unload["minutes"],
    }


@app.get("/tts")
def tts_get(
    text: str = Query(..., min_length=1, max_length=5000),
    voice: str = DEFAULT_VOICE,
    speed: float = Query(1.0, ge=0.5, le=2.0),
    format: str = "wav",
):
    return tts(TTSRequest(text=text, voice=voice, speed=speed, format=format))


# --- OpenAI-compatible API, so tools that speak OpenAI TTS (Open WebUI, Home Assistant,
# n8n, the openai SDK...) can use this server by pointing base_url at http://<host>:8010/v1

# OpenAI's voice names, mapped to the closest voices here (female/male as OpenAI's are).
OPENAI_VOICES = {
    "alloy": "lightblue/noa", "nova": "lightblue/noa", "shimmer": "lightblue/lily", "coral": "lightblue/lily",
    "sage": "omnivoice/female", "fable": "lightblue/daniel", "echo": "lightblue/adam", "ash": "lightblue/adam",
    "onyx": "kokoro/shaul", "ballad": "lightblue/daniel", "verse": "lightblue/adam",
}
OPENAI_MODELS = {"tts-1", "tts-1-hd", "gpt-4o-mini-tts"}


class OpenAISpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str = Field(..., min_length=1, max_length=5000)
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    instructions: str | None = None  # accepted for compatibility; not used


def resolve_openai_voice(model: str, voice: str) -> str:
    """'lightblue/noa' as-is; model=<engine> + voice=<name>; an OpenAI voice name; or any
    voice of that name in the first engine that has it."""
    if "/" in voice:
        return voice
    if model in engines:
        return f"{model}/{voice}"
    if voice in OPENAI_VOICES and resolve_voice_id_exists(OPENAI_VOICES[voice]):
        return OPENAI_VOICES[voice]
    for engine in engines.values():
        if voice in engine.voices:
            return f"{engine.name}/{voice}"
    return voice  # resolve_voice reports it as unknown


def resolve_voice_id_exists(voice_id: str) -> bool:
    engine_name, _, name = voice_id.partition("/")
    return engine_name in engines and name in engines[engine_name].voices


@app.post("/v1/audio/speech")
def openai_speech(req: OpenAISpeechRequest):
    """OpenAI's POST /v1/audio/speech. Formats: mp3, wav, flac, opus, pcm (aac isn't
    available here). Speed is clamped to this server's 0.5-2.0."""
    fmt = req.response_format.lower()
    if fmt not in MEDIA_TYPES:
        raise HTTPException(400, f"response_format '{fmt}' isn't supported; use {', '.join(MEDIA_TYPES)}")
    voice_id = resolve_openai_voice(req.model, req.voice)
    return tts(TTSRequest(text=req.input, voice=voice_id, speed=min(max(req.speed, 0.5), 2.0), format=fmt))


@app.get("/v1/models")
def openai_models():
    ids = sorted(OPENAI_MODELS) + list(engines)
    return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "local"} for i in ids]}


@app.get("/v1/audio/models")
def openai_audio_models():
    return {"models": [{"id": i} for i in sorted(OPENAI_MODELS) + list(engines)]}


@app.get("/v1/audio/voices")
def openai_audio_voices():
    """Every voice id here plus OpenAI's names, for tools that list voices."""
    ids = [v["id"] for v in list_voices()] + [n for n, v in OPENAI_VOICES.items() if resolve_voice_id_exists(v)]
    return {"voices": [{"id": i, "name": i} for i in ids]}


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
