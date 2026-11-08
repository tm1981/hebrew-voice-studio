"""Saved readings: each one keeps its sentences as separate WAVs so a single sentence can
be regenerated or edited later, and doubles as the cache for repeated requests.

outputs/library/<id>/
    meta.json   {"id", "created", "title", "kind", "text", "voice", "settings", "cache_key",
                 "group", "sample_rate", "items": [...], "audio_seconds", "generation_seconds"}
    s<N>.wav    one per speech item; "items" lists speech {"kind", "text", "voice", "file",
                "seconds"} and pause {"kind": "pause", "seconds"} entries in reading order
    full.wav    the whole reading, rebuilt whenever an item changes
"""
import hashlib
import json
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf

from paths import DATA_DIR

LIBRARY_DIR = DATA_DIR / "outputs" / "library"
SENTENCE_GAP = 0.25  # seconds between consecutive sentences, as in engines.join_with_pauses


def cache_key(**parts) -> str:
    """Stable hash of everything that changes the audio (not the output format)."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:32]


def to_rate(audio: np.ndarray, sample_rate: int, target: int) -> np.ndarray:
    if sample_rate == target:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=sample_rate, target_sr=target)


class Library:
    def __init__(self, folder: Path = LIBRARY_DIR):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.reindex()

    def reindex(self) -> None:
        """Reads which readings are cached again, e.g. after a backup was restored into the folder."""
        by_key = {meta["cache_key"]: meta["id"] for meta in self._all_meta() if meta.get("cache_key")}
        with self.lock:
            self.by_key = by_key

    def _all_meta(self) -> list[dict]:
        metas = []
        for path in self.folder.glob("*/meta.json"):
            try:
                metas.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue  # half-written or damaged entry: skip it
        return metas

    def _dir(self, item_id: str) -> Path:
        path = (self.folder / item_id).resolve()
        if path.parent != self.folder.resolve():  # ids come from URLs
            raise KeyError(item_id)
        return path

    def get(self, item_id: str) -> dict:
        path = self._dir(item_id) / "meta.json"
        if not path.exists():
            raise KeyError(item_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        summaries = []
        for m in sorted(self._all_meta(), key=lambda m: m["created"], reverse=True):
            summaries.append({k: m.get(k) for k in (
                "id", "created", "title", "kind", "text", "voice", "group", "audio_seconds", "generation_seconds")})
            summaries[-1]["sentences"] = sum(1 for i in m["items"] if i["kind"] == "speech")
        return summaries

    def find(self, key: str) -> dict | None:
        item_id = self.by_key.get(key)
        if not item_id:
            return None
        try:
            return self.get(item_id)
        except KeyError:
            self.by_key.pop(key, None)
            return None

    def sentence_audio(self, meta: dict, index: int) -> tuple[np.ndarray, int]:
        audio, sr = sf.read(self._dir(meta["id"]) / meta["items"][index]["file"], dtype="float32")
        return audio, sr

    def full_path(self, item_id: str) -> Path:
        return self._dir(item_id) / "full.wav"

    def save(self, *, title, kind, text, voice, settings, items, generation_seconds, key=None, group=None,
             readalong=None) -> dict:
        """items: [{"kind": "speech", "text", "voice", "audio", "sample_rate", "words"?} | {"kind": "pause", "seconds"}]
        ("words": when each word starts and ends, from the "mark words" option). readalong: the
        typed text in pieces, each with the sentence and words it became (server.readalong)."""
        item_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        folder = self._dir(item_id)
        folder.mkdir(parents=True)
        rate = max((i["sample_rate"] for i in items if i["kind"] == "speech"), default=24000)
        stored = []
        for n, item in enumerate(items):
            if item["kind"] == "pause":
                stored.append({"kind": "pause", "seconds": item["seconds"]})
                continue
            file = f"s{n}.wav"
            audio = to_rate(item["audio"], item["sample_rate"], rate)
            sf.write(folder / file, audio, rate, subtype="PCM_16")
            stored.append({"kind": "speech", "text": item["text"], "voice": item["voice"], "file": file,
                           "seconds": round(len(audio) / rate, 2)})
            if item.get("words"):
                stored[-1]["words"] = item["words"]
        meta = {
            "id": item_id, "created": time.time(), "title": title, "kind": kind, "text": text, "voice": voice,
            "settings": settings, "cache_key": key, "group": group, "sample_rate": rate, "items": stored,
            "generation_seconds": round(generation_seconds, 2),
        }
        if readalong:
            meta["readalong"] = readalong
        self._write(meta)
        if key:
            self.by_key[key] = item_id
        return meta

    def replace_sentence(self, item_id: str, index: int, text: str, audio: np.ndarray, sample_rate: int) -> dict:
        with self.lock:
            meta = self.get(item_id)
            item = meta["items"][index]
            if item["kind"] != "speech":
                raise ValueError("That item is a pause, not a sentence.")
            audio = to_rate(audio, sample_rate, meta["sample_rate"])
            item["file"] = f"s{index}-{uuid.uuid4().hex[:4]}.wav"  # new name: browsers cache by URL
            sf.write(self._dir(item_id) / item["file"], audio, meta["sample_rate"], subtype="PCM_16")
            item["text"] = text
            item["seconds"] = round(len(audio) / meta["sample_rate"], 2)
            item.pop("words", None)  # the old take's word times; the page estimates until there are new ones
            self._write(meta)
            return meta

    def delete(self, item_id: str) -> None:
        meta = self.get(item_id)
        if meta.get("cache_key"):
            self.by_key.pop(meta["cache_key"], None)
        shutil.rmtree(self._dir(item_id))

    def _write(self, meta: dict) -> None:
        """Rebuild full.wav from the items, then write meta.json (last, so a crash mid-way
        leaves the previous meta pointing at files that still exist)."""
        folder = self._dir(meta["id"])
        rate = meta["sample_rate"]
        gap = np.zeros(int(rate * meta.get("settings", {}).get("sentence_pause", SENTENCE_GAP)), dtype=np.float32)
        parts, previous_speech, at = [], False, 0
        for item in meta["items"]:
            if item["kind"] == "pause":
                item["start"] = round(at / rate, 3)
                parts.append(np.zeros(int(rate * item["seconds"]), dtype=np.float32))
                at += len(parts[-1])
                previous_speech = False
                continue
            if previous_speech:
                parts.append(gap)
                at += len(gap)
            item["start"] = round(at / rate, 3)  # where it starts in full.wav, for marking it as it plays
            parts.append(sf.read(folder / item["file"], dtype="float32")[0])
            at += len(parts[-1])
            previous_speech = True
        full = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        sf.write(folder / "full.wav", full, rate, subtype="PCM_16")
        meta["audio_seconds"] = round(len(full) / rate, 2)
        for stale in folder.glob("s*.wav"):
            if stale.name not in {i.get("file") for i in meta["items"]}:
                stale.unlink()
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


class MemoryCache:
    """Small LRU of finished audio for API calls (/tts, /v1/audio/speech), which aren't
    saved to the library so tools calling the API don't fill it."""

    def __init__(self, size: int = 200):
        self.size = size
        self.items: OrderedDict[str, tuple[np.ndarray, int]] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str):
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                return self.items[key]
        return None

    def put(self, key: str, audio: np.ndarray, sample_rate: int) -> None:
        with self.lock:
            self.items[key] = (audio, sample_rate)
            self.items.move_to_end(key)
            while len(self.items) > self.size:
                self.items.popitem(last=False)
