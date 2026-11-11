"""Saved readings: each one keeps its sentences as separate WAVs so a single sentence can
be regenerated or edited later, and doubles as the cache for repeated requests.

outputs/library/<id>/
    meta.json   {"id", "created", "title", "kind", "text", "voice", "settings", "cache_key",
                 "group", "sample_rate", "items": [...], "audio_seconds", "generation_seconds"}
    s<N>.wav    one per speech item; "items" lists speech {"kind", "text", "voice", "file",
                "seconds"} and pause {"kind": "pause", "seconds"} entries in reading order
    full.flac   the whole reading, rebuilt whenever an item changes (lossless, about half the size
                of the WAV it used to be; readings saved before that keep their full.wav)

The list of readings is kept in memory (Library.index): the page and /status ask for it often, and
reading every meta.json each time got slower with every reading saved. A reading can be pinned;
prune() deletes unpinned readings older than a number of days.
"""
from __future__ import annotations  # the class has a method called list

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
        """Reads the folder again, e.g. after a backup was restored into it: the list of readings
        and which ones are cached. An entry missing what the list needs is left out, not fatal."""
        index, by_key = {}, {}
        for meta in self._all_meta():
            try:
                index[meta["id"]] = self._summary(meta)
            except (KeyError, TypeError):
                continue
            if meta.get("cache_key"):
                by_key[meta["cache_key"]] = meta["id"]
        with self.lock:
            self.index, self.by_key = index, by_key

    def _summary(self, meta: dict) -> dict:
        """What the list shows for one reading, with its size on disk."""
        summary = {k: meta.get(k) for k in (
            "id", "title", "kind", "text", "voice", "group", "audio_seconds", "generation_seconds")}
        summary["created"] = float(meta["created"])
        summary["sentences"] = sum(1 for i in meta["items"] if i["kind"] == "speech")
        summary["pinned"] = bool(meta.get("pinned"))
        summary["bytes"] = sum(f.stat().st_size for f in self._dir(meta["id"]).iterdir() if f.is_file())
        return summary

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
        """Newest first."""
        return sorted((dict(s) for s in list(self.index.values())), key=lambda s: s["created"], reverse=True)

    def size(self) -> int:
        """Bytes on disk."""
        return sum(s["bytes"] for s in list(self.index.values()))

    def pin(self, item_id: str, pinned: bool) -> dict:
        """A pinned reading is never deleted by prune()."""
        with self.lock:
            meta = self.get(item_id)
            meta["pinned"] = bool(pinned)
            (self._dir(item_id) / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            self.index[item_id] = self._summary(meta)
            return meta

    def prunable(self, days: float, now: float | None = None) -> list[str]:
        """Ids of the unpinned readings made more than `days` days ago."""
        before = (now or time.time()) - days * 86400
        return [s["id"] for s in list(self.index.values()) if not s["pinned"] and s["created"] < before]

    def prune(self, days: float, now: float | None = None) -> list[str]:
        """Deletes them. -> the ids deleted."""
        deleted = []
        for item_id in self.prunable(days, now):
            try:
                self.delete(item_id)
                deleted.append(item_id)
            except (KeyError, OSError):
                continue
        return deleted

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
        flac = self._dir(item_id) / "full.flac"
        return flac if flac.exists() else self._dir(item_id) / "full.wav"

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
        self.index.pop(item_id, None)

    def _write(self, meta: dict) -> None:
        """Rebuild the full file from the items, then write meta.json (last, so a crash mid-way
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
            item["start"] = round(at / rate, 3)  # where it starts in the full file, for marking it as it plays
            parts.append(sf.read(folder / item["file"], dtype="float32")[0])
            at += len(parts[-1])
            previous_speech = True
        full = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        sf.write(folder / "full.flac", full, rate, format="FLAC", subtype="PCM_16")
        (folder / "full.wav").unlink(missing_ok=True)  # a reading saved before full files were FLAC
        meta["audio_seconds"] = round(len(full) / rate, 2)
        for stale in folder.glob("s*.wav"):
            if stale.name not in {i.get("file") for i in meta["items"]}:
                stale.unlink()
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        self.index[meta["id"]] = self._summary(meta)


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
