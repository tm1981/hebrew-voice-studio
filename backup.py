"""Backup and restore of the user's data (DATA_DIR) as one zip file, made and restored from the
web UI (Settings), since in Docker the data sits inside a volume:

    hebrew-voice-backup.json   {"version", "created", "library"}
    settings.json presets.json pronunciations.json api_key.txt
    voices/...                 the custom voices (recordings), without voices/.cache
    outputs/library/...        the library of readings, if asked for (it can be large)

Restoring adds to what is there: a voice or reading of the same name is replaced, and the JSON
files and the key are handed back to the server to merge and apply (see restore).

Only the standard library here, like paths.py.
"""
import json
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import IO, Callable

from paths import DATA_DIR

MANIFEST = "hebrew-voice-backup.json"
JSON_FILES = ("settings.json", "presets.json", "pronunciations.json")
KEY_FILE = "api_key.txt"
VOICES = "voices"
LIBRARY = ("outputs", "library")


class BackupError(ValueError):
    pass


def write_backup(target: Path, library: bool) -> None:
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.writestr(MANIFEST, json.dumps({"version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                         "library": library}))
        for name in (*JSON_FILES, KEY_FILE):
            if (DATA_DIR / name).is_file():
                z.write(DATA_DIR / name, name)
        for folder in [DATA_DIR / VOICES] + ([DATA_DIR.joinpath(*LIBRARY)] if library else []):
            for path in sorted(folder.rglob("*")) if folder.is_dir() else []:
                rel = path.relative_to(DATA_DIR)
                if path.is_file() and not any(part.startswith(".") for part in rel.parts):
                    z.write(path, rel.as_posix())


def restore(source: Path | IO[bytes], clear_voice: Callable[[str], bool]) -> dict:
    """Unpacks the voices and readings of a backup into DATA_DIR. clear_voice(name) is called
    before a voice's files are written, so the restored recordings replace the old ones; False
    skips that voice (e.g. a name taken by a built-in voice).
    Returns {"json": {file name: its content}, "api_key": str | None, "voices": [names],
    "library": [reading ids], "skipped": count}: the JSON files and the key are left to the caller."""
    try:
        z = zipfile.ZipFile(source)
    except (zipfile.BadZipFile, OSError, ValueError):
        raise BackupError("That file isn't a backup from this app (not a zip file).")
    with z:
        if MANIFEST not in z.namelist():
            raise BackupError("That zip file isn't a backup from this app.")
        out = {"json": {}, "api_key": None, "voices": set(), "library": set(), "skipped": 0}
        files = []  # (member, parts) to unpack
        for info in z.infolist():
            name = info.filename
            parts = PurePosixPath(name).parts
            if info.is_dir() or name == MANIFEST:
                continue
            if name in JSON_FILES:
                try:
                    content = json.loads(z.read(info).decode("utf-8"))
                except ValueError:
                    content = None
                if isinstance(content, dict):
                    out["json"][name] = content
                else:
                    out["skipped"] += 1
            elif name == KEY_FILE:
                out["api_key"] = z.read(info).decode("utf-8", "replace").strip()
            elif not safe(parts):
                out["skipped"] += 1
            elif parts[0] == VOICES and len(parts) in (2, 3):
                out["voices"].add(voice_name(parts))
                files.append((info, parts))
            elif parts[:2] == LIBRARY and len(parts) == 4:
                out["library"].add(parts[2])
                files.append((info, parts))
            else:
                out["skipped"] += 1
        refused = {name for name in out["voices"] if not clear_voice(name)}
        out["voices"] -= refused
        out["skipped"] += sum(1 for _, parts in files if parts[0] == VOICES and voice_name(parts) in refused)
        files = [(info, parts) for info, parts in files if parts[0] != VOICES or voice_name(parts) not in refused]
        root = DATA_DIR.resolve()
        for info, parts in files:
            target = DATA_DIR.joinpath(*parts)
            if root not in target.resolve().parents:
                out["skipped"] += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
    out["voices"], out["library"] = sorted(out["voices"]), sorted(out["library"])
    return out


def voice_name(parts: tuple[str, ...]) -> str:
    """voices/<name>.<ext> or voices/<name>/<clip> -> name"""
    return PurePosixPath(parts[1]).stem if len(parts) == 2 else parts[1]


def safe(parts: tuple[str, ...]) -> bool:
    """A path inside the backup that may be written: relative, no "..", no hidden or drive parts."""
    return bool(parts) and all(p and p not in ("..", ".") and not p.startswith(".") and not any(c in p for c in ':\\/')
                               for p in parts)
