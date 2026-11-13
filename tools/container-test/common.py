"""Shared by the check_*.py scripts here: where the test container is and how to call it.

The container (start.sh) answers on localhost:8092. A request that comes in through Docker's port
mapping doesn't count as local, so the app wants its API key: it is read from the container.
Set VOICE_TEST_URL / VOICE_TEST_CONTAINER to check another copy.
"""
import os
import subprocess
import tempfile
from pathlib import Path

import requests

URL = os.environ.get("VOICE_TEST_URL", "http://localhost:8092")
CONTAINER = os.environ.get("VOICE_TEST_CONTAINER", "voice-wsl")
DOCKER = ["wsl", "docker"] if os.name == "nt" else ["docker"]
OUT = Path(tempfile.gettempdir()) / "voice-container-test"  # the readings the checks make, to listen to
OUT.mkdir(exist_ok=True)


def run(*command: str) -> str:
    return subprocess.run(command, capture_output=True, text=True).stdout.strip()


HEADERS = {"X-API-Key": run(*DOCKER, "exec", CONTAINER, "cat", "/data/api_key.txt")}


def get(path: str, **kwargs):
    return requests.get(URL + path, headers=HEADERS, **kwargs)


def post(path: str, **kwargs):
    return requests.post(URL + path, headers=HEADERS, **kwargs)


def voices() -> list[str]:
    return [v["id"] for v in get("/voices").json()]


def read(text: str, voice: str, timeout: float = 2400, **more):
    """One reading that isn't taken from the cache. -> the response (WAV in .content when .ok)."""
    return post("/tts", json={"text": text, "voice": voice, "use_cache": False, "options": {}, **more}, timeout=timeout)


def gpu_used_mb() -> int:
    """Memory in use on the whole GPU, as nvidia-smi says."""
    command = ["wsl", "nvidia-smi"] if os.name == "nt" else ["nvidia-smi"]
    return int(run(*command, "--query-gpu=memory.used", "--format=csv,noheader,nounits").splitlines()[0])


def free_gpu() -> None:
    post("/gpu/free", timeout=180)
