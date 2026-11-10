"""ZONOS3-Mini worker (runs in the server's own env; Linux only).

Zyphra's small text-to-speech model (800M parameters, Apache-2.0), through its C++/ggml server
built with CUDA in the Docker image (/opt/zonos3). This worker starts that server on a free
local port and passes requests on. The model's PyTorch server needs kernels that only exist for
some GPUs, so the portable ggml build with the quantized weights is what runs here: about 7x
real time and about 3.4 GB of GPU memory on an RTX 5090.

Hebrew is not on the model's language list, but it reads Hebrew well when the voice is cloned
from a Hebrew recording (0-1 of 30 test sentences with words added or dropped); with its own
English voices Hebrew comes out as babble (13 of 15). So the voices are only the recordings in
voices/, by speaker embedding.

Pace: the model takes a speaking rate in UTF-8 bytes per second (Hebrew letters are 2 bytes).
Left unset it reads Hebrew slowly, about 14; 18 sounds like the other engines. From 22 up words
start to go missing (2 of 5 at 22, 5 of 5 at 26), so speed is kept inside that range.
"""
import atexit
import base64
import ctypes
import io
import json
import os
import signal
import site
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download

from worker_common import serve

SERVER = Path(os.environ.get("ZONOS3_SERVER", "/opt/zonos3/zonos3-server"))
REPO = "Zyphra/ZONOS3-Mini"
FILES = ["gguf/zonos3-ptq.gguf", "gguf/zac-keep2.gguf", "gguf/spk-encoder.gguf"]  # the server finds the last two beside the first
RATE, MIN_RATE, MAX_RATE = 18.0, 12.0, 21.0


def die_with_parent() -> None:
    """The ggml server must not outlive this worker, which exits abruptly when the engine is unloaded."""
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


class Zonos3Backend:
    native_speed = True
    custom_voices = True
    sample_rate = 44100

    def __init__(self):
        self.voices: dict = {}
        self.builtin_voices: set = set()

    def load(self):
        print("Loading ZONOS3-Mini...", flush=True)
        model = [hf_hub_download(REPO, f) for f in FILES][0]
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        # The server's CUDA libraries are the nvidia-* wheels of this env (the image has no CUDA
        # toolkit); its own ggml libraries sit beside it.
        libs = [str(SERVER.parent), *(str(p) for folder in site.getsitepackages() for p in Path(folder, "nvidia").glob("*/lib"))]
        self.process = subprocess.Popen([str(SERVER), model, "--gpu", "--port", str(port)], preexec_fn=die_with_parent,
                                        env={**os.environ, "LD_LIBRARY_PATH": ":".join(libs)})
        atexit.register(self.process.kill)
        deadline = time.monotonic() + 600  # the first start warms up the model, about a minute
        while True:
            try:
                urllib.request.urlopen(f"{self.url}/v2/capabilities", timeout=2).read()
                break
            except (urllib.error.URLError, OSError):
                if self.process.poll() is not None:
                    raise RuntimeError(f"the ZONOS3 server exited with code {self.process.returncode}")
                if time.monotonic() > deadline:
                    raise RuntimeError("the ZONOS3 server didn't start within 10 minutes")
                time.sleep(1)

    def add_voice(self, name: str, path: Path):
        self.voices[name] = base64.b64encode(Path(path).read_bytes()).decode()

    def remove_voice(self, name: str):
        self.voices.pop(name, None)

    def synthesize(self, text: str, voice: str, speed: float, seed=None, **_):
        # The model stops right on the last sound, so a reading's end is clipped: 14 of 15 test
        # readings ended while the voice was still sounding. Streaming, the other condition
        # values, more or fewer prefill frames and other endings ("...", " .") changed nothing or
        # less; blank lines after the text did best (7 and 11 of 15, fewest Whisper errors).
        # The short fade takes the click off what is left.
        body = {"text": text.rstrip() + "\n\n", "stream": False, "format": "wav", "fade_out_ms": 40,
                "speaker_audio_base64": self.voices[voice], "speaker_mode": "embedding",
                "condition_values": {"speaking_rate": min(max(RATE * speed, MIN_RATE), MAX_RATE)}}
        if seed is not None:
            body["seed"] = seed
        request = urllib.request.Request(f"{self.url}/v2/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            data = urllib.request.urlopen(request, timeout=570).read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(e.read().decode(errors="replace")[:300])
        audio, self.sample_rate = sf.read(io.BytesIO(data), dtype="float32")
        return audio


if __name__ == "__main__":
    serve(Zonos3Backend(), "ZONOS3-Mini worker")
