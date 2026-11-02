"""HTTP protocol shared by engine workers (spoken by engines.RemoteEngine).

A worker runs in its own conda env, so this file only uses packages every env has:
fastapi, uvicorn, numpy, soundfile.

    GET    /info                                  -> sample_rate, voices, builtin_voices, native_speed, custom_voices
    POST   /synthesize {"text", "voice", "speed", "options"} -> WAV (PCM 16-bit); options are the
           engine's settings (engines.WORKER_OPTIONS), already cleaned by the main server
    POST   /voices {"name", "path"}               -> build a voice from a recording on disk
    DELETE /voices/{name}
    POST   /shutdown                              -> exit (the server unloading this engine)

The main server registers custom voices through POST /voices after /info answers, so the
voice-folder logic (clips, trimming, caching paths) lives in one place, engines.py.
"""
import argparse
import io
import os
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel


class SynthesizeRequest(BaseModel):
    text: str
    voice: str
    speed: float = 1.0
    options: dict = {}


class VoiceRequest(BaseModel):
    name: str
    path: str


def serve(backend, description: str) -> None:
    """backend: sample_rate, voices (dict), builtin_voices (set), native_speed, custom_voices,
    load(), synthesize(text, voice, speed, **options) -> np.ndarray, add_voice(name, path), remove_voice(name)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    backend.load()

    # Model calls run as plain code inside async endpoints, i.e. on the main thread's event
    # loop, one at a time: torch.compile (Higgs) can crash or recompile when it runs in a
    # thread-pool thread, and a worker only does one generation at a time anyway.
    app = FastAPI(title=description)

    @app.get("/info")
    def info():
        return {
            "sample_rate": backend.sample_rate,
            "voices": list(backend.voices),
            "builtin_voices": sorted(backend.builtin_voices),
            "native_speed": backend.native_speed,
            "custom_voices": backend.custom_voices,
        }

    @app.post("/synthesize")
    async def synthesize(req: SynthesizeRequest):
        if req.voice not in backend.voices:
            raise HTTPException(400, f"No voice '{req.voice}'")
        audio = np.asarray(backend.synthesize(req.text, req.voice, req.speed, **req.options), dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, backend.sample_rate, format="WAV", subtype="PCM_16")
        return Response(buf.getvalue(), media_type="audio/wav")

    @app.post("/voices")
    async def add_voice(req: VoiceRequest):
        if not backend.custom_voices:
            raise HTTPException(400, "This engine has fixed voices")
        try:
            backend.add_voice(req.name, Path(req.path))
        except Exception as e:
            raise HTTPException(400, str(e))
        return {"ok": True}

    @app.delete("/voices/{name}")
    def remove_voice(name: str):
        if backend.custom_voices and name not in backend.builtin_voices:  # Magpie has no remove_voice
            backend.remove_voice(name)
        return {"ok": True}

    @app.post("/shutdown")
    def shutdown():
        """The main server unloading this engine: exit (just after answering), which frees the GPU memory."""
        threading.Timer(0.3, os._exit, args=(0,)).start()
        return {"ok": True}

    print(f"{description} ready on port {args.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
