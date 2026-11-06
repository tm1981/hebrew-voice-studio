"""Breeze TTS 2 worker (conda env 'breeze'; code from BreezeBlue's breeze-tts, Apache 2.0, in
vendor/breeze-tts, see tools/fetch_vendor.py). Weights are BreezeBlue/Breeze-TTS-2, research and
non-commercial use only, in the Hugging Face cache (models/hf, see HF_HOME in server.py) at a
pinned revision, 7.2 GB. English and Chinese only: Hebrew text comes out as nonsense
(Whisper heard none of 3 sentences), but a Hebrew recording works as the voice for English.

Two memory modes, with the same voices and settings:
  (default) eager, 2.3-2.6 s per second of audio on an RTX 5090, 9.2 GiB of GPU memory.
  --fast    CUDA graphs, 0.31-0.38 s per second of audio, 16-19 GiB, 25-60 s more to start.
            Not the default for an RTX 3090 (24 GB): Whisper on a voice's first reading takes it
            to 20.4 GiB, which beside the app's 3 GB leaves under 0.5 GiB, and no room for
            another engine. Eager peaks at 12.9 GiB with Whisper.

Voices:
  default   - no recording. NOT a fixed voice: the model invents a speaker for every reading, so
              sentences of one text come out in different voices (83 to 274 Hz in testing). An
              instruction, with or without a recording, describes the voice instead.
  <custom>  - cloned from a recording in voices/. The model needs the clip's exact words, so the
              first reading with a voice runs reference.py (Whisper); later ones use its cache.

Without a seed every take is a little different; the same seed gives the same reading.

Tested on 8 English sentences, 2 seeds: no words wrong in 16 for the default and for a female
clone, 2 of 16 for a male clone. Whisper's Hebrew transcript works as the reference; a transcript
in Latin letters was no better. At each edge of the settings 0 to 3 of 8 clones had a word wrong.
"""
import argparse
import os
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

# Windows only, before torch loads: --fast dies in torch.compile's static launcher, whose kernel
# arguments overflow Windows' 32-bit C long ("Python int too large to convert to C long").
if os.name == "nt":
    os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")

import torch

from worker_common import serve

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "breeze-tts"
sys.path.insert(0, str(VENDOR))  # its packages are called models and breeze_infer

REPO = "BreezeBlue/Breeze-TTS-2"
REVISION = "3e28c5151381a722f1d8661b4118c298caa77aa4"
MAX_NEW_TOKENS = 1500  # the rest are breeze-tts's own infer.py values
MAX_SEQ_LEN = 2048


class BreezeBackend:
    sample_rate = 24000
    native_speed = False  # the server stretches the audio
    custom_voices = True

    def __init__(self, fast: bool):
        self.fast = fast
        self.voices: dict = {"default": None}
        self.builtin_voices = {"default"}

    def load(self):
        from breeze_infer.runtime import load_runtime
        from huggingface_hub import snapshot_download
        from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
        from models.warmup_profile import load_warmup_profile

        print(f"Loading {REPO} ({'fast' if self.fast else 'eager'})...", flush=True)
        checkpoint = Path(snapshot_download(REPO, revision=REVISION, ignore_patterns=["assets/*"]))
        # flash_attention_2 isn't needed: breeze-tts's own infer.py and API load with eager too
        self.tokenizer, self.model, self.audio_tokenizer = load_runtime(checkpoint, device="cuda:0",
                                                                        attn_implementation="eager")
        config = FastStreamingConfig(max_new_tokens=MAX_NEW_TOKENS, max_seq_len=MAX_SEQ_LEN,
                                     fast_all=True if self.fast else None)
        self.runtime = FastBreezeStreamingRuntime(self.model, self.audio_tokenizer, config, tokenizer=self.tokenizer)
        if self.fast:
            profile = load_warmup_profile(VENDOR / "configs" / "fast.json")
            self.runtime.warmup_from_profile(replace(profile, codec_chunk_frames=self.runtime.codec_chunk_frames))
        self.synthesize("Hello, how are you today?", "default", 1.0)  # warm-up: the first reading takes 15 s more

    def add_voice(self, name: str, path: Path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        # Whisper waits for the first reading: the server registers every voice at each start.
        self.voices[name] = {"path": Path(path), "clip": None, "words": None}

    def remove_voice(self, name: str):
        self.voices.pop(name, None)

    def _reference(self, voice: str) -> dict:
        entry = self.voices[voice]
        if entry["clip"] is None:
            from reference import reference

            entry["clip"], entry["words"] = reference(entry["path"])
        return entry

    def synthesize(self, text: str, voice: str, speed: float, temperature=0.9, top_k=50, repetition_penalty=1.1,
                   guidance=4.0, instruction="", seed=None, **_):
        from breeze_infer.runtime import set_all_seeds, update_generation_config_for_breeze
        from breeze_infer.templates import get_template, prepare_inputs, select_template_name

        # temperature 1.1 and above raises the floor between words by 5 dB. guidance stretches an
        # instructed reading (1: 5.5 s, 6: 8.8 s of audio) and cleans the floor from 2 up.
        seed = random.randrange(2 ** 31) if seed is None else int(seed)
        instruction = instruction.strip()
        request = {"id": "reading", "text": text, "speaker": "S0"}
        if instruction:
            request["instruction"] = instruction
        if self.voices[voice] is not None:
            entry = self._reference(voice)
            request["ref_audio_path"], request["ref_text"] = str(entry["clip"]), entry["words"]
        # Read for each reading in both modes, so these settings need no graph capture.
        update_generation_config_for_breeze(self.model, {
            "do_sample": True, "temperature": float(temperature), "top_k": int(top_k), "top_p": 1.0,
            "depth_decoder_do_sample": True, "depth_decoder_temperature": float(temperature),
            "depth_decoder_top_k": int(top_k), "depth_decoder_top_p": 1.0,
            "max_new_tokens": MAX_NEW_TOKENS})
        self.runtime.config = replace(self.runtime.config, repetition_penalty=float(repetition_penalty))
        set_all_seeds(seed)
        # Guidance needs an instruction to steer by: without one the templates refuse any scale but 1.
        inputs = prepare_inputs(self.tokenizer, self.audio_tokenizer, self.model, [request],
                                get_template(select_template_name(request)),
                                guidance_scale=float(guidance) if instruction else 1.0,
                                guidance_scale_ref=None, guidance_scale_ins=None)
        chunks = [c.audio for c in self.runtime.iter_audio_chunks(inputs, request_id="reading", seed=seed)]
        if not chunks:
            raise RuntimeError("Breeze ended the reading without any sound")
        return np.concatenate(chunks).astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    fast, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]  # worker_common reads --port
    serve(BreezeBackend(fast.fast), f"Breeze TTS 2{' fast' if fast.fast else ''} worker")
