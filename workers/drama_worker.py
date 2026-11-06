"""Fish Audio Drama 3 worker (conda env 'drama'). 35B multilingual text-to-speech under the Fish
Audio Community License: free for personal and non-commercial use, and no cloning of a real
person's voice without their consent. Reads Hebrew and English, 44.1 kHz.

Runs ComfyUI-Drama3's model code directly, with ComfyUI as a library and no ComfyUI server. Both
are vendored (tools/fetch_vendor.py) and pinned together: that pack needs ComfyUI's Qwen 3.5 code
of this exact age, comfy-kitchen 0.2.38, and torch built for CUDA 13.x. On torch cu129 every
generation fails with "Unsupported Flash Attention tensor strides".

Weights: fishaudio/drama-3 is 41.5 GB of FP8 and doesn't fit a 24 GB card, and no 4-bit build of
its Oct 29 checkpoint exists in the format this code reads. So the first load downloads the
checkpoint at a pinned revision, converts it to W4A8 in models/drama3 with the pack's
tools/convert.py (about 2.5 minutes) and deletes the download. Later starts read the 24.6 GB file.
The pack asks for about 32 GB of system memory, because the routed experts stay in RAM.

Voices:
  default  - no reference: the model makes up a voice, a different one for each seed
  <custom> - cloned from a recording in voices/ with its transcript (reference.py), which is why
             the first reading with a new voice runs Whisper for a few seconds

Without a seed every take is a little different; the same seed gives the same reading.

Measured on an RTX 5090 through this worker, Whisper word check: 32 English readings cloned from
Hebrew speech, 0 wrong; 16 with the default voice, 0 wrong; 60 Hebrew readings, 5 with words added
or dropped and 23 misheard. The Oct 14 weights raised the floor between words in 8 of 16 English
clones, these in none. 0.54 s to generate each second of English, 0.77 for Hebrew, the same on
Linux in Docker. 21 GB of GPU memory at most, about 50 s to start.
"""
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMFY, PACK = ROOT / "vendor" / "ComfyUI", ROOT / "vendor" / "ComfyUI-Drama3"
sys.path[:0] = [str(ROOT), str(COMFY), str(PACK)]

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# The pack's CUDA kernels compile through NVRTC, which ships in the torch wheels, but torch refuses
# without a CUDA_HOME to name, and the pack then silently falls back to 4x slower code.
os.environ.setdefault("CUDA_HOME", sys.prefix)
if os.name == "nt":  # ComfyUI's start-up forces one GPU on Windows, and this was tested that way
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Order matters, as in ComfyUI's main.py: both must run before anything imports torch.
import cuda_malloc  # noqa: E402,F401  (sets the CUDA allocator)
import comfy_aimdo.control  # noqa: E402

from comfy.cli_args import args as comfy_args  # noqa: E402

comfy_aimdo.control.init(simple_vram_headroom=None, nvml_pressure=True)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

import comfy.audio  # noqa: E402
import comfy.memory_management  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.model_prefetch  # noqa: E402
import comfy_aimdo.model_vbar  # noqa: E402

from paths import MODELS_DIR  # noqa: E402
from worker_common import serve  # noqa: E402

REPO, REVISION = "fishaudio/drama-3", "78e5ab79e49aa9a25abe8799eff2c4d73a1c3216"
MODEL_FILE = MODELS_DIR / "drama3" / f"drama3_w4a8_{REVISION[:7]}.safetensors"
CODEC_FILE = MODELS_DIR / "drama3" / "drama3_codec.safetensors"

# GPU memory for the experts in use; the rest stay in system memory. Peak above idle: 25.4 GB with
# a cache of 16, 21 GB with 12, 12.7 GB with 4, so 12 fits a 24 GB RTX 3090 with nothing else on it.
EXPERT_CACHE_GB = 12

MAX_FRAMES = 600  # per chunk of text, about 28 s; a chunk is about 300 bytes
TAKE_GAP = 0.3  # seconds between chunks, as the pack's Decode node
MAX_PAUSE = 0.75  # Fish's own demo shortens longer pauses to this
MAX_TAKES = 2


def seconds_expected(text: str) -> float:
    """The least a reading of `text` can take; the model sometimes ends after the first clause."""
    return sum(0.06 if "֐" <= c <= "׿" else 0.03 for c in text if c.isalpha())


def ensure_weights() -> None:
    if MODEL_FILE.exists() and CODEC_FILE.exists():
        return
    from huggingface_hub import snapshot_download

    print(f"Downloading {REPO} and converting it to W4A8 (first use, about 40 GB)...", flush=True)
    source = Path(snapshot_download(REPO, revision=REVISION))
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    convert = [sys.executable, str(PACK / "tools" / "convert.py")]
    env = {**os.environ, "PYTHONPATH": str(COMFY), "PYTHONUTF8": "1"}
    subprocess.run([*convert, "codec", str(source / "codec.pth"), str(CODEC_FILE)], env=env, check=True)
    subprocess.run([*convert, "model", str(source), str(MODEL_FILE), "--format", "w4a8"], env=env, check=True)
    shutil.rmtree(source.parents[1])  # nothing reads the 41.5 GB original again


class DramaBackend:
    sample_rate = 44100
    native_speed = False  # the server stretches the audio
    custom_voices = True

    def __init__(self):
        self.voices: dict = {"default": None}
        self.builtin_voices = {"default"}
        self.references: dict = {}  # name -> (words, codes), filled on the voice's first reading

    def load(self):
        ensure_weights()
        # As ComfyUI's main.py does; the pack loads its model through this dynamic patcher.
        devices = comfy.model_management.get_all_torch_devices()
        headroom = int(comfy_args.vram_headroom * 1024 ** 3)
        if not comfy_aimdo.control.init_devices((d.index, headroom) for d in devices):
            raise RuntimeError("comfy-aimdo found no usable GPU")
        comfy_aimdo.control.set_log_warning()
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled = True

        from drama3.loader import load_codec, load_drama3

        print(f"Loading {MODEL_FILE.name}...", flush=True)
        with torch.inference_mode():
            self.model = load_drama3(str(MODEL_FILE), experts_in_ram=True, expert_cache_gb=EXPERT_CACHE_GB)
            self.codec = load_codec(str(CODEC_FILE))
        self.synthesize("Hello, how are you today?", "default", 1.0, seed=1, retake=False)  # warm-up

    def add_voice(self, name: str, path: Path):
        self.voices[name] = Path(path)
        self.references.pop(name, None)

    def remove_voice(self, name: str):
        self.voices.pop(name, None)
        self.references.pop(name, None)

    def _reference(self, voice: str):
        """(words, codes) of a custom voice. Whisper runs here, not in add_voice, which must stay quick."""
        if voice not in self.references:
            from drama3.audio import tidy_clip
            from drama3.codec import SAMPLE_RATE
            from reference import reference

            clip, words = reference(self.voices[voice])
            audio, rate = sf.read(str(clip), dtype="float32", always_2d=True)
            waveform = torch.from_numpy(audio.mean(1, keepdims=True).T)
            waveform = tidy_clip(comfy.audio.resample(waveform, rate, SAMPLE_RATE))
            comfy.model_management.load_models_gpu([self.codec.patcher])
            device = self.codec.patcher.load_device
            with comfy.model_management.cuda_device_context(device):
                codes = self.codec.model.encode(waveform.to(device))[0].cpu()
            self.references[voice] = (words, codes)
        return self.references[voice]

    def synthesize(self, text: str, voice: str, speed: float, seed=None, temperature=1.0, top_p=0.9, top_k=0,
                   retake=True, **_):
        # top_k 0 means no cap. Fish's server uses 30, which leaves takes stuck in silence at pauses.
        seed = random.randrange(2 ** 31) if seed is None else int(seed)
        with torch.inference_mode():
            comfy.model_management.cleanup_models_gc()
            references = [] if self.voices[voice] is None else [self._reference(voice)]
            best = None
            for attempt in range(MAX_TAKES if retake else 1):
                audio = self._take(text, references, seed + attempt, float(temperature), float(top_p), int(top_k))
                if best is None or len(audio) > len(best):
                    best = audio
                if len(best) / self.sample_rate >= seconds_expected(text):
                    break
        return best

    def _take(self, text, references, seed, temperature, top_p, top_k) -> np.ndarray:
        from drama3.audio import trim_pauses
        from drama3.chunking import split_text
        from drama3.codec import DECODE_CONTEXT, DECODE_FRAME_BYTES, DECODE_WINDOW, SAMPLE_RATE
        from drama3.generate import generate
        from drama3.prompt import build_prompt

        model, codec = self.model, self.codec
        semantic = model.config.semantic_start_token_id, model.config.semantic_end_token_id
        comfy.model_management.load_models_gpu(model.patchers())
        takes, codes = [], []
        try:
            with model.session():  # the chunks share one expert cache
                for index, chunk in enumerate(split_text(text) or [text]):
                    # The first and the latest chunk go back in, which keeps a long reading's voice steady.
                    prompt = build_prompt(model.tokenizer, model.config, chunk, references, takes[:1] + takes[1:][-1:])
                    frames = generate(model, prompt, MAX_FRAMES, temperature, top_k, top_p, seed + index)
                    keep = (frames[:, 0] >= semantic[0]) & (frames[:, 0] <= semantic[1])
                    if keep.any():
                        takes.append((chunk, frames))
                        codes.append(frames[keep, 1:].T)
        finally:
            # What ComfyUI's executor does after every node; nothing else would here.
            comfy.model_prefetch.cleanup_prefetch_queues()
            comfy.model_management.reset_cast_buffers()
            comfy_aimdo.model_vbar.vbars_reset_watermark_limits()
        if not codes:
            raise RuntimeError("Drama 3 ended the reading without any sound")
        window = min(max(c.shape[-1] for c in codes), DECODE_CONTEXT + DECODE_WINDOW)
        comfy.model_management.load_models_gpu([codec.patcher], memory_required=window * DECODE_FRAME_BYTES)
        device = codec.patcher.load_device
        with comfy.model_management.cuda_device_context(device):
            pieces = [codec.model.decode(c[None].to(device), torch.float32).float().cpu() for c in codes]
        silence = torch.zeros((1, round(TAKE_GAP * SAMPLE_RATE)))
        joined = torch.cat([part for piece in pieces for part in (silence, piece)][1:], -1)
        return trim_pauses(joined, SAMPLE_RATE, MAX_PAUSE)[0].numpy()


if __name__ == "__main__":
    serve(DramaBackend(), "Fish Audio Drama 3 worker")
