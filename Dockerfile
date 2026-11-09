# syntax=docker/dockerfile:1
#
# Hebrew Voice Studio: the server (env 'voice') plus the eight GPU worker envs, for a Docker host
# with an NVIDIA GPU (built for an RTX 3090 / sm_86; the cu128 torch wheels also cover RTX 50xx).
#
#   docker build -t hebrew-voice:dev .
#   docker run --gpus all -p 8010:8010 -v voice-models:/app/models -v voice-data:/data hebrew-voice:dev
#
# Models are not in the image: the app downloads them into /app/models (a volume) on first use.
# User data (voices, library, settings) lives in /data (a volume).
#
# Base image: plain ubuntu:24.04, not nvidia/cuda. The Linux torch wheels bring their own CUDA
# runtime, cuDNN, cuBLAS, NCCL... as nvidia-* pip wheels (listed in requirements/linux/*.txt),
# and the NVIDIA container runtime mounts the driver (libcuda) into the container. A CUDA base
# image would only add a second copy of the CUDA libraries.
#
# The nine Python envs are uv venvs in /opt/envs/<env>, all on one uv-managed CPython 3.12.16
# (/opt/python). They are installed in ONE layer with the uv cache inside that layer and
# --link-mode hardlink, so a wheel shared by several envs (torch 2.9.1+cu128 and its nvidia-*
# wheels are in seven of them; Drama 3 alone needs torch's cu132 build) is stored once in the
# image. The cache is deleted at the end of the layer; the hardlinked files stay.

FROM ubuntu:24.04

LABEL org.opencontainers.image.source="https://github.com/tm1981/hebrew-voice-studio" \
      org.opencontainers.image.description="Local multi-engine Hebrew text-to-speech (models download on first use)" \
      org.opencontainers.image.licenses="MIT"

ARG DEBIAN_FRONTEND=noninteractive
# ffmpeg: pydub, librosa/audioread, F5's preprocessing. libsndfile1: soundfile (its wheel bundles
# one too). espeak-ng: LightBlue's English goes through phonemizer, which loads libespeak-ng.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates ffmpeg libsndfile1 espeak-ng tzdata \
 && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_LINK_MODE=hardlink \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_HTTP_TIMEOUT=300 \
    UV_NO_PROGRESS=1

COPY requirements/linux/ /tmp/req/

# uv is mounted from its image for this step only, so it isn't in the final image.
# --index-strategy unsafe-best-match: the requirement files add the PyTorch index; without it uv
# would take every package that index mirrors (numpy, pillow...) from it only, at whatever
# versions it happens to carry.
RUN --mount=from=ghcr.io/astral-sh/uv:0.13.6,source=/uv,target=/usr/local/bin/uv \
    set -eux; \
    uv python install 3.12.16; \
    for env in voice omnivoice higgs qwentts magpie f5 ming drama breeze; do \
        uv venv --python 3.12.16 "/opt/envs/$env"; \
        uv pip install --python "/opt/envs/$env/bin/python" --no-deps -r "/tmp/req/$env.txt"; \
    done; \
    rm -rf /tmp/uv-cache /tmp/req /root/.cache

# A C compiler for torch.compile (Higgs, Breeze Fast) and Drama 3: Inductor and Triton build small
# kernels and launchers at run time. Higgs falls back to slower eager mode without one, Drama 3
# fails. Its own layer after the envs, so changing it doesn't reinstall them.
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*

# Chatterbox's Chinese word splitter downloads its 34 MB model from GitHub the first time it loads,
# which was slow enough (minutes) to look like Chatterbox hanging on every new container.
# (The zip stays: it's what the splitter checks for before downloading again.)
RUN /opt/envs/voice/bin/python -c "from spacy_pkuseg import pkuseg; pkuseg()"

WORKDIR /app
COPY . /app

# Third-party code at pinned versions.
RUN /opt/envs/voice/bin/python tools/fetch_vendor.py \
 && find /app -name __pycache__ -prune -exec rm -rf {} +

ENV VOICE_ENVS=/opt/envs \
    DATA_DIR=/data \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Compiled-kernel caches go to the data volume so they survive image updates. Without this, the
# first reading after every update compiles everything again: torch.compile (Higgs, Breeze Fast)
# through Inductor and Triton, and the NVIDIA driver's kernel cache (about 1 GB once all engines
# have run). Entries made by other library or driver versions are just ignored, not misused.
# The backup zip only takes voices and readings, so these never end up in it.
ENV CUDA_CACHE_PATH=/data/cache/nv \
    TORCHINDUCTOR_CACHE_DIR=/data/cache/inductor \
    TRITON_CACHE_DIR=/data/cache/triton \
    PYTORCH_KERNEL_CACHE_PATH=/data/cache/torch-kernels

VOLUME ["/app/models", "/data"]
EXPOSE 8010 10200

ENTRYPOINT ["/opt/envs/voice/bin/python", "-X", "utf8", "server.py", "--host", "0.0.0.0"]
