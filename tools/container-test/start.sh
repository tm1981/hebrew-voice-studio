#!/bin/bash
# Starts (or replaces) the test container "voice-wsl" on localhost:8092 from a local image.
#
#   wsl bash tools/container-test/start.sh [image]        (image: hebrew-voice:dev by default)
#
# Models come from this checkout's models/ folder, so nothing is downloaded twice; the data
# (voices, library, API key) is in the Docker volume voice-wsl-data, apart from the app's own.
# Drama 3's weights are copied once into a Docker volume: read from a Windows folder mounted into
# WSL, they crash the engine (see the README).
set -e
repo=$(cd "$(dirname "$0")/../.." && pwd)
image=${1:-hebrew-voice:dev}
drama=()
if [ -d "$repo/models/drama3" ]; then
  docker volume create voice-drama-test >/dev/null
  docker run --rm -v voice-drama-test:/to -v "$repo/models/drama3":/from:ro ubuntu:24.04 sh -c 'cp -n /from/* /to/ 2>/dev/null; true'
  drama=(-v voice-drama-test:/app/models/drama3)
fi
docker rm -f voice-wsl >/dev/null 2>&1 || true
docker run -d --name voice-wsl --gpus all --restart no -p 8092:8010 -p 10292:10200 \
  -v "$repo/models":/app/models "${drama[@]}" -v voice-wsl-data:/data "$image" >/dev/null
started=$(date +%s)
for _ in $(seq 1 300); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8092/health)" != 000 ] && break
  sleep 2
done
echo "up after $(( $(date +%s) - started ))s: http://localhost:8092"
