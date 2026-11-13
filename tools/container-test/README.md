# Checking a Docker image before it is published

These scripts run a locally built image as a test container and put it through the things that
have broken before. They are for whoever builds the image; the app doesn't use them.

On Windows the container runs in WSL's Docker; the commands below are typed in the project folder.

```
wsl docker build -t hebrew-voice:dev /mnt/c/dev/voice
wsl bash tools/container-test/start.sh
```

`start.sh` starts the container `voice-wsl` on <http://localhost:8092>. It reads the models from
this checkout's `models/` folder and keeps its own data (voices, library, API key) in the Docker
volume `voice-wsl-data`, apart from the app's.

Then, with the Python of the server's env (it needs `requests` and `soundfile`):

| Script | What it checks |
|---|---|
| `check_readings.py` | Twelve short readings that each broke something once (units, list numbers, a one-word sentence, Hebrew and Arabic digits, a cloned voice). |
| `check_engines.py [engine ...]` | Three Hebrew sentences per engine: the load time, the GPU memory it takes, and that unloading gives it back. Without names: every GPU engine. |
| `check_documents.py` | A long-document job to its ZIP, a pin, a download; deletes what it made. |
| `check_speed.py [engine ...]` | That speed 1.1 and 0.9 make a reading 10% shorter and longer, for engines whose pace the server changes. |

Each exits with 1 if something failed. The audio they make is left in the system temp folder
(`voice-container-test`) for listening.

The unit tests run inside the container:

```
wsl docker exec voice-wsl sh -c "cd /app && /opt/envs/voice/bin/python -X utf8 -m unittest discover tests"
```

WSL shuts down when nothing is using it, and the container stops with it: keep a WSL terminal
open (for example `wsl docker logs -f --tail 100 voice-wsl`) while testing.
