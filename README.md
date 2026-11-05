# Hebrew Voice Studio

A local text-to-speech server for Hebrew that runs several open models behind one API and
one web UI (English and Hebrew). Everything runs on your own machines: nothing is sent to
a cloud service.

- **Web UI** at `http://localhost:8010`: read text aloud, compare engines, edit single
  sentences, a library of everything you made, long documents chapter by chapter, voice
  cloning with a clip editor, and a pronunciation dictionary.
- **HTTP API**, including an **OpenAI-compatible** `/v1/audio/speech`, so tools that speak
  to OpenAI TTS (Home Assistant, Open WebUI, scripts) can use it by changing the base URL.

## Engines

| Engine | Runs on | Built-in voices | Clones a voice from a recording | Notes |
|---|---|---|---|---|
| **LightBlue** (BlueTTS v2) | CPU, in the server | adam, daniel, lily, noa | yes (rough: tone only) | Very fast; the default. Reads English inside Hebrew text. |
| **Blue 2.5** (BlueTTS v2.5) | CPU, in the server | Female (Hebrew), Christabel, Liberty, Decker, Jason | yes (voice likeness on par with LightBlue) | Newer model by LightBlue's author (more training, 7 languages in its mix); about half LightBlue's speed. |
| **Kokoro** Hebrew | CPU, in the server | shaul, plus heart, bella, emma, michael and their blends with Shaul | no | Fast and clear. Non-commercial license. |
| **Chatterbox** Multilingual | GPU, in the server | default | yes | Expression and voice-adherence controls. |
| **OmniVoice** | GPU, worker process (env `omnivoice`) | female, male | yes (good) | Started and stopped by the server. |
| **Higgs Audio v3** (Boson AI's Higgs TTS 3, 4B) | GPU, worker process (env `higgs`) | 6 generic presets | yes (good) | About 2.5x real time. 9 of 60 test sentences had a garbled word in testing. |
| **QwenTTS-he** (Qwen3-TTS 1.7B + Hebrew adapter) | GPU, worker process (env `qwentts`) | female, male | yes (good) | Clearest in testing; about 8x real time. |
| **Magpie-TTS Hebrew** (NVIDIA Magpie 357M + Hebrew fine-tune) | GPU, worker process (env `magpie`) | 15 Hebrew voices (female1-5, male1-5, voice1-3, female1-he, male1-he) | no | Very clear (98-99% of words heard in testing); about 1.4x real time, 22 kHz. |
| **F5-TTS Hebrew** (F5-TTS + Hebrew fine-tune) | GPU, worker process (env `f5`) | your recorded voices only | yes | Clear (no words added or dropped in 30 test sentences); about 2.5x real time, 24 kHz. Needs at least one recorded voice. |
| **VibeVoice 1.5B / 7B** (Microsoft) | GPU, worker process (env `omnivoice`) | alice, carter, frank, maya, samuel | yes (no transcript needed) | **English first.** Long-form, natural English; Hebrew with an English accent; about real time (1.5B). Starts unloaded: the first reading loads it (1.5B: about 7 GB of GPU memory, 7B: about 19 GB). |
| **Zonos-Hebrew** | GPU, Docker on another machine | default | yes | Optional, off by default. Hebrew only; has emotions. |

English words inside Hebrew text, and English-only text, are read by every engine except
F5-TTS and Zonos, which read Hebrew only (the app marks them "Hebrew only"). Kokoro and Magpie
read English with their Hebrew voices, so with a Hebrew accent. VibeVoice is the reverse
(marked "English first"): Hebrew comes out with an English accent and some words added or
dropped. It does best on the 7B model with a clean recording of a Hebrew speaker, and worse with
niqqud or romanized text. Some takes come out with sound behind the voice, so a noisy take is
read again, up to 3 times; turn this off in Advanced for readings about twice as fast.

Every engine is optional at runtime: one that fails to load is skipped, and the server
runs with the rest.

**Self-contained:** the code, models, voices and fonts all live inside this folder. Model
downloads go to `models\hf` (the server points `HF_HOME` and `TORCH_HOME` at folders under
`models\`), not to your user profile, so the folder can be copied or packed into a Docker
image as it is. The only things outside it are the conda environments themselves.

## Requirements

- Windows 10/11 (developed and tested on Windows 11). Linux should work for everything
  except the `.bat` files, but is untested.
- An NVIDIA GPU for Chatterbox, OmniVoice, Higgs, QwenTTS-he, Magpie and F5-TTS: about 23 GB
  with all of them loaded. Tested on an RTX 5090 (32 GB), which needs the CUDA 12.8 build of PyTorch.
  LightBlue, Blue 2.5 and Kokoro run on the CPU.
- [Miniconda](https://docs.anaconda.com/miniconda/) (or Anaconda) and Git.
- About 50 GB of disk for all models (most of it for Chatterbox, OmniVoice, Higgs,
  QwenTTS-he, Magpie and F5-TTS in `models\hf`).

## Install

All commands are for a normal Command Prompt in the project folder (`C:\dev\voice` here).

### 1. The main environment (`voice`)

```bat
conda create -y -n voice python=3.12
conda activate voice
pip install --no-deps -r requirements\voice.lock.txt
```

The lock file includes `bluecodec` (BlueTTS's audio codec, installed from GitHub as it is
not on PyPI), which Blue 2.5 needs to clone voices. `voice.lock.txt` holds the exact versions of the working setup. Install it with `--no-deps`:
Chatterbox asks for an older torch, but the RTX 50xx needs `torch 2.9.1+cu128`, so the lock
file deliberately overrides it. Don't run `pip install -U` on torch, setuptools (must stay
below 81), diffusers (0.29) or numpy (1.26) in this env.

### 2. Third-party code

```bat
python tools\fetch_vendor.py
```

This downloads BlueTTS's runtime into `vendor\` at the version this app was tested with
(it isn't part of this repository).

**Models download by themselves** into `models\` the first time you use an engine (Hugging
Face models into `models\hf`), so that first use takes a while. All engines need about 40 GB of
disk. Some models ask you to accept their terms on the Hugging Face website first;
then run `hf auth login`. To start only some engines (and download only their models), set
`VOICE_ENGINES`, for example `set VOICE_ENGINES=lightblue,kokoro,qwen,magpie`.

- If you already have models in another Hugging Face cache on this PC,
  `python tools\copy_hf_models.py [other cache folders]` copies them in instead.
- Higgs's built-in voice presets aren't downloadable; without them Higgs speaks with your
  own recorded voices only.
- The web UI's default fonts are in `web\fonts` (`python tools\fetch_fonts.py` downloads them again).

### 3. OmniVoice (optional)

```bat
conda create -y -n omnivoice python=3.12
conda activate omnivoice
pip install --no-deps -r requirements\omnivoice.lock.txt
```

The server finds the env by name and starts `workers\omnivoice_worker.py` in it. Without
this env OmniVoice is skipped. VibeVoice runs in the same env (`workers\vibevoice_worker.py`),
so this env also gives you VibeVoice 1.5B and 7B; their models (5.4 and 18.7 GB) download on
their first reading.

### 4. Higgs Audio v3 (optional)

```bat
conda create -y -n higgs python=3.12
conda activate higgs
pip install --no-deps -r requirements\higgs.lock.txt
```

The model code comes with the model from Hugging Face (the transformers port, at a pinned
revision) and the voice presets are in `models\higgs-voices`. The server starts
`workers\higgs_worker.py` in this env. Without the env Higgs is skipped.

### 5. QwenTTS-he (optional)

```bat
conda create -y -n qwentts python=3.12
conda activate qwentts
pip install --no-deps -r requirements\qwentts.lock.txt
```

`qwen-tts` needs `transformers 4.57.3`, hence its own env. The base model
(`Qwen/Qwen3-TTS-12Hz-1.7B-Base`) and the Hebrew adapter (`notmax123/QwenTTS-he-1.7B`)
download into `models\hf` on the first start. The worker converts the text to IPA with the
same RenikudPlus model as LightBlue (`models\renikud-plus`). Without the env QwenTTS-he is
skipped. The "SoX could not be found" message at its start is harmless (it's for a
tokenizer this model doesn't use).

### 6. Magpie-TTS Hebrew (optional)

```bat
conda create -y -n magpie python=3.12
conda activate magpie
pip install --no-deps -r requirements\magpie.lock.txt
```

The model runs on NVIDIA NeMo (`nemo_toolkit` 3.0.0 from PyPI). The Hebrew fine-tune
(`notmax123/magpie_tts_multilingual_357m-he`) and its audio codec
(`nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps`) download into `models\hf` on the first
start. Like QwenTTS-he, the text is converted to IPA with RenikudPlus. Magpie can't clone
voices (NVIDIA removed the reference-audio encoder from the release), so it has its fixed
voices only. Two NeMo extras don't install on Windows and aren't needed for Hebrew:
`nemo_text_processing` and `pyopenjtalk` (the worker stands in for the latter).

### 7. F5-TTS Hebrew (optional)

```bat
conda create -y -n f5 python=3.12
conda activate f5
pip install -r requirements\f5.lock.txt
```

The Hebrew fine-tune (`Yzamari/f5tts-hebrew-v2`) and its vocoder download into `models\hf`
on the first start. Text gets niqqud from Phonikud, as for Kokoro. F5 has no voices of its
own: it clones each voice in `voices\`. The first time it uses a voice it cuts the recording to
under 12 seconds and writes down its words with Whisper (cached next to the voice). Don't
install `torchcodec` in this env: it needs FFmpeg DLLs and breaks Whisper without them.

### 8. Zonos-Hebrew (optional, off by default)

Zonos runs in Docker on a machine with an NVIDIA GPU (an RTX 3090 here).
`docker\zonos-hebrew\docker-compose.yml` has the stack and instructions for Portainer.

Then turn it on in the web UI: **Settings → Zonos-Hebrew**, enter its address (for
example `http://<docker-host>:7860/`) and press **Connect**. It connects right away, no
restart needed, and the address is saved in `settings.json` for the next start. **Turn off**
drops it again.

Instead of the web UI you can also use the `ZONOS_URL` environment variable, or
`start.bat --zonos-url <address>` for one run. The order is: `--zonos-url`, then the address
saved in the web UI, then `ZONOS_URL`. Without any of them, Zonos is not used.

## Docker

The image holds the code and every engine's Python environment (about 9.4 GB to pull), and no
models: the app downloads them into a volume the first time each engine is used.

```bash
docker run -d --gpus all -p 8010:8010 -p 10200:10200   -v /srv/voice/models:/app/models   -v /srv/voice/data:/data   ghcr.io/tm1981/hebrew-voice-studio:latest
```

To build the image yourself: `docker build -t hebrew-voice:dev .` in this folder.

- `/app/models` holds the models (about 40 GB with every engine). Map it to a host folder to
  keep downloads across rebuilds, or to reuse a models folder you already have.
- `/data` holds your data: recorded voices, the library, presets, settings, pronunciations and
  `api_key.txt`.
- `VOICE_ENGINES=lightblue,kokoro,qwen,magpie` starts only those engines (and downloads only
  their models). GPU engines start unloaded and load when you choose one of their voices, so
  only what you use takes GPU memory; unused ones unload by themselves after 30 minutes.
- `ZONOS_URL=http://<host>:7860/` connects Zonos-Hebrew (or set it in the web UI).
- Port 10200 is the Wyoming port for [Home Assistant](#home-assistant).
- In Docker, clients that reach the app through Docker's own gateway address (for example
  from the Docker host itself) always need the API key, since that address can stand for
  anyone. The key is in the log at startup and under **Settings → API key**, where it can
  be changed.
- For Portainer or `docker compose`, use [`docker/voice/docker-compose.yml`](docker/voice/docker-compose.yml).
- The host needs the NVIDIA driver and the NVIDIA Container Toolkit. The image is built for
  CUDA 12.8 and runs on RTX 30xx, 40xx and 50xx cards.
- Higgs's built-in voice presets aren't downloadable; copy `*.wav` + `*.lab` presets into
  `models/higgs-voices` if you have them, or use your own recorded voices.

### Docker in WSL on Windows

WSL 2 puts its containers behind a private network address, so `http://localhost:<port>`
works on the Windows PC itself but other devices on the home network can't reach the app.
Two ways to open it to them:

- **Mirrored networking (Windows 11 22H2 and later):** put this in
  `C:\Users\<you>\.wslconfig`, run `wsl --shutdown`, and start the container again:

  ```ini
  [wsl2]
  networkingMode=mirrored
  ```

  WSL then shares the PC's addresses, and the app is at `http://<this PC's IP>:8010` (the
  port on the left of `-p`). Windows Firewall must allow that port: in an administrator
  PowerShell, `New-NetFirewallRule -DisplayName "Hebrew Voice Studio" -Direction Inbound
  -Protocol TCP -LocalPort 8010,10200 -Action Allow`.
- **A port proxy (older Windows):** in an administrator PowerShell, with the address that
  `wsl hostname -I` prints first (it changes when WSL restarts):
  `netsh interface portproxy add v4tov4 listenport=8010 listenaddress=0.0.0.0 connectport=8010 connectaddress=<WSL address>`,
  plus the same firewall rule.

Requests forwarded this way can arrive from WSL's or Docker's gateway address rather than
the device's own, and then need the API key (see above).

## Run

Double-click **`start.bat`**, or from a Command Prompt:

```bat
start.bat
```

Only the CPU engines load at startup, so it starts quickly; each GPU engine loads when you
choose one of its voices. When the console shows
`Uvicorn running on http://0.0.0.0:8010`, open:

- **On this PC:** http://localhost:8010
- **From your phone or another computer on the home network:** `http://<this PC's IP>:8010`

Close the window (or press Ctrl+C) to stop the server; the worker processes stop with it.

Options (they pass through `start.bat`):

| Option | What it does |
|---|---|
| `--port 8011` | Use another port (default 8010; port 8000 is taken by Windows on this PC). Worker ports follow it (+21, +22). |
| `--zonos-url URL` | Use Zonos-Hebrew at that address for this run (normally set in the web UI instead). |
| `--no-auth` | Accept requests from anywhere without the API key. |
| `--lan-needs-key` | Also require the key from the home network. |
| `--api-key KEY` | Use this key instead of the one in `api_key.txt`. |

Environment: `TTS_DEVICE=cuda:1` puts Chatterbox on another GPU.

**Start with Windows:** run `install_autostart.bat` once (it adds a shortcut to your Startup
folder; the server starts minimized when you sign in). `remove_autostart.bat` undoes it.

### Access and the API key

- This PC and the home network (192.168.x.x, 10.x.x.x, 172.16-31.x.x) need nothing.
- Anything else needs the key from `api_key.txt` (created on the first start and printed in
  the console), sent as `Authorization: Bearer <key>`, `X-API-Key: <key>` or `?api_key=<key>`.
  The web UI asks for it once and remembers it.
- The microphone recorder in the web UI only works on `http://localhost:8010` (browsers
  block the microphone on plain-HTTP network addresses). From other devices, upload a file.

## Using the web UI

- **Read:** type or paste Hebrew, pick a voice, press **Read aloud** (Ctrl+Enter). Sentences
  play as soon as each one is ready. **With niqqud** shows how the engine will read the text.
  **Compare all engines** reads the text with one voice from each engine.
- **Text tricks:** `[הפסקה 2]` or `[pause 2]` adds a 2-second pause. Turn on **Dialogue** and
  write lines like `נועה: שלום` to give each speaker its own voice. Numbers, dates, prices and
  common abbreviations are read as words, in English when the words around a number are
  English ("I am 10 years old", "on 12/03/2025"). Niqqud you type is respected word by word.
- **Sentences** (button on the player): replay, re-read or edit one sentence; the full file is
  rebuilt.
- **Library:** every reading, kept across restarts, with search and download (WAV, MP3, FLAC, Opus).
- **Long documents:** a TXT, DOCX or PDF is read chapter by chapter in the background, with a
  ZIP of all chapters at the end.
- **Voices:** all voices by engine. **Add a voice** takes up to 5 recordings of one speaker
  (upload or record); each can be played, trimmed by dragging, cut to the speech, and leveled
  before saving. **Check quality** flags noise, clipping and length and shows what Whisper
  hears in each recording. The new voice appears in every engine that can clone, from
  recordings leveled to the same volume. On your own voices, **Edit** loads the saved
  recordings back into the form to add, trim or remove some and save again, and the download
  button saves them as a ZIP.
- **Pronunciation:** words an engine says wrong, with how to read them; applies to every engine.
- **Transcribe:** upload an audio or video file, or record from the microphone, and get the
  text back, with Copy and Download .txt. It runs on the built-in speech-to-text or on a
  Whisper server elsewhere on the network (see [Speech-to-text](#speech-to-text)); any
  language, automatic by default. The browser decodes the audio, so m4a, webm and mp4 work too.
- **Settings:** which engines run, and GPU memory. GPU engines (Chatterbox, OmniVoice, Higgs,
  QwenTTS-he, Magpie, F5-TTS, VibeVoice) start unloaded, and choosing a voice loads its engine.
  **Unload** frees one engine's memory, **Free GPU memory** frees all of them, and engines
  unused for a while (30 minutes by default; can be changed or turned off) unload by
  themselves. An unloaded engine stays in the voice list; API requests load it too. Loading
  takes 10 to 60 seconds (VibeVoice 7B about 3 minutes). An engine that is reading isn't unloaded.
- **Advanced** (under the Read settings): the pause between sentences, and the selected
  engine's own settings, such as quality steps, text adherence, variation, a seed (the same
  number gives the same reading again), Kokoro's blend with Shaul, OmniVoice's voice design
  (age, pitch, whisper, accent) and Zonos's emotion and pitch variation. Each engine keeps
  its own values; the defaults are how the engines read without them. Settings are saved
  with each reading, so re-reading a sentence uses the same ones.
- **Presets** keep a set of settings under a name; **Aa** changes the fonts and text size;
  **EN | עב** switches the interface language.

## API examples

```bash
# Hebrew text to MP3 (put Hebrew in a file: Windows curl mangles Hebrew arguments)
curl -X POST http://localhost:8010/tts -H "Content-Type: application/json" --data-binary @request.json -o out.mp3
#   request.json: {"text": "שלום עולם", "voice": "lightblue/noa", "speed": 1.0, "format": "mp3"}

# OpenAI-compatible
curl http://localhost:8010/v1/audio/speech -H "Content-Type: application/json" --data-binary @openai.json -o out.mp3
#   openai.json: {"model": "tts-1", "input": "שלום עולם", "voice": "nova", "response_format": "mp3"}

# All voices
curl http://localhost:8010/voices

# Each engine's settings (key, type, default, range)
curl http://localhost:8010/options
#   used in a request as {"options": {"magpie": {"temperature": 0.5, "seed": 7}}, "sentence_pause": 0.4, ...}
```

Voices are named `<engine>/<voice>`, for example `lightblue/noa`, `kokoro/emma_shaul` or
`omnivoice/<your voice>`. OpenAI voice names (alloy, nova, shimmer, onyx...) map to voices
here; which voice each one plays can be changed under **Settings → Home Assistant and
other apps**. With `"response_format": "pcm"` or `"wav"` the OpenAI endpoint streams the audio
sentence by sentence; the other formats come as one finished file. The full list of
endpoints is at the top of `server.py`, and interactive docs are at http://localhost:8010/docs.

## Home Assistant

The server also speaks the **Wyoming protocol** (port 10200), which Home Assistant uses for
local voices and speech-to-text:

1. In Home Assistant: **Settings → Devices & services → Add integration → Wyoming Protocol**.
2. Host: the address of the machine running this app; port: `10200`.
3. Every voice appears in Home Assistant (in the TTS options of automations and of voice
   assistants). Without a choice there, the **default voice** set under **Settings** is
   used.

Audio goes back sentence by sentence, and a voice assistant's reply is read while it's still
being written. For quick announcements:

- Tick **Keep loaded** for the GPU engine you use (Settings): it loads when the server
  starts and isn't unloaded for being idle, so announcements don't wait 10-60 seconds for a
  model. The CPU engines (LightBlue, Blue 2.5, Kokoro) are always ready.
- Wyoming has no API key, so it lets in only the clients the HTTP API lets in without one:
  this machine and the home network (neither with `--lan-needs-key`, and never Docker's
  gateway address). To allow exactly some addresses instead, set e.g.
  `WYOMING_ALLOW=192.168.1.20` (Home Assistant's address; networks like `10.0.0.0/24` work
  too). Don't forward port 10200 from the internet. `--wyoming-port 0` turns it off.

The OpenAI-compatible endpoint works too (e.g. the OpenAI TTS custom integration): URL
`http://<host>:8010/v1/audio/speech`, your API key, any voice name from this app.

### Speech-to-text

The same Wyoming entry also offers **Hebrew speech-to-text**, so the voice assistant can
understand spoken Hebrew without the cloud. In **Settings → Voice assistants**, pick a
Hebrew assistant and choose this app as both speech-to-text and text-to-speech.

- The model is [ivrit-ai/whisper-large-v3-turbo](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo),
  Whisper fine-tuned on Hebrew (Apache-2.0). It downloads (about 1.6 GB) on first use and takes
  about 2 GB of GPU memory.
- It shows up in **Settings** with the engines: tick **Keep loaded** so the first command
  doesn't wait a few seconds for it to load. Once loaded, a short command takes about 0.1 s on
  an RTX 5090.
- **Settings → Speech-to-text → Built-in model** switches to OpenAI's
  [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (MIT, same
  size) for other languages, for Home Assistant too. `--stt-model` picks any Whisper model from
  Hugging Face instead (then it can't be switched in Settings); `STT_MODEL` sets the model used
  until one is picked in Settings; `off` in either turns speech-to-text off.
- **Settings → Speech-to-text → Homelab Whisper server:** the address of a Wyoming
  speech-to-text server, such as the `lscr.io/linuxserver/faster-whisper` container Home
  Assistant uses (port 10300), for the Transcribe page. **Test connection** shows the model it
  runs, which is set in its own compose file (`WHISPER_MODEL`). An ivrit-ai model there hears
  every recording as Hebrew; it writes English correctly with Hebrew or automatic as the
  language, but forced to English it writes English speech in Hebrew.
- **Words for speech-to-text** (Settings → Home Assistant): names of rooms, devices and
  people, spelled the way Home Assistant knows them (`סלון, נאנוליף, יואב`). Whisper then
  writes them that way (נאנוליף, not ננוליף), so the command matches.
- Recordings without speech (a false wake word, room noise) come back empty instead of as
  Whisper's made-up "תודה רבה": each part is checked with [Silero VAD](https://github.com/snakers4/silero-vad)
  first.
- OpenAI-compatible too: `POST /v1/audio/transcriptions` with a `file` (wav, mp3, flac, ogg; up to 15 minutes),
  optional `language` (default `he`), `prompt` (hint words; the saved ones if empty) and
  `response_format` (`json` or `text`).

## The API key

**Settings → API key** shows the key, copies it, sets your own or generates a new one
(saved to `api_key.txt` in the data folder, and used at once). A key passed with `--api-key` or
`TTS_API_KEY` is shown but can only be changed there.

The home network needs no key only when the server is opened by an IP address, `localhost`, a
one-word machine name or a `.local`/`.lan`/`.home`/`.internal` name, and, for requests from a
web page, only from the app's own page. This stops other websites you visit from using the
server through your browser. To skip the key under another name, set
`TTS_ALLOWED_HOSTS=voice.example.com` (comma-separated).

## Backup and restore

**Settings → Backup** downloads one zip with your recorded voices, presets, settings,
pronunciations and the API key, and optionally the whole library of readings. **Restore** adds a
backup to what's there: voices and readings with the same name are replaced, presets,
pronunciations and settings are merged, and everything applies without a restart. This is the
easy way to move to another machine or rebuild the Docker container, where the data otherwise
sits inside a volume. The backup holds the API key, so keep it private. From scripts:
`GET /backup?library=true` and `POST /restore` (a file upload named `file`).

## Project layout

| Path | What's there |
|---|---|
| `server.py` | The API and web server; start here. |
| `engines.py` | One class per engine behind a common interface, and the worker launcher. |
| `workers/` | OmniVoice, Higgs, QwenTTS-he, Magpie and F5-TTS servers, each run in its own conda env. |
| `text_prep.py`, `niqqud.py`, `script.py` | Hebrew normalization, niqqud handling, pauses and dialogue. |
| `library.py`, `documents.py` | Saved readings and long-document jobs. |
| `backup.py`, `wyoming_service.py` | Backup/restore of your data; the Wyoming server for Home Assistant. |
| `chatterbox_fixes.py` | Fixes that make Chatterbox stable on Hebrew. |
| `web/index.html` | The whole web UI (one file). |
| `voices/` | Your cloned voices (one file or folder per voice); `voices/.cache` is rebuilt as needed. These and the next two rows are in `DATA_DIR` (this folder unless set; `/data` in Docker). |
| `outputs/library/` | Saved readings. |
| `pronunciations.json`, `presets.json`, `settings.json` | Your dictionary, presets and web UI settings. |
| `paths.py`, `model_files.py` | Where data and models live; models downloaded on first use. |
| `models/`, `vendor/` | Downloaded models and third-party code; `models/hf` is the Hugging Face cache. |
| `requirements/` | Exact package versions for each env (`*.lock.txt` for Windows, `linux/` for Docker). |
| `tools/` | Helpers: fetch the third-party code, copy existing Hugging Face models in, download the UI fonts. |
| `Dockerfile`, `docker/voice/` | The Docker image and its Compose file. |
| `docker/zonos-hebrew/` | Docker Compose file for Zonos-Hebrew. |

## Troubleshooting

- **`UnicodeDecodeError` / cp1255 errors:** the server re-runs itself in Python's UTF-8 mode;
  if you start it some other way, add `-X utf8` (`python -X utf8 server.py`).
- **Port already in use:** use `--port`. On this PC, 8000 belongs to a Windows service.
- **An engine is missing from the voice list:** look for `[WARN]` lines in the console; the
  **Settings** screen shows which engines are running.
- **Out of GPU memory:** another program, or a second copy of this server, is using the GPU.
  Use **Free GPU memory** on the Settings screen, or a shorter auto-unload time.
- **`pkg_resources` not found:** `pip install "setuptools<81"` in the `voice` env.
- **CUDA errors on an RTX 50xx:** torch must be the `cu128` build (see Install, step 1).
- **The web UI keeps asking for an API key:** you're coming from outside the home network;
  paste the key from `api_key.txt`.

## Models and licenses

The code in this folder is yours. Each model has its own license, and several are
**non-commercial**: check them before any commercial use. Cloning a voice needs the
consent of the person whose voice it is.

### Speech engines

| Engine | Model | License | Links |
|---|---|---|---|
| LightBlue | BlueTTS v2 ONNX (`notmax123/blue-onnx-v2`), by Max Melichov | MIT | [model](https://huggingface.co/notmax123/blue-onnx-v2) · [code](https://github.com/maxmelichov/BlueTTS) |
| Blue 2.5 | BlueTTS v2.5 ONNX (`notmax123/BlueTTS2.5-onnx`), same author | MIT. Its "Female (Hebrew)" voice is an in-house recording, and the author asks to check its rights before shipping it. | [model](https://huggingface.co/notmax123/BlueTTS2.5-onnx) · [code](https://github.com/maxmelichov/BlueTTS) |
| Kokoro | Kokoro-82M fine-tuned on SASPEECH (`thewh1teagle/kokoro-hebrew-nc`, from `avris/kokoro-hebrew-saspeech`) | **Non-commercial** (SASPEECH dataset terms) | [model](https://huggingface.co/thewh1teagle/kokoro-hebrew-nc) · [original](https://huggingface.co/avris/kokoro-hebrew-saspeech) · [dataset](https://www.openslr.org/134/) |
| Kokoro extra voices | Kokoro-82M English voice styles (`voices-v1.0.bin`) | Apache-2.0 | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) · [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) |
| Chatterbox | Chatterbox Multilingual (`ResembleAI/chatterbox`), by Resemble AI | MIT | [model](https://huggingface.co/ResembleAI/chatterbox) · [code](https://github.com/resemble-ai/chatterbox) |
| OmniVoice | OmniVoice (`k2-fsa/OmniVoice`), by k2-fsa | Apache-2.0 (its audio tokenizer has its own license; see the model page) | [model](https://huggingface.co/k2-fsa/OmniVoice) · [code](https://github.com/k2-fsa/OmniVoice) |
| QwenTTS-he | Hebrew LoRA adapter `notmax123/QwenTTS-he-1.7B` (by Max Melichov) on Alibaba's `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Apache-2.0 (adapter and base model) | [adapter](https://huggingface.co/notmax123/QwenTTS-he-1.7B) · [base model](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) · [code](https://github.com/QwenLM/Qwen3-TTS) |
| Magpie-TTS Hebrew | Hebrew fine-tune `notmax123/magpie_tts_multilingual_357m-he` (by Max Melichov) of NVIDIA's `nvidia/magpie_tts_multilingual_357m`, with the codec `nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps` | NVIDIA Open Model License (commercial use allowed under its terms) | [model](https://huggingface.co/notmax123/magpie_tts_multilingual_357m-he) · [base model](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) · [code](https://github.com/maxmelichov/magpie_tts_multilingual_357m-he) |
| Higgs | Higgs TTS 3 (4B) by Boson AI (`bosonai/higgs-tts-3-4b`), run through the plain-transformers port `multimodalart/higgs-audio-v3-tts-4b-transformers` (the same weights file, plus modeling code), with the codec `bosonai/higgs-audio-v2-tokenizer` | **Research and non-commercial** (Boson Higgs TTS 3 license; its Creator Use Grant allows monetized creator content if it credits Boson AI's Higgs Audio) | [model](https://huggingface.co/bosonai/higgs-tts-3-4b) · [port](https://huggingface.co/multimodalart/higgs-audio-v3-tts-4b-transformers) · [codec](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer) |
| VibeVoice | `vibevoice/VibeVoice-1.5B-hf` and `vibevoice/VibeVoice-7B-hf` (Microsoft's VibeVoice in Transformers format); sample voices from `bezzam/vibevoice_samples` | MIT | [1.5B](https://huggingface.co/vibevoice/VibeVoice-1.5B-hf) · [7B](https://huggingface.co/vibevoice/VibeVoice-7B-hf) · [code](https://github.com/microsoft/VibeVoice) |
| F5-TTS Hebrew | Hebrew fine-tune `Yzamari/f5tts-hebrew-v2` of F5-TTS v1 Base, with the Vocos vocoder `charactr/vocos-mel-24khz` | Apache 2.0 (the F5-TTS v1 Base weights it starts from are CC BY-NC 4.0, so commercial use is unclear) | [model](https://huggingface.co/Yzamari/f5tts-hebrew-v2) · [code](https://github.com/SWivid/F5-TTS) |
| Zonos (optional) | Zonos-Hebrew (`notmax123/Zonos-Hebrew`), built on Zyphra's Zonos | **CC BY-NC 4.0** (non-commercial) | [model](https://huggingface.co/notmax123/Zonos-Hebrew) · [code](https://github.com/maxmelichov/Zonos-Hebrew) |
| Speech-to-text | Hebrew fine-tune `ivrit-ai/whisper-large-v3-turbo` of OpenAI's Whisper large-v3-turbo | Apache 2.0 | [model](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo) |
| Speech detection | Silero VAD | MIT | [code](https://github.com/snakers4/silero-vad) |

### Hebrew text and helper models

| Used for | Model | License | Links |
|---|---|---|---|
| Niqqud for Chatterbox | DictaBERT char menaked, ONNX export (`dicta-1.0.int8.onnx`) | CC BY 4.0 (Dicta's model) | [ONNX](https://github.com/thewh1teagle/dicta-onnx) · [original](https://huggingface.co/dicta-il/dictabert-large-char-menaked) |
| Niqqud and phonemes for Kokoro | Phonikud (`phonikud-1.0.int8.onnx`) | No license stated; see the project | [model](https://huggingface.co/thewh1teagle/phonikud-onnx) · [code](https://github.com/thewh1teagle/phonikud) |
| Turning a recording into a Blue 2.5 voice | BlueCodec (`notmax123/blue-codec`, `model.safetensors`), same author | MIT | [model](https://huggingface.co/notmax123/blue-codec) · [code](https://github.com/maxmelichov/blue-codec) |
| Hebrew phonemes for LightBlue and Blue 2.5 | RenikudPlus (`notmax123/RenikudPlus`) | Apache-2.0 | [model](https://huggingface.co/notmax123/RenikudPlus) |
| Transcribing cloned-voice recordings for OmniVoice | Whisper large-v3 turbo (`openai/whisper-large-v3-turbo`) | MIT | [model](https://huggingface.co/openai/whisper-large-v3-turbo) |
| Chatterbox's sample voice (`female1.wav`) | phonikud-chatterbox | No license stated; see the project | [code](https://github.com/thewh1teagle/phonikud-chatterbox) |

The Blue 2.5 voices Christabel, Liberty, Decker and Jason come from LibriTTS-R readers
(CC BY 4.0, Google LLC). The interface fonts (Heebo, Noto Serif Hebrew and the others in
the font menu) are under the SIL Open Font License.
