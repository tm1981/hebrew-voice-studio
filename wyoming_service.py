"""Wyoming protocol server, so Home Assistant's Wyoming integration can use this app as a local
text-to-speech and speech-to-text service: Settings > Devices & services > Add integration >
Wyoming Protocol, host = this machine, port = 10200 (--wyoming-port). One entry gives both.

Speech-to-text (Hebrew Whisper, stt.py): Home Assistant sends transcribe, then the recording
(audio-start / -chunk / -stop) and gets the transcript back.

The same protocol as a client (remote_models, remote_transcribe) reaches a speech-to-text server
elsewhere, such as faster-whisper on a home server, for the Transcribe page.

Home Assistant gets every voice in the app (as "<engine>/<name>") and reads with any of them.
It can send the text all at once (synthesize) or as it's being written by a voice assistant
(synthesize-start / -chunk / -stop); either way audio goes back sentence by sentence, so
playback starts after the first sentence.

Wyoming has no keys or passwords, so it lets in only the clients the HTTP API lets in without a
key: this machine and the home network (neither with --lan-needs-key; never Docker's gateway
address, which can stand for any client). WYOMING_ALLOW="192.168.1.20,10.0.0.0/24" lets in
exactly those addresses instead. Don't forward the port from the internet.
"""
import asyncio
import ipaddress
import re
import threading
from typing import Callable, Iterator, Optional

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncTcpServer
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped

from stt import BIDI_MARKS

CHUNK_SAMPLES = 2048  # audio goes out in chunks of this many samples
STREAM_GAP = 0.25  # seconds of silence between parts of a streamed text, as between sentences
# Where a stream of text may be cut and read so far: after . ! ? : ; or a newline, at a space.
SENTENCE_END = re.compile(r"[.!?:;\n]+[\"'״)\]]*\s+")
ATTRIBUTION = Attribution(name="Hebrew Voice Studio", url="https://github.com/tm1981/hebrew-voice-studio")

# speak(text, voice_id) -> (sample_rate, pieces): audio as float32 arrays at that rate, the
# next piece computed when asked for. Raises ValueError with a message for the client.
Speak = Callable[[str, str], tuple[int, Iterator[np.ndarray]]]
# transcribe(audio, language) -> text: mono float32 audio at 16 kHz.
TranscribeFn = Callable[[np.ndarray, Optional[str]], str]
STT_RATE = 16000
MAX_RECORDING_SECONDS = 120  # longer recordings are cut there (a voice command is a few seconds)
MAX_CHUNK_BYTES = 1 << 20  # one audio chunk; Home Assistant sends a few KB at a time


def client_policy(trusted: Callable[[str], bool], allow: str) -> Callable[[str], bool]:
    """Who may connect: the addresses and networks in allow ("192.168.1.20,10.0.0.0/24") if
    given, else trusted (the HTTP API's rule for clients that need no key). Fails closed."""
    networks = [ipaddress.ip_network(n.strip(), strict=False) for n in allow.split(",") if n.strip()]

    def allowed(host: str) -> bool:
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return False
        if ip.version == 6 and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if networks:
            return any(ip in n for n in networks)
        return trusted(str(ip))

    return allowed


class Handler(AsyncEventHandler):
    def __init__(self, reader, writer, voices: Callable[[], list[dict]], speak: Speak, default_voice: Callable[[], str],
                 allowed: Callable[[str], bool], transcribe: TranscribeFn | None = None,
                 stt_model: Callable[[], tuple[str, list[str]]] | None = None):
        super().__init__(reader, writer)
        self.voices, self.speak, self.default_voice = voices, speak, default_voice
        self.transcribe, self.stt_model = transcribe, stt_model
        self.stream: dict | None = None  # {"voice", "text", "spoken"} while text arrives in chunks
        self.language: str | None = None  # from transcribe, for the recording that follows
        self.recording: list[bytes] | None = None  # 16 kHz 16-bit mono, between audio-start and -stop
        self.converter = AudioChunkConverter(rate=STT_RATE, width=2, channels=1)
        peer = writer.get_extra_info("peername")
        self.allowed = bool(peer) and allowed(peer[0])
        if not self.allowed:
            print(f"[wyoming] refused {peer[0] if peer else 'unknown client'} (see WYOMING_ALLOW)")

    async def handle_event(self, event: Event) -> bool:
        if not self.allowed:
            await self.write_event(Error(text="This address may not connect (see WYOMING_ALLOW).").event())
            return False
        if Describe.is_type(event.type):
            await self.write_event(self.info().event())
        elif SynthesizeStart.is_type(event.type):
            self.stream = {"voice": self.voice_id(SynthesizeStart.from_event(event).voice), "text": "", "spoken": False}
        elif SynthesizeChunk.is_type(event.type) and self.stream is not None:
            self.stream["text"] += SynthesizeChunk.from_event(event).text
            ready, self.stream["text"] = split_ready(self.stream["text"])
            if ready:
                await self.read_stream_part(ready)
        elif SynthesizeStop.is_type(event.type) and self.stream is not None:
            if self.stream["text"].strip():
                await self.read_stream_part(self.stream["text"])
            self.stream = None
            await self.write_event(SynthesizeStopped().event())
        elif Transcribe.is_type(event.type):
            self.language = Transcribe.from_event(event).language
        elif AudioStart.is_type(event.type) and self.transcribe:
            self.recording = []
        elif AudioChunk.is_type(event.type) and self.recording is not None:
            self.add_recording_chunk(AudioChunk.from_event(event))
        elif AudioStop.is_type(event.type) and self.recording is not None:
            pcm, self.recording = b"".join(self.recording), None
            await self.send_transcript(pcm)
        elif Synthesize.is_type(event.type) and self.stream is None:
            # While streaming, clients also send the whole text this way for older servers; skip it.
            synth = Synthesize.from_event(event)
            await self.read_aloud(synth.text, self.voice_id(synth.voice))
        return True

    def info(self) -> Info:
        voices = [
            TtsVoice(name=v["id"], description=f"{v['name']} ({v['engine_label']})", attribution=ATTRIBUTION,
                     installed=True, version=None, languages=["he"] if v["hebrew_only"] else ["he", "en"])
            for v in self.voices()
        ]
        program = TtsProgram(name="hebrew-voice-studio", description="Hebrew Voice Studio", attribution=ATTRIBUTION,
                             installed=True, version=None, voices=voices, supports_synthesize_streaming=True)
        asr = []
        if self.transcribe:
            name, languages = self.stt_model()
            model = AsrModel(name=name, description=f"Whisper ({name})", attribution=ATTRIBUTION,
                             installed=True, version=None, languages=languages)
            asr = [AsrProgram(name="hebrew-voice-studio-stt", description="Hebrew Voice Studio speech-to-text",
                              attribution=ATTRIBUTION, installed=True, version=None, models=[model])]
        return Info(tts=[program], asr=asr)

    def add_recording_chunk(self, chunk: AudioChunk) -> None:
        """Converted to 16 kHz mono and kept, up to MAX_RECORDING_SECONDS; a chunk in an odd
        format or far too large is dropped rather than converted."""
        if not (8000 <= chunk.rate <= 96000 and chunk.width in (1, 2, 3, 4) and 1 <= chunk.channels <= 8
                and len(chunk.audio) <= MAX_CHUNK_BYTES):
            return
        room = MAX_RECORDING_SECONDS * STT_RATE * 2 - sum(map(len, self.recording))
        if room > 0:
            self.recording.append(self.converter.convert(chunk).audio[:room])

    async def send_transcript(self, pcm: bytes) -> None:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        try:
            text = await asyncio.to_thread(self.transcribe, audio, self.language or "he")
        except Exception as e:  # a failed load (e.g. no GPU memory): say so instead of hanging up
            print(f"[wyoming] speech-to-text failed: {e}")
            await self.write_event(Error(text=f"Speech-to-text failed: {e}", code="stt-failed").event())
            return
        print(f"[wyoming] transcribed {len(audio) / STT_RATE:.1f}s of speech ({len(text)} characters)")
        await self.write_event(Transcript(text=text, language=self.language or "he").event())

    def voice_id(self, voice) -> str:
        """The voice Home Assistant asked for by name, else the default voice."""
        return voice.name if voice is not None and voice.name else self.default_voice()

    async def read_stream_part(self, text: str) -> None:
        """Part of a streamed text, after a sentence gap if something was already read."""
        await self.read_aloud(text, self.stream["voice"], lead=STREAM_GAP if self.stream["spoken"] else 0.0)
        self.stream["spoken"] = True

    async def read_aloud(self, text: str, voice_id: str, lead: float = 0.0) -> None:
        """AudioStart, the audio chunk by chunk as each sentence is ready, AudioStop. lead:
        seconds of silence first."""
        try:
            rate, pieces = await asyncio.to_thread(self.speak, text, voice_id)
            fmt = {"rate": rate, "width": 2, "channels": 1}
            await self.write_event(AudioStart(**fmt).event())
            if lead:
                await self.write_event(AudioChunk(audio=bytes(2 * int(rate * lead)), **fmt).event())
            while (piece := await asyncio.to_thread(next, pieces, None)) is not None:
                pcm = (np.clip(piece, -1.0, 1.0) * 32767).astype("<i2")
                for start in range(0, len(pcm), CHUNK_SAMPLES):
                    await self.write_event(AudioChunk(audio=pcm[start:start + CHUNK_SAMPLES].tobytes(), **fmt).event())
            await self.write_event(AudioStop().event())
        except ValueError as e:
            await self.write_event(Error(text=str(e), code="tts-failed").event())


def split_ready(text: str) -> tuple[str, str]:
    """Streamed text -> (complete sentences to read now, the unfinished rest)."""
    cut = 0
    for m in SENTENCE_END.finditer(text):
        cut = m.end()
    return text[:cut], text[cut:]


# --- Client: speech-to-text on another Wyoming server, such as wyoming-faster-whisper
# (lscr.io/linuxserver/faster-whisper, rhasspy/wyoming-whisper) on a home server.

CONNECT_TIMEOUT = 5.0
REMOTE_TIMEOUT = 300.0  # the whole transcription; a few minutes of audio takes seconds on a GPU


async def _read_until(client, wanted, timeout: float) -> Event:
    """The next event of type wanted; an Error event from the server is raised as ValueError."""
    async def read():
        while (event := await client.read_event()) is not None:
            if Error.is_type(event.type):
                raise ValueError(f"The server said: {Error.from_event(event).text}")
            if wanted.is_type(event.type):
                return event
        raise ValueError("The server closed the connection.")

    return await asyncio.wait_for(read(), timeout)


def _run(host: str, port: int, talk, timeout: float):
    """Connects to host:port and runs talk(client); connection failures as ValueError messages."""
    from wyoming.client import AsyncTcpClient

    async def main():
        client = AsyncTcpClient(host, port)
        try:
            await asyncio.wait_for(client.connect(), CONNECT_TIMEOUT)
        except (OSError, asyncio.TimeoutError) as e:
            raise ValueError(f"Couldn't connect to {host}:{port} ({e or 'timed out'}).")
        try:
            return await talk(client)
        except asyncio.TimeoutError:
            raise ValueError(f"No answer from {host}:{port} in {timeout:.0f} s.")
        finally:
            await client.disconnect()

    return asyncio.run(main())


def remote_models(host: str, port: int) -> list[str]:
    """The speech-to-text models a Wyoming server offers (its Describe answer)."""
    async def talk(client):
        await client.write_event(Describe().event())
        info = Info.from_event(await _read_until(client, Info, CONNECT_TIMEOUT * 2))
        return [m.name for program in info.asr for m in program.models]

    models = _run(host, port, talk, CONNECT_TIMEOUT * 2)
    if not models:
        raise ValueError(f"{host}:{port} answers, but offers no speech-to-text.")
    return models


def remote_transcribe(host: str, port: int, audio: np.ndarray, language: str | None) -> tuple[str, str | None]:
    """Mono float32 audio at 16 kHz -> (text, language the server reports, if it does).
    language None: the server's own setting (wyoming-faster-whisper: WHISPER_LANG, auto by default)."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    fmt = {"rate": STT_RATE, "width": 2, "channels": 1}

    async def talk(client):
        await client.write_event(Transcribe(language=language).event())
        await client.write_event(AudioStart(**fmt).event())
        step = CHUNK_SAMPLES * 2
        for start in range(0, len(pcm), step):
            await client.write_event(AudioChunk(audio=pcm[start:start + step], **fmt).event())
        await client.write_event(AudioStop().event())
        transcript = Transcript.from_event(await _read_until(client, Transcript, REMOTE_TIMEOUT))
        return BIDI_MARKS.sub("", transcript.text).strip(), transcript.language

    return _run(host, port, talk, REMOTE_TIMEOUT)


def start(host: str, port: int, voices: Callable[[], list[dict]], speak: Speak, default_voice: Callable[[], str],
          allowed: Callable[[str], bool], transcribe: TranscribeFn | None = None,
          stt_model: Callable[[], tuple[str, list[str]]] | None = None) -> None:
    """Serves Wyoming on host:port from a background thread; allowed(ip) decides who may connect.
    transcribe: speech-to-text, offered too when given; stt_model() -> (model name, languages)."""

    def run():
        server = AsyncTcpServer(host, port)
        try:
            asyncio.run(server.run(lambda r, w: Handler(r, w, voices, speak, default_voice, allowed, transcribe, stt_model)))
        except OSError as e:
            print(f"[WARN] Wyoming server not started on port {port}: {e}")

    threading.Thread(target=run, daemon=True, name="wyoming").start()
    print(f"Wyoming (Home Assistant) on port {port}")
