"""Wyoming protocol server, so Home Assistant's Wyoming integration can use this app as a local
text-to-speech service: Settings > Devices & services > Add integration > Wyoming Protocol,
host = this machine, port = 10200 (--wyoming-port).

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
from typing import Callable, Iterator

import numpy as np
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncTcpServer
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped

CHUNK_SAMPLES = 2048  # audio goes out in chunks of this many samples
STREAM_GAP = 0.25  # seconds of silence between parts of a streamed text, as between sentences
# Where a stream of text may be cut and read so far: after . ! ? : ; or a newline, at a space.
SENTENCE_END = re.compile(r"[.!?:;\n]+[\"'״)\]]*\s+")
ATTRIBUTION = Attribution(name="Hebrew Voice Studio", url="https://github.com/tm1981/hebrew-voice-studio")

# speak(text, voice_id) -> (sample_rate, pieces): audio as float32 arrays at that rate, the
# next piece computed when asked for. Raises ValueError with a message for the client.
Speak = Callable[[str, str], tuple[int, Iterator[np.ndarray]]]


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
                 allowed: Callable[[str], bool]):
        super().__init__(reader, writer)
        self.voices, self.speak, self.default_voice = voices, speak, default_voice
        self.stream: dict | None = None  # {"voice", "text", "spoken"} while text arrives in chunks
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
        return Info(tts=[program])

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


def start(host: str, port: int, voices: Callable[[], list[dict]], speak: Speak, default_voice: Callable[[], str],
          allowed: Callable[[str], bool]) -> None:
    """Serves Wyoming on host:port from a background thread; allowed(ip) decides who may connect."""

    def run():
        server = AsyncTcpServer(host, port)
        try:
            asyncio.run(server.run(lambda r, w: Handler(r, w, voices, speak, default_voice, allowed)))
        except OSError as e:
            print(f"[WARN] Wyoming server not started on port {port}: {e}")

    threading.Thread(target=run, daemon=True, name="wyoming").start()
    print(f"Wyoming (Home Assistant) on port {port}")
