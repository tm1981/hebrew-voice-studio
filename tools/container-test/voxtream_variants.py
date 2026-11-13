"""How voxtream2-he reads with different reference clips and sampling settings, counted by Whisper.

Runs inside the container, in voxtream's own env (it loads the model directly, not through the app):

    wsl docker exec voice-wsl /opt/envs/voxtream/bin/python -X utf8 /app/tools/container-test/voxtream_variants.py [variant ...]

For each variant and each recorded voice it reads the 15 test sentences, then Whisper (the app's
Hebrew model) listens. A sentence is "wrong" when words were added or dropped, or a run of words
was replaced by a different number of words; "one word off" is a same-length swap, often just
Whisper's spelling. Also reported: seconds of silence the model put before the speech (trimmed
from the result, but waited for), and audio seconds made per second of work.
"""
import difflib
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/app/workers")
import torchaudio  # noqa: E402


def load_with_soundfile(path, *_, **__):  # as workers/voxtream_worker.py does
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(audio.T.copy()), rate


torchaudio.load = load_with_soundfile
from voxtream2_he import SpeechGenerator  # noqa: E402
from voxtream_worker import trim_silence  # noqa: E402

HERE = Path(__file__).resolve().parent
SENTENCES = [s for s in (HERE / "hebrew_sentences.txt").read_text(encoding="utf-8").split("\n") if s.strip()]
VOICES = sorted(p for p in Path("/data/voices/.cache").glob("*.wav") if p.name.count(".") == 1)
OUT = Path("/tmp/voxtream-variants")

# name -> (which clip, clean it with Sidon, generator settings, generate_stream settings)
VARIANTS = {
    "now": ("full", True, {}, {}),                        # what the worker does today
    "sentence-clip": ("ref", True, {}, {}),               # the clip cut after a finished sentence
    "sentence-clip-raw": ("ref", False, {}, {}),          # the same, not cleaned
    "cooler": ("ref", True, {"temperature": 0.7}, {}),
    "more-guidance": ("ref", True, {}, {"cfg_gamma": 2.0}),
    "cooler-guided": ("ref", True, {"temperature": 0.7}, {"cfg_gamma": 2.0}),
}


def clip_for(voice: Path, kind: str) -> Path:
    if kind == "full":
        return voice
    from reference import reference

    return reference(voice)[0]


def words(text: str) -> list[str]:
    return re.sub(r"[^א-ת ]", "", text).split()


def main() -> None:
    chosen = sys.argv[1:] or [n for n, v in VARIANTS.items() if not v[2]]
    # The library can build its model only once per process, so one run takes variants that share
    # their generator settings (the default: those with none). Run the others separately.
    if len({tuple(sorted(VARIANTS[n][2].items())) for n in chosen}) > 1:
        raise SystemExit("These variants need different generator settings: run them in separate runs.")
    generators, made = {}, {}
    for name in chosen:
        kind, clean, settings, stream = VARIANTS[name]
        key = tuple(sorted(settings.items()))
        if key not in generators:
            generators[key] = SpeechGenerator(reset_streaming_state=True, **settings)
        gen = generators[key]
        phonemizer, rate = gen.make_phonemizer(), gen.config.mimi_sr
        for voice in VOICES:
            prompt = gen.prepare_prompt_from_path(str(clip_for(voice, kind)), enhance_with_sidon=clean)
            folder = OUT / name
            folder.mkdir(parents=True, exist_ok=True)
            lead = work = sound = 0.0
            for i, text in enumerate(SENTENCES):
                torch.manual_seed(i + 1)
                started = time.time()
                frames = [np.asarray(f, dtype=np.float32) for f, _ in gen.generate_stream(
                    text=phonemizer.phonemize(text, language="he"), prepared_prompt=prompt, language="he",
                    normalize_text=False, input_is_phonemes=True, **stream)]
                raw = np.concatenate(frames) if frames else np.zeros(1, np.float32)
                audio = trim_silence(raw, rate)
                work += time.time() - started
                sound += len(audio) / rate
                lead += max(len(raw) - len(audio), 0) / rate
                sf.write(folder / f"{voice.stem}_{i}.wav", audio, rate)
            made[(name, voice.stem)] = (lead, sound / max(work, 1e-9))
            print(f"made {name} / {voice.stem}", flush=True)
    del generators
    torch.cuda.empty_cache()

    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", "ivrit-ai/whisper-large-v3-turbo",
                   revision="f33172a8c3c6efbc040a7200e00835257cac0447", device="cuda", torch_dtype=torch.float16)
    print(f"\n{'variant':18s} {'voice':14s} wrong  one-off  silence-before  speed")
    for (name, voice), (lead, speed) in made.items():
        wrong = off = 0
        for i, text in enumerate(SENTENCES):
            audio, rate = sf.read(OUT / name / f"{voice}_{i}.wav", dtype="float32")
            heard = asr({"raw": audio, "sampling_rate": rate}, generate_kwargs={"language": "he", "task": "transcribe"})["text"]
            ops = difflib.SequenceMatcher(None, words(text), words(heard), autojunk=False).get_opcodes()
            bad = any(op in ("insert", "delete") or (op == "replace" and i2 - i1 != j2 - j1) for op, i1, i2, j1, j2 in ops)
            wrong += bad
            off += (not bad) and any(op == "replace" for op, *_ in ops)
        print(f"{name:18s} {voice:14s} {wrong:2d}/15  {off:2d}/15   {lead:6.1f} s        {speed:.1f}x", flush=True)


if __name__ == "__main__":
    main()
