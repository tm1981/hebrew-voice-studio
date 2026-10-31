"""QwenTTS-he worker (conda env 'qwentts': qwen-tts pins transformers 4.57.3).

Qwen3-TTS-12Hz-1.7B-Base with notmax123's Hebrew LoRA adapter on its talker. The adapter was
trained on stressed IPA, not Hebrew script, so text goes through RenikudPlus first (the G2P
the training data used; typed niqqud decides a word's reading).

Voices are all clones (the Base model has no named speakers), from the speaker embedding of
a recording only (x_vector_only_mode, as the adapter was sampled):
  female, male  - the adapter's own Hebrew samples as reference
  <custom>      - recordings in voices/; the embedding is cached as voices/.cache/<name>.qwen.pt
"""
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download

from worker_common import serve

FAST = True  # qwen_fast's CUDA-graph decode loop instead of HF generate (~20x faster)

ROOT = Path(__file__).resolve().parent.parent
BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
ADAPTER = "notmax123/QwenTTS-he-1.7B"
sys.path.insert(0, str(ROOT))
from model_files import ensure as ensure_model  # noqa: E402

RENIKUD = "renikud-plus/model.onnx"  # in models/, downloaded on first use
SAMPLE_VOICES = {"female": "samples/00_female1_hebrew.wav", "male": "samples/01_male1_hebrew.wav"}
# The adapter card's settings: greedy decoding, no repetition penalty.
GENERATE = dict(
    language="Auto", non_streaming_mode=True, do_sample=False, subtalker_dosample=False,
    repetition_penalty=1.0, max_new_tokens=2048,
)


class QwenBackend:
    native_speed = False
    custom_voices = True

    def __init__(self):
        self.voices: dict = {}
        self.builtin_voices = set(SAMPLE_VOICES)
        self.sample_rate = 24000

    def load(self):
        from peft import PeftModel
        from qwen_tts import Qwen3TTSModel
        from renikud_onnx import G2P

        print("Loading QwenTTS-he...", flush=True)
        self.g2p = G2P(str(ensure_model("renikud-plus", RENIKUD)), niqqud="use")
        # Local folders, not repo ids: given an id, transformers 4.57 asks the Hub about it even
        # when the files are cached, and fails offline.
        self.tts = Qwen3TTSModel.from_pretrained(
            snapshot_download(BASE), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        talker = PeftModel.from_pretrained(self.tts.model.talker, snapshot_download(ADAPTER))
        # Merged in memory (the files stay as they are): 1.65x faster on the 5090 (0.30x -> 0.50x
        # real time). The adapter card advises against merging only to keep the base model's
        # other languages, which this Hebrew engine doesn't use.
        self.tts.model.talker = talker.merge_and_unload()
        self.tts.model.eval()
        if FAST:
            import qwen_fast

            qwen_fast.install(self.tts.model).capture()
        for name, file in SAMPLE_VOICES.items():
            self.voices[name] = self._prompt(hf_hub_download(ADAPTER, file))
        self.synthesize("שלום, מה שלומך היום?", "female", 1.0)  # warm-up

    def _prompt(self, path: str):
        return self.tts.create_voice_clone_prompt(ref_audio=path, x_vector_only_mode=True)

    def add_voice(self, name: str, path: Path):
        cache = path.with_suffix(".qwen.pt")
        if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
            from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

            embedding = torch.load(cache, map_location="cuda:0", weights_only=True)
            prompt = [VoiceClonePromptItem(ref_code=None, ref_spk_embedding=embedding, x_vector_only_mode=True, icl_mode=False)]
        else:
            prompt = self._prompt(str(path))
            torch.save(prompt[0].ref_spk_embedding, cache)
        self.voices[name] = prompt

    def remove_voice(self, name: str):
        self.voices.pop(name, None)

    def synthesize(self, text: str, voice: str, speed: float, sampling=False, temperature=0.9, top_p=1.0,
                   repetition_penalty=1.0, seed=None, **_):
        """sampling or a repetition penalty leaves the fast greedy path (qwen_fast) for the stock
        generate, which is several times slower."""
        settings = dict(GENERATE, repetition_penalty=repetition_penalty)
        if sampling:
            settings.update(do_sample=True, subtalker_dosample=True, temperature=temperature, top_p=top_p)
        if seed is not None:
            torch.manual_seed(seed)
        wavs, self.sample_rate = self.tts.generate_voice_clone(
            text=self.g2p.phonemize(text), voice_clone_prompt=self.voices[voice], **settings
        )
        return wavs[0]


if __name__ == "__main__":
    serve(QwenBackend(), "QwenTTS-he worker")
