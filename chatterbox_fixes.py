"""Workarounds for chatterbox-tts 0.1.7 bugs that affect Hebrew generation."""
import copy
import re
from pathlib import Path

import torch.nn.functional as F
from dicta_onnx import Dicta

import chatterbox.models.tokenizers.tokenizer as cb_tokenizer
from chatterbox.models.s3tokenizer import SPEECH_VOCAB_SIZE
from chatterbox.models.t3.inference.alignment_stream_analyzer import LLAMA_ALIGNED_HEADS
from model_files import ensure as ensure_model
from niqqud import RespectfulDicta

DICTA_MODEL = "dicta-1.0.int8.onnx"  # in models/, downloaded on first use
ALIGNED_LAYERS = {layer for layer, _ in LLAMA_ALIGNED_HEADS}

# Normal Hebrew speech measured at <= ~1.5 speech tokens per text token; runs far past
# that are the model inventing words after the sentence ends (one test sentence: 5.5).
# English needs more: its text has no niqqud, so fewer tokens per spoken second (measured
# 1.8-2.7, the higher for cloned voices); 1.6 cut the ends off English sentences.
MAX_SPEECH_PER_TEXT_TOKEN = 1.6
MAX_SPEECH_PER_ENGLISH_TOKEN = 3.2
MIN_SPEECH_TOKENS = 20
# Arabic has no niqqud either. 72 readings of 8 sentences: every normal one fit 2.6 per text
# token plus a fixed 50 (short sentences run long), and the 3 that invented speech didn't.
ARABIC_SPEECH_PER_TOKEN, ARABIC_EXTRA_TOKENS = 2.6, 50
HEBREW_LETTER, LATIN_LETTER = re.compile(r"[א-ת]"), re.compile(r"[A-Za-z]")
ARABIC_LETTER = re.compile(r"[ء-ي]")


def is_arabic(text: str) -> bool:
    """Whether a chunk is mostly Arabic, so Chatterbox should read it as Arabic."""
    return len(ARABIC_LETTER.findall(text)) > len(HEBREW_LETTER.findall(text)) + len(LATIN_LETTER.findall(text))


def speech_token_ratio(text: str) -> float:
    """Cap ratio for a chunk: Hebrew's, moving to English's with the share of Latin letters."""
    if is_arabic(text):
        return ARABIC_SPEECH_PER_TOKEN + ARABIC_EXTRA_TOKENS / (len(text) + 4)  # about 4 tokens more than letters
    hebrew, latin = len(HEBREW_LETTER.findall(text)), len(LATIN_LETTER.findall(text))
    share = latin / (hebrew + latin) if hebrew + latin else 0.0
    return MAX_SPEECH_PER_TEXT_TOKEN + share * (MAX_SPEECH_PER_ENGLISH_TOKEN - MAX_SPEECH_PER_TEXT_TOKEN)

# At 0.8 (Chatterbox's default) ~1 in 3 Hebrew sentences ran on with invented speech;
# at 0.4 none did in the same test.
DEFAULT_TEMPERATURE = 0.4


def load_dicta() -> RespectfulDicta:
    """Chatterbox calls Dicta() with no model path, which fails on current dicta-onnx.
    Inject a loaded instance so Hebrew text gets niqqud before tokenization, wrapped so
    words the user already pointed keep their niqqud instead of Dicta's guess."""
    cb_tokenizer._dicta = RespectfulDicta(Dicta(str(ensure_model("dicta", DICTA_MODEL))))
    return cb_tokenizer._dicta


def guard_speech_tokens(model) -> None:
    """When T3 forces an early EOS it can emit tokens >= SPEECH_VOCAB_SIZE that
    drop_invalid_tokens() lets through. The S3Gen decoder then indexes out of bounds,
    triggering a CUDA device-side assert that breaks the GPU context for every later
    request. Filter them out before decoding."""
    inference = model.s3gen.inference

    def safe_inference(speech_tokens, **kwargs):
        return inference(speech_tokens=speech_tokens[speech_tokens < SPEECH_VOCAB_SIZE], **kwargs)

    model.s3gen.inference = safe_inference


def enable_speed_control(model) -> None:
    """Chatterbox has no speed setting. Stretch the mel spectrogram in time before the
    HiFT vocoder, which re-derives pitch from the mel, so speed changes without
    changing pitch (the CosyVoice approach S3Gen inherits). Set model.speech_speed."""
    model.speech_speed = 1.0
    flow_inference = model.s3gen.flow_inference

    def stretched_flow_inference(*args, **kwargs):
        mels = flow_inference(*args, **kwargs)
        if model.speech_speed != 1.0:
            mels = F.interpolate(mels, size=max(1, int(mels.shape[-1] / model.speech_speed)), mode="linear")
        return mels

    model.s3gen.flow_inference = stretched_flow_inference


def optimize_t3(model) -> None:
    """Speed and stability fixes for the T3 token generator:
    - Every call creates an AlignmentStreamAnalyzer that adds forward hooks to three
      attention layers and never removes them, so each request gets slower. Remove them.
    - The analyzer switches all 30 layers to slow eager attention, though it only reads
      three. Point the other layers at an SDPA config (~20% faster, same output).
    - Cap generated tokens relative to text length to stop runaway invented speech; set
      model.t3.speech_ratio = speech_token_ratio(chunk) before each generate."""
    tfmr = model.t3.tfmr
    sdpa_config = copy.copy(tfmr.config)
    sdpa_config._attn_implementation = "sdpa"
    for i, layer in enumerate(tfmr.layers):
        if i not in ALIGNED_LAYERS:
            layer.self_attn.config = sdpa_config

    inference = model.t3.inference
    model.t3.speech_ratio = MAX_SPEECH_PER_TEXT_TOKEN

    def safe_inference(*args, text_tokens, max_new_tokens=None, **kwargs):
        cap = max(MIN_SPEECH_TOKENS, int(text_tokens.shape[-1] * model.t3.speech_ratio))
        max_new_tokens = min(max_new_tokens or cap, cap)
        before = {i: set(tfmr.layers[i].self_attn._forward_hooks) for i in ALIGNED_LAYERS}
        try:
            return inference(*args, text_tokens=text_tokens, max_new_tokens=max_new_tokens, **kwargs)
        finally:
            for i, keys in before.items():
                hooks = tfmr.layers[i].self_attn._forward_hooks
                for k in list(hooks):
                    if k not in keys:
                        del hooks[k]

    model.t3.inference = safe_inference
