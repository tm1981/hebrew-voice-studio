"""Fast decoding for Chatterbox Multilingual's T3 token generator (chatterbox-tts 0.1.7).

The stock loop spends about 18 ms a token on the 5090, and most of that isn't the model: HF's
per-layer Python, a growing KV cache, and attention weights copied to the CPU. Here one token is
one CUDA graph over static KV caches, with the layers' own weights and fused kernels, so only
sampling and the alignment checks run in Python.

Nothing about the reading changes: the same embeddings, the same CFG, the stock AlignmentStreamAnalyzer
(its hallucination checks read the same three attention heads), and the same logit processors in the
same order, so a seed picks the same tokens. Anything unusual falls back to the stock loop.
"""
import torch
import torch.nn.functional as F
from chatterbox.models.t3.inference.alignment_stream_analyzer import LLAMA_ALIGNED_HEADS, AlignmentStreamAnalyzer
from transformers.generation.logits_process import MinPLogitsWarper, RepetitionPenaltyLogitsProcessor, TopPLogitsWarper

BUCKETS = (256, 512, 1024)  # KV lengths a graph attends over; the last is the longest reading handled
BATCH = 2  # conditional and unconditional, for CFG


class _Analyzer(AlignmentStreamAnalyzer):
    """The stock checks, fed by FastT3 instead of by forward hooks on the model."""

    def _add_attention_spy(self, *_):
        pass

    def step(self, logits, next_token=None):
        i, j = self.text_tokens_slice  # the stock check for repeating skips the last 5 text tokens,
        return logits if j - i <= 5 else super().step(logits, next_token=next_token)  # so "No." fails


def _fuse(layers):
    """One matrix for each layer's q/k/v and gate/up. The modules' weights become views of it, so
    no memory is duplicated and the stock loop keeps working."""
    for layer in layers:
        attn, mlp = layer.self_attn, layer.mlp
        for name, linears in (("qkv", (attn.q_proj, attn.k_proj, attn.v_proj)), ("gate_up", (mlp.gate_proj, mlp.up_proj))):
            fused = torch.cat([l.weight.data for l in linears])
            for l, part in zip(linears, fused.split([l.out_features for l in linears])):
                l.weight = torch.nn.Parameter(part, requires_grad=False)
            setattr(layer, f"_fast_{name}", fused)


def _norm(norm, x):
    return F.rms_norm(x, x.shape[-1:], norm.weight, norm.variance_epsilon)


class FastT3:
    def __init__(self, t3):
        self.t3 = t3
        self.tfmr = t3.tfmr
        self.layers = t3.tfmr.layers
        self.max_len = BUCKETS[-1]
        p = next(t3.parameters())
        self.device, self.dtype = p.device, p.dtype
        cfg = t3.cfg
        self.heads, self.head_dim, self.hidden = cfg.num_attention_heads, cfg.head_dim, cfg.hidden_size
        self.scale = self.head_dim ** -0.5
        self.aligned_heads = {layer: (slot, head) for slot, (layer, head) in enumerate(LLAMA_ALIGNED_HEADS)}
        _fuse(self.layers)
        n = len(self.layers)
        size = (n, BATCH * self.heads, self.max_len, self.head_dim)
        self.k = torch.zeros(size, device=self.device, dtype=self.dtype)
        self.v = torch.zeros_like(self.k)
        # The model's own RoPE, as tables. sin's first half is negated so x*cos + roll(x)*sin equals
        # HF's x*cos + rotate_half(x)*sin.
        positions = torch.arange(self.max_len, device=self.device)[None]
        cos, sin = self.tfmr.rotary_emb(torch.zeros(1, self.max_len, self.hidden, device=self.device, dtype=self.dtype), positions)
        half = self.head_dim // 2
        sin = torch.cat((-sin[..., :half], sin[..., half:]), dim=-1)
        self.cos, self.sin = cos[:, :, None].to(self.dtype), sin[:, :, None].to(self.dtype)
        # static graph inputs / outputs
        self.token = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        self.speech_pos = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        self.pos = torch.zeros(1, device=self.device, dtype=torch.long)
        self.logits = torch.zeros(BATCH, t3.speech_head.out_features, device=self.device, dtype=self.dtype)
        self.aligned = torch.zeros(len(LLAMA_ALIGNED_HEADS), self.max_len, device=self.device, dtype=self.dtype)
        self.graphs = {}
        self.pool = None

    def _qkv(self, layer, x, cos, sin):
        """x (B, T, H) -> q, k, v (B, heads, T, D), q and k RoPE'd."""
        batch, length = x.shape[:2]
        qkv = F.linear(x, layer._fast_qkv).view(batch, length, 3 * self.heads, self.head_dim)
        qk, v = qkv[:, :, :2 * self.heads], qkv[:, :, 2 * self.heads:]
        qk = qk * cos + qk.roll(self.head_dim // 2, dims=-1) * sin
        return qk[:, :, :self.heads].transpose(1, 2), qk[:, :, self.heads:].transpose(1, 2), v.transpose(1, 2)

    def _layer(self, layer, x, attend):
        x = x + layer.self_attn.o_proj(attend(layer, _norm(layer.input_layernorm, x)))
        gate, up = F.linear(_norm(layer.post_attention_layernorm, x), layer._fast_gate_up).chunk(2, dim=-1)
        return x + layer.mlp.down_proj(F.silu(gate) * up)

    def _prefill(self, x):
        """Conditioning, text and BOS (B, T, H). Fills the caches and self.logits, and returns the
        three aligned heads' attention maps (T, T), as the stock analyzer's first step reads them."""
        length = x.shape[1]
        cos, sin = self.cos[:, :length], self.sin[:, :length]
        maps = [None] * len(LLAMA_ALIGNED_HEADS)

        def attend(layer, h):
            i = layer.self_attn.layer_idx
            q, k, v = self._qkv(layer, h, cos, sin)
            self.k[i, :, :length] = k.flatten(0, 1)
            self.v[i, :, :length] = v.flatten(0, 1)
            if i in self.aligned_heads:
                slot, head = self.aligned_heads[i]
                scores = q @ k.transpose(-1, -2) * self.scale
                future = torch.ones(length, length, device=self.device, dtype=torch.bool).triu(1)
                weights = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
                maps[slot] = weights[0, head].cpu()
                out = weights @ v
            else:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return out.transpose(1, 2).reshape(BATCH, length, self.hidden)

        for layer in self.layers:
            x = self._layer(layer, x, attend)
        self.logits.copy_(self.t3.speech_head(_norm(self.tfmr.norm, x[:, -1:]))[:, 0])
        return maps

    def _step(self, length):
        """One token: self.token at self.pos -> self.logits and self.aligned."""
        t3 = self.t3
        x = (t3.speech_emb(self.token) + t3.speech_pos_emb.emb(self.speech_pos)).expand(BATCH, -1, -1)
        cos, sin = self.cos.index_select(1, self.pos), self.sin.index_select(1, self.pos)
        mask = torch.zeros(1, 1, length, device=self.device, dtype=self.dtype)
        mask.masked_fill_(torch.arange(length, device=self.device) > self.pos, float("-inf"))

        def attend(layer, h):
            i = layer.self_attn.layer_idx
            q, k, v = self._qkv(layer, h, cos, sin)
            self.k[i].index_copy_(1, self.pos, k.flatten(0, 1))
            self.v[i].index_copy_(1, self.pos, v.flatten(0, 1))
            keys, values = self.k[i, :, :length], self.v[i, :, :length]
            scores = torch.baddbmm(mask, q.flatten(0, 1), keys.transpose(1, 2), alpha=self.scale)
            weights = torch.softmax(scores, dim=-1)
            if i in self.aligned_heads:
                slot, head = self.aligned_heads[i]
                self.aligned[slot, :length] = weights[head, 0]  # head of the conditional batch
            return torch.bmm(weights, values).reshape(BATCH, 1, self.hidden)

        for layer in self.layers:
            x = self._layer(layer, x, attend)
        self.logits.copy_(t3.speech_head(_norm(self.tfmr.norm, x))[:, 0])

    def _graph(self, length):
        if length not in self.graphs:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._step(length)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                self._step(length)
            self.pool = graph.pool()
            self.graphs[length] = graph
        return self.graphs[length]

    @torch.inference_mode()
    def capture(self):
        """Capture every bucket's graph when the model loads, so no reading pays for it."""
        for length in BUCKETS:
            self._graph(length)

    @torch.inference_mode()
    def inference(self, *, t3_cond, text_tokens, max_new_tokens, temperature, top_p, min_p, repetition_penalty, cfg_weight):
        t3, hp = self.t3, self.t3.hp
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)
        start = hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, len_cond = t3.prepare_input_embeds(
            t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=start, cfg_weight=cfg_weight)
        bos = torch.tensor([[hp.start_speech_token]], dtype=torch.long, device=self.device)
        bos_embed = t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(0)
        inputs = torch.cat([embeds, bos_embed.expand(BATCH, -1, -1)], dim=1)

        text_end = len_cond + text_tokens.size(-1)
        analyzer = _Analyzer(self.tfmr, None, text_tokens_slice=(len_cond, text_end), eos_idx=hp.stop_speech_token)
        analyzer.last_aligned_attns = self._prefill(inputs)

        min_p_warper, top_p_warper = MinPLogitsWarper(min_p=min_p), TopPLogitsWarper(top_p=top_p)
        repetition = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
        ids = torch.zeros(1, max_new_tokens + 1, device=self.device, dtype=torch.long)
        ids[0, 0] = hp.start_speech_token
        cfg = torch.as_tensor(cfg_weight, device=self.device, dtype=self.dtype)
        last, predicted = hp.start_speech_token, []
        for i in range(max_new_tokens):
            cond, uncond = self.logits[:1], self.logits[1:2]
            logits = analyzer.step(cond + cfg * (cond - uncond), next_token=last)
            so_far = ids[:, :i + 1]
            logits = repetition(so_far, logits)
            if temperature != 1.0:
                logits = logits / temperature
            logits = top_p_warper(so_far, min_p_warper(so_far, logits))
            token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            predicted.append(token)
            ids[0, i + 1] = token
            last = token.item()
            if last == hp.stop_speech_token:
                break
            self.token.copy_(token)
            self.speech_pos.fill_(i + 1)
            pos = inputs.shape[1] + i
            self.pos.fill_(pos)
            self._graph(next(b for b in BUCKETS if b > pos)).replay()
            maps = self.aligned[:, :text_end].cpu()
            analyzer.last_aligned_attns = [m[None] for m in maps]
        return torch.cat(predicted, dim=1)


def install(t3) -> None:
    """Route t3.inference through FastT3 when the request is one it handles. Call it before
    chatterbox_fixes.optimize_t3, which wraps t3.inference."""
    if not t3.hp.is_multilingual or t3.is_gpt or t3.device.type != "cuda":
        return
    fast = FastT3(t3)
    fast.capture()
    stock = t3.inference

    def inference(*, t3_cond, text_tokens, max_new_tokens=None, temperature=0.8, top_p=0.95, min_p=0.05,
                  repetition_penalty=1.2, cfg_weight=0.5, **other):
        tokens = torch.atleast_2d(text_tokens)
        prompt = t3.prepare_conditioning(t3_cond).shape[1] + tokens.shape[1] + 2  # conditioning, text, start, BOS
        if other or tokens.shape[0] != BATCH or not max_new_tokens or prompt + max_new_tokens > fast.max_len:
            return stock(t3_cond=t3_cond, text_tokens=text_tokens, max_new_tokens=max_new_tokens, temperature=temperature,
                         top_p=top_p, min_p=min_p, repetition_penalty=repetition_penalty, cfg_weight=cfg_weight, **other)
        return fast.inference(t3_cond=t3_cond, text_tokens=tokens, max_new_tokens=max_new_tokens, temperature=temperature,
                              top_p=top_p, min_p=min_p, repetition_penalty=repetition_penalty, cfg_weight=cfg_weight)

    t3.inference = inference
    t3.stock_inference = stock
