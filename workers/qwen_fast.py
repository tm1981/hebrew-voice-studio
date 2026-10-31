"""Fast greedy decoding for Qwen3-TTS (qwen-tts 0.1.1), batch 1, replacing talker.generate.

The stock loop runs HF generate for the talker and, inside every 12 Hz frame, HF generate again
for the code predictor's 15 residual codebooks: ~20 small forward passes a frame, each paying
Python and kernel-launch overhead. Here a whole frame (code predictor + one talker step) is one
CUDA graph over static KV caches, so a frame is a single graph replay, and the layers use fewer,
fused kernels (one GEMV for q/k/v and for gate/up, fused RMSNorm, bmm attention for one query).
On the 5090 that is 0.42x -> ~8.8x real time.

The math is the model's own weights and ops (it matches HF to 2e-6 in fp32). In bf16 the rounding
differs slightly, so greedy codes can part ways at near-ties (deep codebooks often have exact
bf16 ties), as they also do between HF's own code paths; Whisper hears the same words either way.

Only what the worker uses is handled: batch 1, greedy talker and code predictor, no repetition
penalty, non-streaming text (every frame adds tts_pad). Anything else falls back to the stock
generate.
"""
import torch
import torch.nn.functional as F

BUCKETS = (256, 512, 1024, 2048, 4096)  # talker KV lengths a graph attends over


def _rope(inv_freq, pos, dtype):
    """cos, sin of shape (len(pos), head_dim), as the model's rotary embeddings compute them.
    The talker's interleaved multimodal RoPE reduces to this when the 3 position rows are equal,
    which they always are for TTS text."""
    freqs = pos.float()[:, None] * inv_freq.float()[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rope_tables(cos, sin):
    """(T, D) cos/sin -> (1, T, 1, D) tables for _qkv: x*cos + roll(x)*sin equals HF's
    x*cos + rotate_half(x)*sin once the first half of sin is negated."""
    half = sin.shape[-1] // 2
    sin = torch.cat((-sin[..., :half], sin[..., half:]), dim=-1)
    return cos[None, :, None], sin[None, :, None]


def _fuse(layers):
    """Concatenate each layer's q/k/v and gate/up weights into one matrix (one GEMV instead of
    three or two). The modules' own weights become views into it, so no memory is duplicated and
    the stock generate path keeps working."""
    for layer in layers:
        attn, mlp = layer.self_attn, layer.mlp
        for name, linears in (("qkv", (attn.q_proj, attn.k_proj, attn.v_proj)), ("gate_up", (mlp.gate_proj, mlp.up_proj))):
            fused = torch.cat([l.weight.data for l in linears])
            for l, part in zip(linears, fused.split([l.out_features for l in linears])):
                l.weight = torch.nn.Parameter(part, requires_grad=False)
            setattr(layer, f"_fast_{name}", fused)
        heads = attn.config.num_attention_heads
        kv_heads = attn.config.num_key_value_heads
        layer._fast_qk_norm = torch.cat((attn.q_norm.weight.expand(heads, -1), attn.k_norm.weight.expand(kv_heads, -1)))
        layer._fast_heads = (heads, kv_heads)


def _qkv(layer, x, cos, sin):
    """x (1, T, H) -> q (1, heads, T, D), k and v (1, kv_heads, T, D), q/k-normed and RoPE'd."""
    attn = layer.self_attn
    heads, kv_heads = layer._fast_heads
    d = attn.head_dim
    qkv = F.linear(x, layer._fast_qkv).view(1, x.shape[1], -1, d)
    qk, v = qkv[:, :, :heads + kv_heads], qkv[:, :, heads + kv_heads:]
    qk = F.rms_norm(qk, (d,), eps=attn.q_norm.variance_epsilon) * layer._fast_qk_norm
    qk = qk * cos + qk.roll(d // 2, dims=-1) * sin
    return qk[:, :, :heads].transpose(1, 2), qk[:, :, heads:].transpose(1, 2), v.transpose(1, 2)


def _attend(attn, q, k, v, mask, causal=False):
    """q (1, heads, T, D), k/v (1, kv_heads, L, D) -> (1, T, heads*D).

    Decoding (T == 1) is plain baddbmm + softmax + bmm, as HF's eager attention computes it: SDPA's
    kernel takes ~14 us for one query row, three small kernels ~4 us. Grouped-query attention
    without repeating K/V: the query heads sharing a KV head become extra query rows of it.
    mask: additive, broadcastable to (kv_heads, groups, L)."""
    groups = attn.num_key_value_groups
    if causal:  # prefill: T > 1, once per utterance
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=attn.scaling)
        return out.transpose(1, 2).reshape(1, q.shape[2], -1)
    h, d = q.shape[1], q.shape[3]
    q = q.reshape(h // groups, groups, d)
    scores = torch.baddbmm(mask, q, k[0].transpose(1, 2), alpha=attn.scaling)
    return torch.bmm(torch.softmax(scores, dim=-1), v[0]).reshape(1, 1, h * d)


def _norm(norm, x):
    return F.rms_norm(x, x.shape[-1:], norm.weight, norm.variance_epsilon)


def _layer(layer, x, attend):
    """attend(layer, normed x) -> attention output (1, T, heads*D)."""
    x = x + layer.self_attn.o_proj(attend(layer, _norm(layer.input_layernorm, x)))
    gate, up = F.linear(_norm(layer.post_attention_layernorm, x), layer._fast_gate_up).chunk(2, dim=-1)
    return x + layer.mlp.down_proj(F.silu(gate) * up)


class FastTalker:
    def __init__(self, talker, max_len: int = 4096):
        self.talker = talker
        self.cp = talker.code_predictor
        cfg = talker.config
        p = next(talker.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.groups = cfg.num_code_groups
        self.eos = cfg.codec_eos_token_id
        self.max_len = max_len
        kv_heads, head_dim = cfg.num_key_value_heads, cfg.head_dim
        n = cfg.num_hidden_layers
        self.k = torch.zeros(n, 1, kv_heads, max_len, head_dim, device=self.device, dtype=self.dtype)
        self.v = torch.zeros_like(self.k)
        cpc = cfg.code_predictor_config
        self.cp_k = torch.zeros(cpc.num_hidden_layers, 1, cpc.num_key_value_heads, self.groups + 1,
                                cpc.head_dim, device=self.device, dtype=self.dtype)
        self.cp_v = torch.zeros_like(self.cp_k)
        self.cp_cos, self.cp_sin = _rope_tables(*_rope(  # (1, 16, 1, D)
            self.cp.model.rotary_emb.inv_freq, torch.arange(self.groups, device=self.device), self.dtype))
        _fuse(talker.model.layers)
        _fuse(self.cp.model.layers)
        # suppress_tokens: the top 1024 codec ids except EOS (control tokens, never audio)
        self.suppress = torch.zeros(cfg.vocab_size, device=self.device, dtype=torch.float32)
        self.suppress[cfg.vocab_size - 1024:] = float("-inf")
        self.suppress[self.eos] = 0.0
        # static graph inputs / outputs
        self.past_hidden = torch.zeros(1, 1, cfg.hidden_size, device=self.device, dtype=self.dtype)
        self.token = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        self.pos = torch.zeros(1, device=self.device, dtype=torch.long)
        self.pad_embed = torch.zeros(1, 1, cfg.hidden_size, device=self.device, dtype=self.dtype)
        self.eos_bias = torch.zeros(cfg.vocab_size, device=self.device, dtype=torch.float32)
        self.codes = torch.zeros(1, self.groups, device=self.device, dtype=torch.long)
        self.no_mask = torch.zeros(1, 1, 1, device=self.device, dtype=self.dtype)
        self.graphs = {}
        self.pool = None

    # ---- code predictor: 15 residual codebooks for one frame ----
    def _code_predictor(self, past_hidden, last_id_hidden):
        cp, model = self.cp, self.cp.model
        x = cp.small_to_mtp_projection(torch.cat((past_hidden, last_id_hidden), dim=1))
        cos, sin = self.cp_cos[:, :2], self.cp_sin[:, :2]

        def prefill(layer, h):
            q, k, v = _qkv(layer, h, cos, sin)
            i = layer.self_attn.layer_idx
            self.cp_k[i, :, :, :2] = k
            self.cp_v[i, :, :, :2] = v
            return _attend(layer.self_attn, q, k, v, None, causal=True)

        for layer in model.layers:
            x = _layer(layer, x, prefill)
        h = _norm(model.norm, x)[:, -1:]
        codes = [cp.lm_head[0](h).argmax(-1)]
        embeds = [model.codec_embedding[0](codes[-1])]
        for step in range(1, self.groups - 1):
            x = cp.small_to_mtp_projection(embeds[-1])
            pos = step + 1
            cos, sin = self.cp_cos[:, pos:pos + 1], self.cp_sin[:, pos:pos + 1]

            def decode(layer, h):
                q, k, v = _qkv(layer, h, cos, sin)
                i = layer.self_attn.layer_idx
                self.cp_k[i, :, :, pos:pos + 1] = k
                self.cp_v[i, :, :, pos:pos + 1] = v
                return _attend(layer.self_attn, q, self.cp_k[i, :, :, :pos + 1], self.cp_v[i, :, :, :pos + 1], self.no_mask)

            for layer in model.layers:
                x = _layer(layer, x, decode)
            codes.append(cp.lm_head[step](_norm(model.norm, x)).argmax(-1))
            embeds.append(model.codec_embedding[step](codes[-1]))
        return torch.cat(codes, dim=1), embeds

    # ---- talker ----
    def _talker_layers(self, x, cos, sin, length=None, mask=None):
        """Prefill when length is None (positions 0..T-1), else one decode step at self.pos
        attending over the first `length` cache slots."""
        def attend(layer, h):
            attn = layer.self_attn
            q, k, v = _qkv(layer, h, cos, sin)
            i = attn.layer_idx
            if length is None:
                self.k[i, :, :, :k.shape[2]] = k
                self.v[i, :, :, :v.shape[2]] = v
                return _attend(attn, q, k, v, None, causal=True)
            self.k[i].index_copy_(2, self.pos, k)
            self.v[i].index_copy_(2, self.pos, v)
            return _attend(attn, q, self.k[i, :, :, :length], self.v[i, :, :, :length], mask)

        for layer in self.talker.model.layers:
            x = _layer(layer, x, attend)
        return _norm(self.talker.model.norm, x)

    def _frame(self, length):
        """One frame: residual codes for self.token, then the talker step producing the next token."""
        t = self.talker
        last_id_hidden = t.get_input_embeddings()(self.token)
        rest, embeds = self._code_predictor(self.past_hidden, last_id_hidden)
        self.codes.copy_(torch.cat((self.token, rest), dim=1))
        x = torch.cat([last_id_hidden] + embeds, dim=1).sum(1, keepdim=True) + self.pad_embed
        cos, sin = _rope(t.model.rotary_emb.inv_freq, self.pos, self.dtype)
        mask = torch.zeros(1, 1, length, device=self.device, dtype=self.dtype)
        mask.masked_fill_(torch.arange(length, device=self.device) > self.pos, float("-inf"))
        hidden = self._talker_layers(x, *_rope_tables(cos, sin), length, mask)
        self.past_hidden.copy_(hidden)
        logits = t.codec_head(hidden[:, -1]).float() + self.suppress + self.eos_bias
        self.token.copy_(logits.argmax(-1, keepdim=True))

    def _graph(self, length):
        if length not in self.graphs:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            # warm-up writes only the KV slot at self.pos, which the real frame rewrites
            saved = [b.clone() for b in (self.past_hidden, self.token)]
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._frame(length)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                self._frame(length)
            self.pool = graph.pool()
            for b, s in zip((self.past_hidden, self.token), saved):
                b.copy_(s)
            self.graphs[length] = graph
        return self.graphs[length]

    @torch.inference_mode()
    def capture(self):
        """Capture every bucket's graph up front, so no request pays for a capture."""
        self.pos.zero_()
        for length in BUCKETS:
            if length <= self.max_len:
                self._graph(length)

    @torch.inference_mode()
    def generate(self, inputs_embeds, max_new_tokens, min_new_tokens=2):
        """inputs_embeds (1, T, H) -> codes (frames, 16) up to (not including) EOS."""
        t = self.talker
        prompt = inputs_embeds.shape[1]
        cos, sin = _rope(t.model.rotary_emb.inv_freq, torch.arange(prompt, device=self.device), self.dtype)
        hidden = self._talker_layers(inputs_embeds, *_rope_tables(cos, sin))
        self.past_hidden.copy_(hidden[:, -1:])
        logits = t.codec_head(hidden[:, -1]).float() + self.suppress
        if min_new_tokens > 0:
            logits[:, self.eos] = float("-inf")
        self.token.copy_(logits.argmax(-1, keepdim=True))
        frames = []
        for step in range(1, max_new_tokens):  # token `step - 1` is in self.token
            if self.token.item() == self.eos:
                break
            pos = prompt + step - 1  # < max_len: install() only sends prompts that fit
            self.pos.fill_(pos)
            self.eos_bias[self.eos] = float("-inf") if step < min_new_tokens else 0.0
            length = next(b for b in BUCKETS if b > pos)
            self._graph(length).replay()
            frames.append(self.codes.clone())
        return torch.cat(frames) if frames else torch.zeros(0, self.groups, dtype=torch.long, device=self.device)


class _Result:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


def install(model, max_len: int = 4096):
    """Route model.talker.generate (called by Qwen3TTSForConditionalGeneration.generate) through
    FastTalker when the request is one it handles."""
    talker = model.talker
    fast = FastTalker(talker, max_len)
    stock = talker.generate

    def generate(inputs_embeds=None, attention_mask=None, trailing_text_hidden=None, tts_pad_embed=None, **kw):
        handled = (
            inputs_embeds is not None and inputs_embeds.shape[0] == 1
            and not kw.get("do_sample") and not kw.get("subtalker_dosample")
            and kw.get("repetition_penalty", 1.0) == 1.0
            and trailing_text_hidden is not None and trailing_text_hidden.shape[1] == 1
            and inputs_embeds.shape[1] + kw.get("max_new_tokens", 2048) <= fast.max_len
        )
        if not handled:
            return stock(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                         trailing_text_hidden=trailing_text_hidden, tts_pad_embed=tts_pad_embed, **kw)
        # non-streaming mode: trailing text is just tts_pad, which every frame adds
        fast.pad_embed.copy_(trailing_text_hidden[:, :1])
        codes = fast.generate(inputs_embeds, kw.get("max_new_tokens", 2048), kw.get("min_new_tokens", 2))
        # The shape Qwen3TTSForConditionalGeneration.generate reads: per step (hidden_states, codec_ids),
        # codec_ids None for the prefill step; it keeps codes up to the first EOS.
        dummy = (inputs_embeds[:, -1:],)
        steps = [(dummy, None)] + [(dummy, codes[i:i + 1]) for i in range(codes.shape[0])]
        steps.append((dummy, torch.full((1, fast.groups), fast.eos, device=codes.device)))
        return _Result(steps)

    talker.generate = generate
    return fast
