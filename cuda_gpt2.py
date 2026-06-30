"""
cuda_gpt2.py
GPT-2 with a switchable attention backend:
  - backend="torch": F.scaled_dot_product_attention (the PyTorch baseline)
  - backend="cuda" : custom flash_prefill_causal + flash_decode kernels

Both paths use an identical KV cache, so the benchmark isolates the attention
compute. Weights are loaded from HuggingFace GPT-2 checkpoints (fp32).
"""
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# --- JIT-compile the CUDA extension on import -------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
ext = load(
    name="kvcache_ext",
    sources=[os.path.join(_HERE, "kvcache_kernels.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head

    def forward(self, x, k_cache, v_cache, pos, backend):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        # (B, nh, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # write new K/V into the cache at [pos, pos+T)
        k_cache[:, :, pos:pos + T, :] = k
        v_cache[:, :, pos:pos + T, :] = v
        seqlen = pos + T

        if backend == "torch":
            kk = k_cache[:, :, :seqlen, :]
            vv = v_cache[:, :, :seqlen, :]
            # prefill: causal mask; decode (T==1): attend to all cached keys
            y = F.scaled_dot_product_attention(q, kk, vv, is_causal=(T > 1))

        elif backend == "cuda":
            if T > 1:  # prefill
                y = ext.flash_prefill_causal(q.contiguous(), k.contiguous(), v.contiguous())
            else:      # decode
                qd = q.squeeze(2).contiguous()           # (B, nh, hd)
                y = ext.flash_decode(qd, k_cache, v_cache, seqlen)  # (B, nh, hd)
                y = y.unsqueeze(2)                       # (B, nh, 1, hd)
        else:
            raise ValueError(backend)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, k_cache, v_cache, pos, backend):
        x = x + self.attn(self.ln_1(x), k_cache, v_cache, pos, backend)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def _alloc_cache(self, B, max_len, device, dtype):
        nh, hd = self.cfg.n_head, self.cfg.n_embd // self.cfg.n_head
        return [(torch.zeros(B, nh, max_len, hd, device=device, dtype=dtype),
                 torch.zeros(B, nh, max_len, hd, device=device, dtype=dtype))
                for _ in range(self.cfg.n_layer)]

    def forward_step(self, idx, caches, pos, backend):
        """Run one chunk (prefill if T>1, else single decode token)."""
        B, T = idx.shape
        device = idx.device
        positions = torch.arange(pos, pos + T, device=device)
        x = self.transformer.wte(idx) + self.transformer.wpe(positions)
        for blk, (kc, vc) in zip(self.transformer.h, caches):
            x = blk(x, kc, vc, pos, backend)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)  # (B, T, vocab)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, backend, greedy=True):
        B, T0 = idx.shape
        max_len = T0 + max_new_tokens
        caches = self._alloc_cache(B, max_len, idx.device, next(self.parameters()).dtype)

        logits = self.forward_step(idx, caches, pos=0, backend=backend)
        next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
        out = [next_tok]
        pos = T0
        for _ in range(max_new_tokens - 1):
            logits = self.forward_step(next_tok, caches, pos=pos, backend=backend)
            next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
            out.append(next_tok)
            pos += 1
        return torch.cat([idx] + out, dim=1)

    @classmethod
    def from_pretrained(cls, model_type="gpt2"):
        from transformers import GPT2LMHeadModel
        cfg_args = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        cfg = GPTConfig(**cfg_args)
        model = cls(cfg)
        sd = model.state_dict()
        keys = [k for k in sd if not k.endswith(".attn.bias")]

        hf = GPT2LMHeadModel.from_pretrained(model_type)
        hf_sd = hf.state_dict()
        # HF uses Conv1D for these -> transpose into nn.Linear
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight",
                      "mlp.c_fc.weight", "mlp.c_proj.weight"]
        hf_keys = [k for k in hf_sd
                   if not k.endswith(".attn.masked_bias")
                   and not k.endswith(".attn.bias")]
        for k in hf_keys:
            tgt = k
            if tgt not in sd:
                continue
            if any(k.endswith(w) for w in transposed):
                assert hf_sd[k].shape[::-1] == sd[tgt].shape
                with torch.no_grad():
                    sd[tgt].copy_(hf_sd[k].t())
            else:
                assert hf_sd[k].shape == sd[tgt].shape, (k, hf_sd[k].shape, sd[tgt].shape)
                with torch.no_grad():
                    sd[tgt].copy_(hf_sd[k])
        return model
