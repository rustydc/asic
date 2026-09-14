"""Research PyTorch model for a hardware-shaped, fixed-weight LLM.

The decoder repeats three bounded-state recurrent layers followed by one sparse
global layer.  This is a correctness-oriented reference implementation; the
global retrieval scan intentionally materializes token-by-block scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ASICLMConfig:
    """Model geometry and runtime configuration."""

    vocab_size: int = 128_256
    hidden_size: int = 4096
    num_layers: int = 32
    intermediate_size: int = 14_336
    recurrent_intermediate_size: Optional[int] = None
    global_intermediate_size: Optional[int] = None
    num_heads: int = 32
    head_dim: int = 128
    recurrent_every: int = 4
    global_layer_offset: int = 3
    local_window: int = 512
    retrieval_block_size: int = 16
    top_blocks: int = 32
    index_dim: int = 128
    index_bits: int = 4
    rms_eps: float = 1e-5
    tie_embeddings: bool = True
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.num_heads * self.head_dim != self.hidden_size:
            raise ValueError("num_heads * head_dim must equal hidden_size")
        if self.recurrent_every <= 0:
            raise ValueError("recurrent_every must be positive")
        if self.num_layers <= 0 or self.num_layers % self.recurrent_every:
            raise ValueError("num_layers must be a positive multiple of recurrent_every")
        if not 0 <= self.global_layer_offset < self.recurrent_every:
            raise ValueError("global_layer_offset must select a layer in the pattern")
        for name in ("local_window", "retrieval_block_size", "top_blocks", "index_dim", "index_bits"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.index_bits > 8:
            raise ValueError("index_bits must be at most 8")
        for name in ("intermediate_size", "recurrent_intermediate_size", "global_intermediate_size"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when specified")

    def is_global_layer(self, layer_idx: int) -> bool:
        return layer_idx % self.recurrent_every == self.global_layer_offset

    def layer_intermediate_size(self, layer_idx: int) -> int:
        """Return the FFN width, allowing weights to move toward R stages."""
        override = (self.global_intermediate_size if self.is_global_layer(layer_idx)
                    else self.recurrent_intermediate_size)
        return self.intermediate_size if override is None else override


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DeltaState:
    """Bounded recurrent state for one recurrent mixer."""

    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix

    def detach(self) -> "DeltaState":
        return DeltaState(self.matrix.detach())


class DeltaNetLikeMixer(nn.Module):
    """Fixed-shape delta-rule mixer with one D-by-D state per head."""

    def __init__(self, cfg: ASICLMConfig):
        super().__init__()
        d = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.beta_proj = nn.Linear(d, cfg.num_heads, bias=True)
        self.decay_proj = nn.Linear(d, cfg.num_heads, bias=True)

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> DeltaState:
        return DeltaState(torch.zeros(batch_size, self.num_heads, self.head_dim, self.head_dim,
                                      device=device, dtype=dtype))

    def forward(self, x: torch.Tensor, state: Optional[DeltaState] = None) -> tuple[torch.Tensor, DeltaState]:
        b, t, d = x.shape
        h, hd = self.num_heads, self.head_dim
        q = self.q_proj(x).view(b, t, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(b, t, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, t, h, hd).transpose(1, 2)
        q = F.normalize(q.float(), dim=-1).to(x.dtype)
        k = F.normalize(k.float(), dim=-1).to(x.dtype)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        decay = torch.sigmoid(self.decay_proj(x) + 4.0).transpose(1, 2)
        s = (state or self.initial_state(b, device=x.device, dtype=x.dtype)).matrix
        outs = []
        for i in range(t):
            qi, ki, vi = q[:, :, i], k[:, :, i], v[:, :, i]
            bi = beta[:, :, i].unsqueeze(-1)
            ai = decay[:, :, i].unsqueeze(-1).unsqueeze(-1)
            pred = torch.einsum("bhd,bhde->bhe", ki, s)
            update = ki.unsqueeze(-1) * (vi - pred).unsqueeze(-2)
            s = ai * s + bi.unsqueeze(-1) * update
            outs.append(torch.einsum("bhd,bhde->bhe", qi, s))
        y = torch.stack(outs, dim=2).transpose(1, 2).contiguous().view(b, t, d)
        return self.o_proj(y), DeltaState(s)


class SparseGlobalMixer(nn.Module):
    """Causal local attention plus a FULL scan of compressed old blocks.

    Every global layer independently builds and scans its own index. Candidate
    lists are not accepted or returned: keeping identical retrieval work at all
    global layers matches the balanced-stage appliance architecture.
    """

    def __init__(self, cfg: ASICLMConfig):
        super().__init__()
        self.cfg = cfg
        d, h, hd = cfg.hidden_size, cfg.num_heads, cfg.head_dim
        self.q_proj = nn.Linear(d, h * hd, bias=False)
        self.k_proj = nn.Linear(d, hd, bias=False)
        self.v_proj = nn.Linear(d, hd, bias=False)
        self.o_proj = nn.Linear(h * hd, d, bias=False)
        self.index_q = nn.Linear(d, cfg.index_dim, bias=False)
        self.index_k = nn.Linear(d, cfg.index_dim, bias=False)

    @staticmethod
    def _window(x: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        _, t, _ = x.shape
        padded = F.pad(x, (0, 0, width - 1, 0))
        windows = padded.unfold(1, width, 1).permute(0, 1, 3, 2).contiguous()
        pos = torch.arange(t, device=x.device)
        slot = torch.arange(width, device=x.device)
        valid = slot.unsqueeze(0) >= (width - 1 - pos).clamp_min(0).unsqueeze(1)
        return windows, valid

    def _block_mean(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        bs = self.cfg.retrieval_block_size
        nb = math.ceil(t / bs)
        pad = nb * bs - t
        xp = F.pad(x, (0, 0, 0, pad)).view(b, nb, bs, d)
        valid = F.pad(torch.ones(t, device=x.device, dtype=x.dtype), (0, pad)).view(nb, bs)
        counts = valid.sum(1).clamp_min(1)
        return (xp * valid.view(1, nb, bs, 1)).sum(2) / counts.view(1, nb, 1), counts

    def _fake_quantize_index(self, x: torch.Tensor) -> torch.Tensor:
        """Signed per-vector QAT with exactly 2**index_bits levels."""
        levels = 2 ** self.cfg.index_bits
        scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized = (x / scale).clamp(-1, 1)
        quantized = ((normalized + 1) * (levels - 1) / 2).round()
        quantized = (quantized * 2 / (levels - 1) - 1) * scale
        return x + (quantized - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h, hd, bs = self.cfg.num_heads, self.cfg.head_dim, self.cfg.retrieval_block_size
        w = min(self.cfg.local_window, t)
        q = self.q_proj(x).view(b, t, h, hd).transpose(1, 2) / math.sqrt(hd)
        k, v = self.k_proj(x), self.v_proj(x)
        k_local, local_valid = self._window(k, w)
        v_local, _ = self._window(v, w)
        local_scores = torch.einsum("bhtd,btwd->bhtw", q, k_local)
        local_scores = local_scores.masked_fill(~local_valid.view(1, 1, t, w), float("-inf"))

        k_block, _ = self._block_mean(k)
        v_block, _ = self._block_mean(v)
        index_source, _ = self._block_mean(self.index_k(x))
        index_query = F.normalize(self.index_q(x).float(), dim=-1)
        index_source = F.normalize(index_source.float(), dim=-1)
        index_query = self._fake_quantize_index(index_query)
        index_source = self._fake_quantize_index(index_source)
        retrieval_scores = torch.einsum("bti,bni->btn", index_query, index_source)
        nb = retrieval_scores.shape[-1]
        token_pos = torch.arange(t, device=x.device)
        block_end = (torch.arange(nb, device=x.device) + 1) * bs - 1
        eligible = block_end.view(1, nb) <= token_pos.view(t, 1) - w
        masked = retrieval_scores.masked_fill(~eligible.view(1, t, nb), float("-inf"))
        k_top = min(self.cfg.top_blocks, nb)
        top_idx = torch.topk(masked, k=k_top, dim=-1).indices
        top_valid = torch.gather(eligible.view(1, t, nb).expand(b, -1, -1), 2, top_idx)
        gather_idx = top_idx.unsqueeze(-1).expand(b, t, k_top, hd)
        kb = torch.gather(k_block.unsqueeze(1).expand(b, t, nb, hd), 2, gather_idx)
        vb = torch.gather(v_block.unsqueeze(1).expand(b, t, nb, hd), 2, gather_idx)
        global_scores = torch.einsum("bhtd,btkd->bhtk", q, kb)
        global_scores = global_scores.masked_fill(~top_valid.unsqueeze(1), float("-inf"))
        weights = F.softmax(torch.cat((global_scores, local_scores), dim=-1).float(), dim=-1).to(x.dtype)
        out = (torch.einsum("bhtk,btkd->bhtd", weights[..., :k_top], vb)
               + torch.einsum("bhtw,btwd->bhtd", weights[..., k_top:], v_local))
        return self.o_proj(out.transpose(1, 2).contiguous().view(b, t, h * hd))


class ASICDecoderLayer(nn.Module):
    def __init__(self, cfg: ASICLMConfig, layer_idx: int):
        super().__init__()
        self.is_global = cfg.is_global_layer(layer_idx)
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.mixer: nn.Module = SparseGlobalMixer(cfg) if self.is_global else DeltaNetLikeMixer(cfg)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.layer_intermediate_size(layer_idx))

    def forward(self, x: torch.Tensor, recurrent_state: Optional[DeltaState] = None) -> tuple[torch.Tensor, Optional[DeltaState]]:
        h = self.norm1(x)
        if self.is_global:
            mixed, next_state = self.mixer(h), None
        else:
            mixed, next_state = self.mixer(h, recurrent_state)
        x = x + mixed
        return x + self.mlp(self.norm2(x)), next_state


class ASICLM(nn.Module):
    def __init__(self, cfg: ASICLMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(ASICDecoderLayer(cfg, i) for i in range(cfg.num_layers))
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=self.cfg.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, *, recurrent_states: Optional[Sequence[Optional[DeltaState]]] = None,
                return_hidden_states: bool = False) -> dict[str, object]:
        x = self.embed(input_ids)
        states = list(recurrent_states) if recurrent_states is not None else [None] * self.cfg.num_layers
        if len(states) != self.cfg.num_layers:
            raise ValueError("recurrent_states must have one entry per layer")
        next_states: list[Optional[DeltaState]] = []
        hidden_states: list[torch.Tensor] = []
        for layer, state in zip(self.layers, states):
            x, state = layer(x, state)
            next_states.append(state)
            if return_hidden_states:
                hidden_states.append(x)
        logits = self.lm_head(self.final_norm(x))
        return {"logits": logits, "recurrent_states": next_states,
                "hidden_states": hidden_states if return_hidden_states else None}


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor, *,
                      temperature: float = 2.0, ce_weight: float = 1.0, kd_weight: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    vocab = student_logits.shape[-1]
    ce = F.cross_entropy(student_logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=-100)
    student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_p = F.softmax(teacher_logits.float() / temperature, dim=-1)
    kd = F.kl_div(student_logp, teacher_p, reduction="batchmean") * temperature**2
    return ce_weight * ce + kd_weight * kd


def hidden_state_distillation_loss(student_states: Sequence[torch.Tensor], teacher_states: Sequence[torch.Tensor],
                                   layer_map: Sequence[tuple[int, int]]) -> torch.Tensor:
    losses = [F.mse_loss(F.normalize(student_states[s].float(), dim=-1),
                         F.normalize(teacher_states[t].float(), dim=-1)) for s, t in layer_map]
    if not losses:
        raise ValueError("layer_map must not be empty")
    return torch.stack(losses).mean()


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def tiny_smoke_test() -> None:
    torch.manual_seed(0)
    cfg = ASICLMConfig(vocab_size=512, hidden_size=128, num_layers=4, intermediate_size=256,
                      num_heads=4, head_dim=32, local_window=16, retrieval_block_size=4,
                      top_blocks=2, index_dim=32)
    model = ASICLM(cfg)
    logits = model(torch.randint(0, cfg.vocab_size, (2, 32)))["logits"]
    assert isinstance(logits, torch.Tensor) and logits.shape == (2, 32, cfg.vocab_size)
    print(f"smoke test OK; parameters={parameter_count(model):,}")


if __name__ == "__main__":
    tiny_smoke_test()
