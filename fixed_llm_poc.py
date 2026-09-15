"""Research PyTorch model for a hardware-shaped, fixed-weight LLM.

The decoder repeats three Gated DeltaNet recurrent layers followed by one
gated-attention global layer, matching the Qwen3.5 dense small models
(9B: hidden 4096, 4B: hidden 2560).  Parameter names and shapes follow the
Hugging Face ``Qwen3_5`` text model so that released checkpoints load with a
prefix rename (see ``training/qwen35_import.py``).  The only additions are the
per-global-layer retrieval index projections (``index_q``/``index_k``) and the
local-window plus top-K block retrieval that replaces full attention.

This is a correctness-oriented reference implementation.  The recurrent scan is
a token loop and the retrieval scan materializes token-by-block scores; neither
is a training kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ASICLMConfig:
    """Model geometry and runtime configuration (defaults: Qwen3.5-9B)."""

    vocab_size: int = 248_320
    hidden_size: int = 4096
    num_layers: int = 32
    intermediate_size: int = 12_288
    recurrent_intermediate_size: Optional[int] = None
    global_intermediate_size: Optional[int] = None
    # Gated DeltaNet recurrent mixer.
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel: int = 4
    # Gated attention global mixer.
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10_000_000.0
    # Layer pattern and retrieval geometry.
    recurrent_every: int = 4
    global_layer_offset: int = 3
    local_window: int = 512
    retrieval_block_size: int = 16
    top_blocks: int = 32
    index_dim: int = 128
    index_bits: int = 4
    rms_eps: float = 1e-6
    tie_embeddings: bool = False
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.recurrent_every <= 0:
            raise ValueError("recurrent_every must be positive")
        if self.num_layers <= 0 or self.num_layers % self.recurrent_every:
            raise ValueError("num_layers must be a positive multiple of recurrent_every")
        if not 0 <= self.global_layer_offset < self.recurrent_every:
            raise ValueError("global_layer_offset must select a layer in the pattern")
        positive = ("vocab_size", "hidden_size", "linear_num_key_heads", "linear_num_value_heads",
                    "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel",
                    "num_attention_heads", "num_key_value_heads", "head_dim", "local_window",
                    "retrieval_block_size", "top_blocks", "index_dim", "index_bits")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.index_bits > 8:
            raise ValueError("index_bits must be at most 8")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError("linear_num_value_heads must be a multiple of linear_num_key_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if not 0 < self.partial_rotary_factor <= 1 or self.rotary_dim % 2:
            raise ValueError("partial_rotary_factor must give an even, positive rotary width")
        for name in ("intermediate_size", "recurrent_intermediate_size", "global_intermediate_size"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when specified")

    # Presets -------------------------------------------------------------

    @classmethod
    def qwen3_5_9b(cls, **overrides) -> "ASICLMConfig":
        """Qwen3.5-9B text geometry: 8.95B parameters, 6.9B in the layer body."""
        return cls(**overrides)

    @classmethod
    def qwen3_5_4b(cls, **overrides) -> "ASICLMConfig":
        """Qwen3.5-4B text geometry: same heads and state formats, hidden 2560."""
        values = dict(hidden_size=2560, intermediate_size=9216, tie_embeddings=True)
        values.update(overrides)
        return cls(**values)

    @classmethod
    def qwen3_5_27b(cls, **overrides) -> "ASICLMConfig":
        """27B-class dense hybrid geometry (Qwen3.5-27B: hidden 5120, 64 layers,
        FFN 17408, 24 query heads on the same 4 KV heads of 256).  The high-end
        stand-in until a released 27B checkpoint fixes the exact shapes."""
        values = dict(hidden_size=5120, num_layers=64, intermediate_size=17_408, num_attention_heads=24)
        values.update(overrides)
        return cls(**values)

    PRESETS = ("qwen3_5_9b", "qwen3_5_4b", "qwen3_5_27b")

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "ASICLMConfig":
        if name not in cls.PRESETS:
            raise ValueError(f"unknown geometry preset {name!r}; choose from {cls.PRESETS}")
        return getattr(cls, name)(**overrides)

    # Derived geometry ------------------------------------------------------

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    def is_global_layer(self, layer_idx: int) -> bool:
        return layer_idx % self.recurrent_every == self.global_layer_offset

    def layer_intermediate_size(self, layer_idx: int) -> int:
        """Return the FFN width, allowing weights to move toward R stages."""
        override = (self.global_intermediate_size if self.is_global_layer(layer_idx)
                    else self.recurrent_intermediate_size)
        return self.intermediate_size if override is None else override


class RMSNorm(nn.Module):
    """Qwen3.5 RMSNorm: zero-centred weight, i.e. ``x_hat * (1 + weight)``."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * (1.0 + self.weight.float())).type_as(x)


class GatedRMSNorm(nn.Module):
    """Gated RMSNorm used after the delta-rule scan: ``weight * x_hat * silu(gate)``."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight.float() * normed * F.silu(gate.float())).type_as(x)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


@dataclass
class DeltaState:
    """Bounded recurrent state for one recurrent mixer.

    ``matrix`` is the per-head delta-rule state ``[batch, v_heads, k_dim, v_dim]``;
    ``conv`` holds the last ``kernel - 1`` pre-convolution inputs
    ``[batch, conv_channels, kernel - 1]``.
    """

    matrix: torch.Tensor
    conv: torch.Tensor

    def detach(self) -> "DeltaState":
        return DeltaState(self.matrix.detach(), self.conv.detach())


class GatedDeltaNetMixer(nn.Module):
    """Gated DeltaNet mixer with the Qwen3.5 parameterization.

    Parameter names mirror ``Qwen3_5GatedDeltaNet`` so checkpoints load directly.
    The scan is a reference token loop equivalent to the Hugging Face
    ``torch_recurrent_gated_delta_rule``.
    """

    def __init__(self, cfg: ASICLMConfig):
        super().__init__()
        d = cfg.hidden_size
        self.num_k_heads = cfg.linear_num_key_heads
        self.num_v_heads = cfg.linear_num_value_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.key_dim = cfg.linear_key_dim
        self.value_dim = cfg.linear_value_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.kernel = cfg.linear_conv_kernel
        self.in_proj_qkv = nn.Linear(d, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(d, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(d, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(d, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, self.kernel, groups=self.conv_dim,
                                bias=False, padding=self.kernel - 1)
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.norm = GatedRMSNorm(self.head_v_dim, cfg.rms_eps)
        self.out_proj = nn.Linear(self.value_dim, d, bias=False)

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> DeltaState:
        matrix = torch.zeros(batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim,
                             device=device, dtype=dtype)
        conv = torch.zeros(batch_size, self.conv_dim, self.kernel - 1, device=device, dtype=dtype)
        return DeltaState(matrix, conv)

    def forward(self, x: torch.Tensor, state: Optional[DeltaState] = None) -> tuple[torch.Tensor, DeltaState]:
        b, t, _ = x.shape
        state = state or self.initial_state(b, device=x.device, dtype=x.dtype)
        conv_in = torch.cat((state.conv, self.in_proj_qkv(x).transpose(1, 2)), dim=-1)
        next_conv = conv_in[..., -(self.kernel - 1):] if self.kernel > 1 else conv_in[..., :0]
        mixed = F.silu(F.conv1d(conv_in, self.conv1d.weight, groups=self.conv_dim)).transpose(1, 2)
        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = F.normalize(q.view(b, t, self.num_k_heads, self.head_k_dim).float(), dim=-1)
        k = F.normalize(k.view(b, t, self.num_k_heads, self.head_k_dim).float(), dim=-1)
        v = v.view(b, t, self.num_v_heads, self.head_v_dim).float()
        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)
        q = q * self.head_k_dim ** -0.5
        beta = torch.sigmoid(self.in_proj_b(x).float())
        decay = torch.exp(-self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias.float()))

        s = state.matrix.float()
        outs = []
        for i in range(t):
            s = s * decay[:, i, :, None, None]
            ki, vi = k[:, i], v[:, i]
            pred = torch.einsum("bhk,bhkv->bhv", ki, s)
            s = s + beta[:, i, :, None, None] * ki.unsqueeze(-1) * (vi - pred).unsqueeze(-2)
            outs.append(torch.einsum("bhk,bhkv->bhv", q[:, i], s))
        y = torch.stack(outs, dim=1).to(x.dtype)
        z = self.in_proj_z(x).view(b, t, self.num_v_heads, self.head_v_dim)
        y = self.norm(y, z).reshape(b, t, self.value_dim)
        return self.out_proj(y), DeltaState(s.to(x.dtype), next_conv)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class SparseGlobalMixer(nn.Module):
    """Gated GQA attention over a local window plus a FULL scan of compressed old blocks.

    Projections, norms, rotary width, and the output gate follow ``Qwen3_5Attention``
    so the released weights load unchanged.  Full causal attention is replaced by
    exact attention over the last ``local_window`` positions and over ``top_blocks``
    retrieved blocks of block-mean keys/values.  Every global layer independently
    builds and scans its own index; candidate lists are not shared between layers.
    """

    def __init__(self, cfg: ASICLMConfig):
        super().__init__()
        self.cfg = cfg
        d, nh, nkv, hd = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.num_heads, self.num_kv_heads, self.head_dim = nh, nkv, hd
        self.q_proj = nn.Linear(d, nh * hd * 2, bias=False)
        self.k_proj = nn.Linear(d, nkv * hd, bias=False)
        self.v_proj = nn.Linear(d, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nh * hd, d, bias=False)
        self.q_norm = RMSNorm(hd, cfg.rms_eps)
        self.k_norm = RMSNorm(hd, cfg.rms_eps)
        self.index_q = nn.Linear(d, cfg.index_dim, bias=False)
        self.index_k = nn.Linear(d, cfg.index_dim, bias=False)
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.rotary_dim, 2).float() / cfg.rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate the first ``rotary_dim`` dims of ``x`` [b, t, heads, hd]."""
        r = self.cfg.rotary_dim
        freqs = positions.float().unsqueeze(-1) * self.inv_freq  # [t, r/2]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().view(1, -1, 1, r), emb.sin().view(1, -1, 1, r)
        head, tail = x[..., :r].float(), x[..., r:]
        head = head * cos + rotate_half(head) * sin
        return torch.cat((head.to(x.dtype), tail), dim=-1)

    @staticmethod
    def _window(x: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        _, t, _ = x.shape
        padded = F.pad(x, (0, 0, width - 1, 0))
        windows = padded.unfold(1, width, 1).permute(0, 1, 3, 2).contiguous()
        pos = torch.arange(t, device=x.device)
        slot = torch.arange(width, device=x.device)
        valid = slot.unsqueeze(0) >= (width - 1 - pos).clamp_min(0).unsqueeze(1)
        return windows, valid

    def _block_mean(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        bs = self.cfg.retrieval_block_size
        nb = math.ceil(t / bs)
        pad = nb * bs - t
        xp = F.pad(x, (0, 0, 0, pad)).view(b, nb, bs, d)
        valid = F.pad(torch.ones(t, device=x.device, dtype=x.dtype), (0, pad)).view(nb, bs)
        counts = valid.sum(1).clamp_min(1)
        return (xp * valid.view(1, nb, bs, 1)).sum(2) / counts.view(1, nb, 1)

    def _fake_quantize_index(self, x: torch.Tensor) -> torch.Tensor:
        """Signed per-vector QAT with exactly 2**index_bits levels."""
        levels = 2 ** self.cfg.index_bits
        scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized = (x / scale).clamp(-1, 1)
        quantized = ((normalized + 1) * (levels - 1) / 2).round()
        quantized = (quantized * 2 / (levels - 1) - 1) * scale
        return x + (quantized - x).detach()

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        b, t, _ = x.shape
        nh, nkv, hd, bs = self.num_heads, self.num_kv_heads, self.head_dim, self.cfg.retrieval_block_size
        group = nh // nkv
        w = min(self.cfg.local_window, t)
        positions = torch.arange(position_offset, position_offset + t, device=x.device)

        q, gate = self.q_proj(x).view(b, t, nh, 2 * hd).chunk(2, dim=-1)
        q = self._rope(self.q_norm(q), positions) * hd ** -0.5
        k = self._rope(self.k_norm(self.k_proj(x).view(b, t, nkv, hd)), positions)
        v = self.v_proj(x).view(b, t, nkv, hd)
        qg = q.reshape(b, t, nkv, group, hd)

        k_local, local_valid = self._window(k.reshape(b, t, nkv * hd), w)
        v_local, _ = self._window(v.reshape(b, t, nkv * hd), w)
        k_local = k_local.view(b, t, w, nkv, hd)
        v_local = v_local.view(b, t, w, nkv, hd)
        local_scores = torch.einsum("btngd,btwnd->btngw", qg, k_local)
        local_scores = local_scores.masked_fill(~local_valid.view(1, t, 1, 1, w), float("-inf"))

        k_block = self._block_mean(k.reshape(b, t, nkv * hd)).view(b, -1, nkv, hd)
        v_block = self._block_mean(v.reshape(b, t, nkv * hd)).view(b, -1, nkv, hd)
        nb = k_block.shape[1]
        index_source = F.normalize(self._block_mean(self.index_k(x)).float(), dim=-1)
        index_query = F.normalize(self.index_q(x).float(), dim=-1)
        index_query = self._fake_quantize_index(index_query)
        index_source = self._fake_quantize_index(index_source)
        retrieval_scores = torch.einsum("bti,bni->btn", index_query, index_source)
        token_pos = torch.arange(t, device=x.device)
        block_end = (torch.arange(nb, device=x.device) + 1) * bs - 1
        eligible = block_end.view(1, nb) <= token_pos.view(t, 1) - w
        masked = retrieval_scores.masked_fill(~eligible.view(1, t, nb), float("-inf"))
        k_top = min(self.cfg.top_blocks, nb)
        top_idx = torch.topk(masked, k=k_top, dim=-1).indices  # [b, t, k]
        top_valid = torch.gather(eligible.view(1, t, nb).expand(b, -1, -1), 2, top_idx)
        batch_idx = torch.arange(b, device=x.device).view(b, 1, 1)
        kb = k_block[batch_idx, top_idx]  # [b, t, k, nkv, hd]
        vb = v_block[batch_idx, top_idx]
        global_scores = torch.einsum("btngd,btknd->btngk", qg, kb)
        global_scores = global_scores.masked_fill(~top_valid.view(b, t, 1, 1, k_top), float("-inf"))

        weights = F.softmax(torch.cat((global_scores, local_scores), dim=-1).float(), dim=-1).to(x.dtype)
        out = (torch.einsum("btngk,btknd->btngd", weights[..., :k_top], vb)
               + torch.einsum("btngw,btwnd->btngd", weights[..., k_top:], v_local))
        out = out.reshape(b, t, nh * hd) * torch.sigmoid(gate.reshape(b, t, nh * hd))
        return self.o_proj(out)


class ASICDecoderLayer(nn.Module):
    """One pipeline stage: pre-norm mixer, pre-norm SwiGLU, HF-compatible names."""

    def __init__(self, cfg: ASICLMConfig, layer_idx: int):
        super().__init__()
        self.is_global = cfg.is_global_layer(layer_idx)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        if self.is_global:
            self.self_attn = SparseGlobalMixer(cfg)
        else:
            self.linear_attn = GatedDeltaNetMixer(cfg)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.layer_intermediate_size(layer_idx))

    @property
    def mixer(self) -> nn.Module:
        return self.self_attn if self.is_global else self.linear_attn

    def forward(self, x: torch.Tensor, recurrent_state: Optional[DeltaState] = None,
                position_offset: int = 0) -> tuple[torch.Tensor, Optional[DeltaState]]:
        h = self.input_layernorm(x)
        if self.is_global:
            mixed, next_state = self.self_attn(h, position_offset), None
        else:
            mixed, next_state = self.linear_attn(h, recurrent_state)
        x = x + mixed
        return x + self.mlp(self.post_attention_layernorm(x)), next_state


class ASICLM(nn.Module):
    def __init__(self, cfg: ASICLMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(ASICDecoderLayer(cfg, i) for i in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=self.cfg.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, *, recurrent_states: Optional[Sequence[Optional[DeltaState]]] = None,
                position_offset: int = 0, return_hidden_states: bool = False) -> dict[str, object]:
        x = self.embed_tokens(input_ids)
        states = list(recurrent_states) if recurrent_states is not None else [None] * self.cfg.num_layers
        if len(states) != self.cfg.num_layers:
            raise ValueError("recurrent_states must have one entry per layer")
        next_states: list[Optional[DeltaState]] = []
        hidden_states: list[torch.Tensor] = []
        for layer, state in zip(self.layers, states):
            x, state = layer(x, state, position_offset)
            next_states.append(state)
            if return_hidden_states:
                hidden_states.append(x)
        logits = self.lm_head(self.norm(x))
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


@dataclass(frozen=True)
class GeometryReport:
    """Analytic parameter budget for one configuration (no weights allocated)."""

    recurrent_layer: int
    global_layer: int
    per_shard: int
    body: int
    embedding: int
    total: int
    per_layer: tuple[int, ...] = field(repr=False)


def geometry_report(cfg: ASICLMConfig) -> GeometryReport:
    """Count parameters per layer by building layers on the meta device."""
    with torch.device("meta"):
        per_layer = tuple(parameter_count(ASICDecoderLayer(cfg, i)) for i in range(cfg.num_layers))
    shard = cfg.recurrent_every
    per_shard = sum(per_layer[:shard])
    body = sum(per_layer)
    embedding = cfg.vocab_size * cfg.hidden_size * (1 if cfg.tie_embeddings else 2)
    recurrent = next(p for i, p in enumerate(per_layer) if not cfg.is_global_layer(i))
    global_ = next(p for i, p in enumerate(per_layer) if cfg.is_global_layer(i))
    return GeometryReport(recurrent, global_, per_shard, body, embedding, body + embedding, per_layer)


def tiny_config(**overrides) -> ASICLMConfig:
    """A small geometry with the same structure, for tests and smoke runs."""
    values = dict(vocab_size=512, hidden_size=96, num_layers=4, intermediate_size=192,
                  linear_num_key_heads=2, linear_num_value_heads=4,
                  linear_key_head_dim=16, linear_value_head_dim=16,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=24,
                  local_window=16, retrieval_block_size=4, top_blocks=2, index_dim=32)
    values.update(overrides)
    return ASICLMConfig(**values)


def tiny_smoke_test() -> None:
    torch.manual_seed(0)
    cfg = tiny_config()
    model = ASICLM(cfg)
    logits = model(torch.randint(0, cfg.vocab_size, (2, 32)))["logits"]
    assert isinstance(logits, torch.Tensor) and logits.shape == (2, 32, cfg.vocab_size)
    print(f"smoke test OK; parameters={parameter_count(model):,}")
    for name in ASICLMConfig.PRESETS:
        report = geometry_report(ASICLMConfig.from_preset(name))
        print(f"{name}: R layer {report.recurrent_layer / 1e6:.1f}M, G layer {report.global_layer / 1e6:.1f}M, "
              f"shard {report.per_shard / 1e6:.0f}M, body {report.body / 1e9:.2f}B, "
              f"embedding {report.embedding / 1e9:.2f}B, total {report.total / 1e9:.2f}B")


if __name__ == "__main__":
    tiny_smoke_test()
