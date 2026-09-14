"""Import Qwen3.5 dense text weights into the ASIC reference model.

The reference model uses the Hugging Face ``Qwen3_5`` parameter names, so the
conversion is a prefix rename plus initialization of the two retrieval index
projections that the appliance adds to every global layer.  The vision tower is
dropped.

Example:

    python -m training.qwen35_import \
        --checkpoint /models/Qwen3.5-9B-Base \
        --geometry qwen3_5_9b \
        --output checkpoints/asic_init.pt

Only the checkpoint's text weights are read.  Loading requires the ``safetensors``
package when the checkpoint is stored as ``.safetensors`` shards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

import torch

from fixed_llm_poc import ASICDecoderLayer, ASICLM, ASICLMConfig

HF_TEXT_PREFIX = "model.language_model."
HF_LEGACY_PREFIX = "model."
DROPPED_PREFIXES = ("model.visual.", "visual.")
NEW_PARAMETER_SUFFIXES = ("self_attn.index_q.weight", "self_attn.index_k.weight")


def rename_hf_key(key: str) -> str | None:
    """Map a Hugging Face key to the reference model, or ``None`` to drop it."""
    if key.startswith(DROPPED_PREFIXES):
        return None
    if key == "lm_head.weight":
        return key
    if key.startswith(HF_TEXT_PREFIX):
        return key[len(HF_TEXT_PREFIX):]
    if key.startswith(HF_LEGACY_PREFIX):
        return key[len(HF_LEGACY_PREFIX):]
    return None


def convert_state_dict(hf_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = {}
    for key, value in hf_state.items():
        renamed = rename_hf_key(key)
        if renamed is not None:
            converted[renamed] = value
    return converted


def config_from_hf(hf_config: Mapping[str, object], **overrides) -> ASICLMConfig:
    """Build a config from a Hugging Face ``config.json`` (text sub-config aware)."""
    text = hf_config.get("text_config", hf_config)
    rope = text.get("rope_parameters", text.get("rope_scaling", {})) or {}
    values = dict(
        vocab_size=text["vocab_size"],
        hidden_size=text["hidden_size"],
        num_layers=text["num_hidden_layers"],
        intermediate_size=text["intermediate_size"],
        linear_num_key_heads=text["linear_num_key_heads"],
        linear_num_value_heads=text["linear_num_value_heads"],
        linear_key_head_dim=text["linear_key_head_dim"],
        linear_value_head_dim=text["linear_value_head_dim"],
        linear_conv_kernel=text["linear_conv_kernel_dim"],
        num_attention_heads=text["num_attention_heads"],
        num_key_value_heads=text["num_key_value_heads"],
        head_dim=text["head_dim"],
        partial_rotary_factor=rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 1.0)),
        rope_theta=rope.get("rope_theta", text.get("rope_theta", 10_000.0)),
        rms_eps=text.get("rms_norm_eps", 1e-6),
        tie_embeddings=hf_config.get("tie_word_embeddings", text.get("tie_word_embeddings", False)),
    )
    layer_types = text.get("layer_types")
    if layer_types:
        globals_ = [i for i, kind in enumerate(layer_types) if kind == "full_attention"]
        period = globals_[1] - globals_[0] if len(globals_) > 1 else len(layer_types)
        expected = list(range(globals_[0], len(layer_types), period))
        if globals_ != expected:
            raise ValueError("layer_types is not a fixed-period pattern; the appliance needs one")
        values.update(recurrent_every=period, global_layer_offset=globals_[0])
    values.update(overrides)
    return ASICLMConfig(**values)


@torch.no_grad()
def init_index_projections(layer: ASICDecoderLayer) -> None:
    """Warm-start the retrieval index from the attention projections.

    The index score ``index_q(x_q) . index_k(x_k)`` should approximate the mean
    attention logit over the query heads sharing each KV head.  With
    ``W_k = U S V^T``, projecting both sides through the top ``index_dim`` left
    singular vectors of ``W_k`` keeps the dominant key directions, so
    ``(U_r^T W_q_bar x_q) . (U_r^T W_k x_k) ~= (W_q_bar x_q) . (W_k x_k)``.
    Norms, rotary rotation, and quantization are ignored; training closes the gap.
    """
    if not layer.is_global:
        raise ValueError("index projections exist only on global layers")
    attn = layer.self_attn
    nh, nkv, hd = attn.num_heads, attn.num_kv_heads, attn.head_dim
    group = nh // nkv
    q_weight = attn.q_proj.weight.float().view(nh, 2 * hd, -1)[:, :hd, :]  # drop the gate half
    q_mean = q_weight.view(nkv, group, hd, -1).mean(dim=1).reshape(nkv * hd, -1)
    k_weight = attn.k_proj.weight.float()  # [nkv * hd, hidden]
    rank = attn.index_k.out_features
    u, _, _ = torch.linalg.svd(k_weight, full_matrices=False)
    basis = u[:, :rank].T  # [rank, nkv * hd]
    attn.index_k.weight.copy_((basis @ k_weight).to(attn.index_k.weight.dtype))
    attn.index_q.weight.copy_((basis @ q_mean).to(attn.index_q.weight.dtype))


def iter_checkpoint_tensors(checkpoint: Path) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield (name, tensor) from a directory of safetensors shards or a torch file."""
    if checkpoint.is_dir():
        shards = sorted(checkpoint.glob("*.safetensors"))
        if shards:
            from safetensors.torch import load_file  # optional dependency
            for shard in shards:
                yield from load_file(str(shard)).items()
            return
        binaries = sorted(checkpoint.glob("*.bin")) + sorted(checkpoint.glob("*.pt"))
        for binary in binaries:
            yield from torch.load(binary, map_location="cpu", weights_only=True).items()
        return
    yield from torch.load(checkpoint, map_location="cpu", weights_only=True).items()


def load_into_model(model: ASICLM, hf_state: Mapping[str, torch.Tensor], *, strict: bool = True) -> list[str]:
    """Load converted weights, initialize index projections, and return the new keys."""
    converted = convert_state_dict(hf_state)
    if model.cfg.tie_embeddings:
        converted.pop("lm_head.weight", None)
    result = model.load_state_dict(converted, strict=False)
    unexpected = list(result.unexpected_keys)
    missing = [key for key in result.missing_keys if not key.endswith(NEW_PARAMETER_SUFFIXES)]
    if strict and (unexpected or missing):
        raise KeyError(f"checkpoint mismatch; missing={missing[:8]} unexpected={unexpected[:8]}")
    for layer in model.layers:
        if layer.is_global:
            init_index_projections(layer)
    return [key for key in result.missing_keys if key.endswith(NEW_PARAMETER_SUFFIXES)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Hugging Face model directory (safetensors) or a torch state-dict file")
    parser.add_argument("--geometry", default=None, choices=ASICLMConfig.PRESETS,
                        help="preset to use instead of reading config.json from the checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16", "float16"))
    args = parser.parse_args()

    if args.geometry:
        cfg = ASICLMConfig.from_preset(args.geometry)
    else:
        cfg = config_from_hf(json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8")))
    with torch.device("meta"):
        model = ASICLM(cfg)
    model = model.to_empty(device="cpu").to(getattr(torch, args.dtype))
    state = dict(iter_checkpoint_tensors(args.checkpoint))
    new_keys = load_into_model(model, state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"wrote {args.output}; initialized {len(new_keys)} index projections from attention weights")


if __name__ == "__main__":
    main()
