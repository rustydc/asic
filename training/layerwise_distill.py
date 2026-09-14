"""Train one ASIC decoder layer against cached teacher activations.

Cache files are torch files containing ``input_hidden`` and ``teacher_output``
tensors shaped [examples, sequence, hidden]. Keeping cache generation separate
lets this harness work with any teacher implementation or checkpoint license.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from fixed_llm_poc import ASICDecoderLayer, ASICLMConfig


def load_variant(path: Path, name: str) -> ASICLMConfig:
    """Resolve a variant: a geometry preset plus retrieval overrides."""
    variants = json.loads(path.read_text(encoding="utf-8"))
    if name not in variants:
        raise ValueError(f"unknown variant {name!r}; choose from {sorted(variants)}")
    values = dict(variants[name])
    geometry = values.pop("geometry", "qwen3_5_9b")
    return ASICLMConfig.from_preset(geometry, **values)


def activation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    normalized = F.mse_loss(
        F.normalize(student.float(), dim=-1),
        F.normalize(teacher.float(), dim=-1),
    )
    magnitude = F.smooth_l1_loss(student.float(), teacher.float())
    return normalized + 0.1 * magnitude


def iter_batches(cache_files: list[Path], batch_size: int, *, shuffle: bool):
    files = list(cache_files)
    if shuffle:
        random.shuffle(files)
    for path in files:
        cache = torch.load(path, map_location="cpu", weights_only=True)
        inputs = cache["input_hidden"]
        targets = cache["teacher_output"]
        if inputs.shape != targets.shape or inputs.ndim != 3:
            raise ValueError(f"invalid activation cache shapes in {path}")
        order = torch.randperm(inputs.shape[0]) if shuffle else torch.arange(inputs.shape[0])
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            yield inputs[indices], targets[indices]


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    cfg = load_variant(args.variants, args.variant)
    if not 0 <= args.layer_index < cfg.num_layers:
        raise ValueError(f"layer-index must be between 0 and {cfg.num_layers - 1}")
    layer = ASICDecoderLayer(cfg, args.layer_index).to(args.device)
    if args.initial_layer:
        layer.load_state_dict(torch.load(args.initial_layer, map_location="cpu", weights_only=True))
    layer.train()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    cache_files = sorted(args.cache.glob("*.pt"))
    if not cache_files:
        raise ValueError(f"no .pt activation caches found below {args.cache}")

    for epoch in range(args.epochs):
        total_loss = 0.0
        steps = 0
        for inputs, targets in iter_batches(cache_files, args.batch_size, shuffle=True):
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            outputs, _ = layer(inputs)
            loss = activation_loss(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(layer.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += loss.detach().item()
            steps += 1
        print(f"epoch={epoch + 1} loss={total_loss / steps:.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(layer.state_dict(), args.output)
    metadata = {
        "variant": args.variant,
        "layer_index": args.layer_index,
        "config": asdict(cfg),
        "cache_files": len(cache_files),
        "epochs": args.epochs,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--variants", type=Path, default=Path("training/variants.json"))
    parser.add_argument("--layer-index", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--initial-layer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
