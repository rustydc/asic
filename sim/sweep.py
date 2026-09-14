"""Sweep retrieval geometry and memory bandwidth without long simulations."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import deque
from dataclasses import replace

from sim.appliance import ApplianceConfig, Simulation, Stage, WorkItem


def comma_numbers(value: str, kind: type = int) -> list:
    return [kind(part) for part in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="sim/config/baseline.json")
    parser.add_argument("--index-dims", default="32,64,96,128")
    parser.add_argument("--index-bits", default="2,4,8")
    parser.add_argument("--compression-ratios", default="4,8,16")
    parser.add_argument("--bandwidth-gbps", default="50,75,100")
    parser.add_argument("--output", help="CSV path; defaults to stdout")
    args = parser.parse_args()

    base = ApplianceConfig.from_json(args.config)
    output = open(args.output, "w", newline="", encoding="utf-8") if args.output else sys.stdout
    fields = [
        "index_dim", "index_bits", "compression_ratio", "bandwidth_gbps",
        "bytes_per_token_per_asic", "global_latency_us", "global_interval_us",
        "global_ceiling_tokens_per_second",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    try:
        combinations = itertools.product(
            comma_numbers(args.index_dims),
            comma_numbers(args.index_bits),
            comma_numbers(args.compression_ratios),
            comma_numbers(args.bandwidth_gbps, float),
        )
        for index_dim, index_bits, compression_ratio, bandwidth_gbps in combinations:
            raw_bytes_per_cycle = bandwidth_gbps * 1_000 / base.clock_mhz
            config = replace(
                base,
                index_dim=index_dim,
                index_bits=index_bits,
                compression_ratio=compression_ratio,
                memory_bytes_per_cycle=raw_bytes_per_cycle,
            )
            simulation = Simulation(config)
            item = WorkItem(0, config.initial_context_tokens, 0, 0)
            stage = Stage(config.layers_per_asic - 1, True, deque())
            latency, interval, memory_bytes = simulation._timing(stage, item)
            writer.writerow({
                "index_dim": index_dim,
                "index_bits": index_bits,
                "compression_ratio": compression_ratio,
                "bandwidth_gbps": bandwidth_gbps,
                "bytes_per_token_per_asic": memory_bytes,
                "global_latency_us": round(latency / config.clock_mhz, 3),
                "global_interval_us": round(interval / config.clock_mhz, 3),
                "global_ceiling_tokens_per_second": round(
                    config.clock_mhz * 1_000_000 / interval, 1
                ),
            })
    finally:
        if args.output:
            output.close()


if __name__ == "__main__":
    main()
