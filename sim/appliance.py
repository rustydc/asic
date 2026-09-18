"""Transaction-level, cycle-stepped model of the 32-stage appliance.

This simulator deliberately moves tags rather than tensors.  It models the
resources that determine throughput: finite queues, stage service intervals,
chip-boundary links, global-layer memory traffic, sampling latency, and the
one-unresolved-token-per-context autoregressive dependency.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ApplianceConfig:
    clock_mhz: float = 500.0
    num_asics: int = 8
    layers_per_asic: int = 4
    global_every: int = 0                    # a global layer every this many layers; 0 means one per ASIC (its last layer)
    resident_contexts: int = 32
    tokens_per_context: int = 16
    initial_context_tokens: int = 131_072
    warmup_tokens_per_context: int = 1
    recurrent_cycles: int = 3_500
    recurrent_weight_scale: float = 1.0
    global_index_compute_cycles: int = 10_000
    global_topk_cycles: int = 500
    global_attention_cycles: int = 2_500
    global_output_cycles: int = 1_000
    global_weight_scale: float = 1.0
    global_max_inflight: int = 4
    # Bounded recurrent state of ONE recurrent layer, in bytes (Gated DeltaNet:
    # value heads x key dim x value dim).  A token's next token for the same
    # context is a whole ring behind it, so the state cannot stay on chip
    # between tokens: every token reads it and writes it back.  Zero keeps the
    # old behaviour, where recurrent stages moved no memory traffic at all.
    recurrent_state_bytes: int = 0
    fifo_depth: int = 2
    sampling_cycles: int = 500
    activation_bytes: int = 4_096
    packet_overhead_bytes: int = 16
    link_bytes_per_cycle: float = 4.0
    memory_bytes_per_cycle: float = 150.0
    memory_efficiency: float = 1.0
    compression_ratio: int = 4
    retrieval_block_size: int = 16
    top_blocks: int = 32
    local_window: int = 512
    index_dim: int = 128
    index_bits: int = 4
    kv_element_bytes: float = 1.0
    head_dim: int = 128
    kv_heads: int = 1
    num_head_asics: int = 0
    head_cycles: int = 40
    head_result_bytes: int = 768
    # Energy model.  mac_energy_pj is the switching plus internal energy of one
    # fixed-weight multiply-accumulate in the column datapath: 0.39 pJ measured
    # on ASAP7 at default activity (fabric/results/pnr_asap7_signoff.json), about
    # 1 pJ derated for real activity, 2 to 4 pJ projected to a 28 nm-class node.
    mac_energy_pj: float = 3.0
    layer_macs_per_token: float = 866e6      # per layer ASIC (9B: 4 layers of ~216M coefficients)
    head_macs_per_token: float = 508e6       # per head ASIC (half of the 248320 x 4096 head)
    memory_energy_pj_per_byte: float = 40.0  # LPDDR5X plus PHY, ~5 pJ/bit end to end
    static_power_w: float = 70.0             # FPGA, DRAM idle, ASIC leakage/IO, housekeeping
    max_cycles: int = 2_000_000_000
    trace: bool = False

    @property
    def global_period(self) -> int:
        return self.global_every or self.layers_per_asic

    @property
    def num_layers(self) -> int:
        return self.num_asics * self.layers_per_asic

    @property
    def num_stages(self) -> int:
        """Layer stages plus one stage per head-mode ASIC."""
        return self.num_layers + self.num_head_asics

    @property
    def num_chips(self) -> int:
        return self.num_asics + self.num_head_asics

    def chip_of_stage(self, stage_id: int) -> int:
        if stage_id < self.num_layers:
            return stage_id // self.layers_per_asic
        return self.num_asics + (stage_id - self.num_layers)

    def validate(self) -> None:
        positive = {
            "clock_mhz": self.clock_mhz,
            "num_asics": self.num_asics,
            "layers_per_asic": self.layers_per_asic,
            "resident_contexts": self.resident_contexts,
            "tokens_per_context": self.tokens_per_context,
            "recurrent_cycles": self.recurrent_cycles,
            "recurrent_weight_scale": self.recurrent_weight_scale,
            "global_index_compute_cycles": self.global_index_compute_cycles,
            "global_topk_cycles": self.global_topk_cycles,
            "global_attention_cycles": self.global_attention_cycles,
            "global_output_cycles": self.global_output_cycles,
            "global_weight_scale": self.global_weight_scale,
            "global_max_inflight": self.global_max_inflight,
            "fifo_depth": self.fifo_depth,
            "activation_bytes": self.activation_bytes,
            "link_bytes_per_cycle": self.link_bytes_per_cycle,
            "memory_bytes_per_cycle": self.memory_bytes_per_cycle,
            "memory_efficiency": self.memory_efficiency,
            "compression_ratio": self.compression_ratio,
            "retrieval_block_size": self.retrieval_block_size,
            "top_blocks": self.top_blocks,
            "local_window": self.local_window,
            "index_dim": self.index_dim,
            "index_bits": self.index_bits,
            "kv_element_bytes": self.kv_element_bytes,
            "head_dim": self.head_dim,
            "kv_heads": self.kv_heads,
            "head_cycles": self.head_cycles,
            "max_cycles": self.max_cycles,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"configuration fields must be positive: {', '.join(invalid)}")
        if self.recurrent_state_bytes < 0:
            raise ValueError("recurrent_state_bytes cannot be negative")
        if (self.sampling_cycles < 0 or self.packet_overhead_bytes < 0
                or self.warmup_tokens_per_context < 0 or self.num_head_asics < 0
                or self.head_result_bytes < 0):
            raise ValueError("sampling, overhead, warm-up, and head values cannot be negative")
        if min(self.mac_energy_pj, self.layer_macs_per_token, self.head_macs_per_token,
               self.memory_energy_pj_per_byte, self.static_power_w) < 0:
            raise ValueError("energy model values cannot be negative")
        if self.memory_efficiency > 1:
            raise ValueError("memory_efficiency cannot exceed one")
        if self.global_every < 0 or self.layers_per_asic % self.global_period:
            raise ValueError("global_every must divide layers_per_asic")

    @classmethod
    def from_json(cls, path: str | Path) -> "ApplianceConfig":
        with Path(path).open(encoding="utf-8") as source:
            values = json.load(source)
        values = {key: value for key, value in values.items() if not key.startswith("_")}
        return cls(**values)


@dataclass(frozen=True)
class WorkItem:
    context_id: int
    position: int
    sequence: int
    injected_cycle: int


@dataclass
class Context:
    context_id: int
    next_position: int
    generated: int = 0
    in_flight: bool = False
    ready_cycle: int = 0


@dataclass
class Counters:
    accepted: int = 0
    completed: int = 0
    busy_cycles: int = 0
    input_starved_cycles: int = 0
    output_stalled_cycles: int = 0
    memory_bytes: int = 0


@dataclass
class Stage:
    stage_id: int
    is_global: bool
    queue: deque[WorkItem]
    is_head: bool = False
    active: list["InFlight"] = field(default_factory=list)
    finished: deque[WorkItem] = field(default_factory=deque)
    last_start_cycle: int = -1
    counters: Counters = field(default_factory=Counters)


@dataclass
class Transit:
    item: WorkItem
    destination: int
    remaining: int


@dataclass
class InFlight:
    item: WorkItem
    remaining: int


@dataclass
class TraceEvent:
    name: str
    category: str
    start_cycle: int
    duration_cycles: int
    resource: int
    context_id: int
    sequence: int

    def chrome_event(self, clock_mhz: float) -> dict[str, Any]:
        microseconds_per_cycle = 1.0 / clock_mhz
        return {
            "name": self.name,
            "cat": self.category,
            "ph": "X",
            "ts": self.start_cycle * microseconds_per_cycle,
            "dur": self.duration_cycles * microseconds_per_cycle,
            "pid": 1,
            "tid": f"{self.category}-{self.resource}",
            "args": {"context": self.context_id, "sequence": self.sequence},
        }


@dataclass(frozen=True)
class SimulationResult:
    cycles: int
    measurement_cycles: int
    completed_tokens: int
    aggregate_tokens_per_second: float
    mean_token_latency_cycles: float
    max_token_latency_cycles: int
    stage_utilization: tuple[float, ...]
    stage_output_stalls: tuple[int, ...]
    fifo_high_watermarks: tuple[int, ...]
    link_utilization: tuple[float, ...]
    memory_bytes_per_asic: tuple[int, ...]
    # Energy model at the measured throughput (see ApplianceConfig.mac_energy_pj).
    compute_energy_per_token_mj: float
    memory_energy_per_token_mj: float
    compute_power_w: float          # all layer and head ASICs
    memory_power_w: float           # LPDDR traffic of the global stages
    board_power_w: float            # compute + memory + static_power_w
    layer_asic_power_w: float       # dynamic power of one layer ASIC

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def energy_per_token_mj(config: ApplianceConfig) -> float:
    """Compute energy for one token through every layer and head ASIC, in millijoules."""
    macs = config.num_asics * config.layer_macs_per_token + config.num_head_asics * config.head_macs_per_token
    return macs * config.mac_energy_pj * 1e-12 * 1e3


def board_power_w(config: ApplianceConfig, tokens_per_second: float, memory_bytes_per_token: float = 0.0) -> float:
    """Board power at a throughput: compute plus memory traffic plus the static floor."""
    compute = energy_per_token_mj(config) * 1e-3 * tokens_per_second
    memory = memory_bytes_per_token * config.memory_energy_pj_per_byte * 1e-12 * tokens_per_second
    return compute + memory + config.static_power_w


class Simulation:
    """Deterministic appliance simulation with round-robin context injection."""

    def __init__(self, config: ApplianceConfig):
        config.validate()
        self.config = config
        self.cycle = 0
        self.stages = [
            Stage(i, (i + 1) % config.global_period == 0, deque())
            for i in range(config.num_layers)
        ] + [
            Stage(config.num_layers + i, False, deque(), is_head=True)
            for i in range(config.num_head_asics)
        ]
        self.contexts = [Context(i, config.initial_context_tokens) for i in range(config.resident_contexts)]
        self.links: list[Transit | None] = [None] * (config.num_chips - 1)
        self.link_busy_cycles = [0] * (config.num_chips - 1)
        self.fifo_high_watermarks = [0] * config.num_stages
        self.memory_bytes_per_asic = [0] * config.num_asics
        # One memory interface per ASIC: every stage on a die shares it, so a
        # transfer reserves it and the other stages of that die wait.
        self.memory_busy_until = [0] * config.num_asics
        self.pending_sampling: list[tuple[int, WorkItem]] = []
        self.completed_latencies: list[int] = []
        self.measurement_start_cycle: int | None = None
        self.trace_events: list[TraceEvent] = []
        self.next_context = 0

    @property
    def kv_bytes_per_position(self) -> float:
        """Stored key plus value bytes for one position across all KV heads."""
        cfg = self.config
        return 2 * cfg.kv_heads * cfg.head_dim * cfg.kv_element_bytes

    def _global_memory_components(self, item: WorkItem) -> tuple[int, int, int]:
        cfg = self.config
        compressed_positions = math.ceil(item.position / cfg.compression_ratio)
        index_bytes = math.ceil(compressed_positions * cfg.index_dim * cfg.index_bits / 8)
        selected_positions = cfg.top_blocks * cfg.retrieval_block_size
        kv_positions = cfg.local_window + selected_positions
        kv_bytes = math.ceil(kv_positions * self.kv_bytes_per_position)
        append_bytes = math.ceil(cfg.index_dim * cfg.index_bits / 8 + self.kv_bytes_per_position)
        return index_bytes, kv_bytes, append_bytes

    def _timing(self, stage: Stage, item: WorkItem) -> tuple[int, int, int]:
        if stage.is_head:
            return self.config.head_cycles, self.config.head_cycles, 0
        if not stage.is_global:
            cfg = self.config
            compute = math.ceil(cfg.recurrent_cycles * cfg.recurrent_weight_scale)
            if not cfg.recurrent_state_bytes:
                return compute, compute, 0
            # Read the state, run the delta rule, write it back.  The memory is
            # the shared resource, so it sets the initiation interval.
            bandwidth = cfg.memory_bytes_per_cycle * cfg.memory_efficiency
            transfer = math.ceil(cfg.recurrent_state_bytes / bandwidth)
            latency = transfer + compute + transfer
            return latency, max(compute, 2 * transfer), 2 * cfg.recurrent_state_bytes
        cfg = self.config
        index_bytes, kv_bytes, append_bytes = self._global_memory_components(item)
        bandwidth = cfg.memory_bytes_per_cycle * cfg.memory_efficiency
        index_memory_cycles = math.ceil(index_bytes / bandwidth)
        kv_memory_cycles = math.ceil((kv_bytes + append_bytes) / bandwidth)
        index_phase = max(math.ceil(cfg.global_index_compute_cycles * cfg.global_weight_scale),
                          index_memory_cycles)
        attention_phase = math.ceil(cfg.global_attention_cycles * cfg.global_weight_scale)
        output_phase = math.ceil(cfg.global_output_cycles * cfg.global_weight_scale)
        latency = index_phase + cfg.global_topk_cycles + kv_memory_cycles + attention_phase + output_phase
        # Index and KV transfers share one memory interface. Other phase engines
        # are independent and may process different contexts concurrently.
        initiation_interval = max(
            index_memory_cycles + kv_memory_cycles,
            math.ceil(cfg.global_index_compute_cycles * cfg.global_weight_scale),
            cfg.global_topk_cycles,
            attention_phase,
            output_phase,
        )
        return latency, initiation_interval, index_bytes + kv_bytes + append_bytes

    def _link_index_after(self, stage_id: int) -> int | None:
        """Chip-boundary link index following ``stage_id``, if any (the last chip's link to the FPGA is not modeled)."""
        if stage_id == self.config.num_stages - 1:
            return None
        chip = self.config.chip_of_stage(stage_id)
        if self.config.chip_of_stage(stage_id + 1) == chip:
            return None
        return chip

    def _advance_sampling(self) -> None:
        still_pending: list[tuple[int, WorkItem]] = []
        for ready_cycle, item in self.pending_sampling:
            if ready_cycle > self.cycle:
                still_pending.append((ready_cycle, item))
                continue
            context = self.contexts[item.context_id]
            if not context.in_flight or item.sequence != context.generated:
                raise RuntimeError("autoregressive context ownership invariant violated")
            context.generated += 1
            context.next_position += 1
            context.in_flight = False
            context.ready_cycle = self.cycle
            if item.sequence >= self.config.warmup_tokens_per_context:
                self.completed_latencies.append(self.cycle - item.injected_cycle)
        self.pending_sampling = still_pending

    def _advance_links(self) -> None:
        for index, transit in enumerate(self.links):
            if transit is None:
                continue
            self.link_busy_cycles[index] += 1
            if transit.remaining > 0:
                transit.remaining -= 1
            destination = self.stages[transit.destination]
            if transit.remaining == 0 and len(destination.queue) < self.config.fifo_depth:
                destination.queue.append(transit.item)
                self.links[index] = None

    def _move_finished(self) -> None:
        for stage in reversed(self.stages):
            if not stage.finished:
                continue
            item = stage.finished[0]
            if stage.stage_id == self.config.num_stages - 1:
                self.pending_sampling.append((self.cycle + self.config.sampling_cycles, item))
            else:
                link_index = self._link_index_after(stage.stage_id)
                if link_index is not None:
                    if self.links[link_index] is not None:
                        stage.counters.output_stalled_cycles += 1
                        continue
                    packet_bytes = self.config.activation_bytes + self.config.packet_overhead_bytes
                    if stage.is_head:
                        # Head chips forward the hidden vector plus their top-k partial result.
                        packet_bytes += self.config.head_result_bytes
                    link_cycles = math.ceil(packet_bytes / self.config.link_bytes_per_cycle)
                    self.links[link_index] = Transit(item, stage.stage_id + 1, link_cycles)
                else:
                    destination = self.stages[stage.stage_id + 1]
                    if len(destination.queue) >= self.config.fifo_depth:
                        stage.counters.output_stalled_cycles += 1
                        continue
                    destination.queue.append(item)
            stage.finished.popleft()
            stage.counters.completed += 1

    def _advance_stages(self) -> None:
        for stage in self.stages:
            if stage.active:
                stage.counters.busy_cycles += 1
            completed = []
            for operation in stage.active:
                operation.remaining -= 1
                if operation.remaining == 0:
                    completed.append(operation)
            for operation in completed:
                stage.active.remove(operation)
                stage.finished.append(operation.item)

            if not stage.queue:
                stage.counters.input_starved_cycles += 1
                continue
            max_inflight = self.config.global_max_inflight if stage.is_global else 1
            if len(stage.active) + len(stage.finished) >= max_inflight:
                continue
            item = stage.queue.popleft()
            service_cycles, initiation_interval, memory_bytes = self._timing(stage, item)
            if stage.last_start_cycle >= 0 and self.cycle - stage.last_start_cycle < initiation_interval:
                stage.queue.appendleft(item)
                continue
            asic = None
            if memory_bytes and not stage.is_head:
                asic = stage.stage_id // self.config.layers_per_asic
                if self.memory_busy_until[asic] > self.cycle:
                    stage.queue.appendleft(item)
                    continue
            stage.active.append(InFlight(item, service_cycles))
            stage.last_start_cycle = self.cycle
            stage.counters.accepted += 1
            stage.counters.memory_bytes += memory_bytes
            if asic is not None:
                self.memory_bytes_per_asic[asic] += memory_bytes
                bandwidth = self.config.memory_bytes_per_cycle * self.config.memory_efficiency
                self.memory_busy_until[asic] = self.cycle + math.ceil(memory_bytes / bandwidth)
            if self.config.trace:
                self.trace_events.append(TraceEvent(
                    "head" if stage.is_head else "FULL global" if stage.is_global else "recurrent",
                    "stage",
                    self.cycle,
                    service_cycles,
                    stage.stage_id,
                    item.context_id,
                    item.sequence,
                ))

    def _inject(self) -> None:
        first = self.stages[0]
        if len(first.queue) >= self.config.fifo_depth:
            return
        for offset in range(self.config.resident_contexts):
            index = (self.next_context + offset) % self.config.resident_contexts
            context = self.contexts[index]
            if (not context.in_flight and context.ready_cycle <= self.cycle
                    and context.generated < self._target_tokens()):
                first.queue.append(WorkItem(
                    context.context_id,
                    context.next_position,
                    context.generated,
                    self.cycle,
                ))
                context.in_flight = True
                if (context.generated >= self.config.warmup_tokens_per_context
                        and self.measurement_start_cycle is None):
                    self.measurement_start_cycle = self.cycle
                self.next_context = (index + 1) % self.config.resident_contexts
                return

    def _record_watermarks(self) -> None:
        for index, stage in enumerate(self.stages):
            self.fifo_high_watermarks[index] = max(self.fifo_high_watermarks[index], len(stage.queue))

    def _done(self) -> bool:
        return all(context.generated >= self._target_tokens() for context in self.contexts)

    def _target_tokens(self) -> int:
        return self.config.warmup_tokens_per_context + self.config.tokens_per_context

    def run(self) -> SimulationResult:
        while not self._done():
            if self.cycle >= self.config.max_cycles:
                raise TimeoutError(f"simulation exceeded max_cycles={self.config.max_cycles}")
            self._advance_sampling()
            self._advance_links()
            self._move_finished()
            self._advance_stages()
            self._inject()
            self._record_watermarks()
            self.cycle += 1

        completed = len(self.completed_latencies)
        measurement_start = self.measurement_start_cycle or 0
        measurement_cycles = self.cycle - measurement_start
        seconds = measurement_cycles / (self.config.clock_mhz * 1_000_000)
        tokens_per_second = completed / seconds
        cfg = self.config
        # Memory traffic is accumulated over the whole run (warm-up included),
        # so per-token bytes use every token the pipeline processed.
        all_tokens = cfg.resident_contexts * self._target_tokens()
        memory_bytes_per_token = sum(self.memory_bytes_per_asic) / all_tokens
        compute_mj = energy_per_token_mj(cfg)
        memory_mj = memory_bytes_per_token * cfg.memory_energy_pj_per_byte * 1e-12 * 1e3
        compute_w = compute_mj * 1e-3 * tokens_per_second
        memory_w = memory_mj * 1e-3 * tokens_per_second
        return SimulationResult(
            cycles=self.cycle,
            measurement_cycles=measurement_cycles,
            completed_tokens=completed,
            aggregate_tokens_per_second=completed / seconds,
            mean_token_latency_cycles=sum(self.completed_latencies) / completed,
            max_token_latency_cycles=max(self.completed_latencies),
            stage_utilization=tuple(stage.counters.busy_cycles / self.cycle for stage in self.stages),
            stage_output_stalls=tuple(stage.counters.output_stalled_cycles for stage in self.stages),
            fifo_high_watermarks=tuple(self.fifo_high_watermarks),
            link_utilization=tuple(cycles / self.cycle for cycles in self.link_busy_cycles),
            memory_bytes_per_asic=tuple(self.memory_bytes_per_asic),
            compute_energy_per_token_mj=compute_mj,
            memory_energy_per_token_mj=memory_mj,
            compute_power_w=compute_w,
            memory_power_w=memory_w,
            board_power_w=compute_w + memory_w + cfg.static_power_w,
            layer_asic_power_w=cfg.layer_macs_per_token * cfg.mac_energy_pj * 1e-12 * tokens_per_second,
        )

    def write_chrome_trace(self, path: str | Path) -> None:
        events = [event.chrome_event(self.config.clock_mhz) for event in self.trace_events]
        with Path(path).open("w", encoding="utf-8") as output:
            json.dump({"traceEvents": events}, output, indent=2)
