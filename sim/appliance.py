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
    max_cycles: int = 2_000_000_000
    trace: bool = False

    @property
    def num_layers(self) -> int:
        return self.num_asics * self.layers_per_asic

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
            "max_cycles": self.max_cycles,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"configuration fields must be positive: {', '.join(invalid)}")
        if (self.sampling_cycles < 0 or self.packet_overhead_bytes < 0
                or self.warmup_tokens_per_context < 0):
            raise ValueError("sampling, overhead, and warm-up values cannot be negative")
        if self.memory_efficiency > 1:
            raise ValueError("memory_efficiency cannot exceed one")

    @classmethod
    def from_json(cls, path: str | Path) -> "ApplianceConfig":
        with Path(path).open(encoding="utf-8") as source:
            values = json.load(source)
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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Simulation:
    """Deterministic appliance simulation with round-robin context injection."""

    def __init__(self, config: ApplianceConfig):
        config.validate()
        self.config = config
        self.cycle = 0
        self.stages = [
            Stage(i, (i + 1) % config.layers_per_asic == 0, deque())
            for i in range(config.num_layers)
        ]
        self.contexts = [Context(i, config.initial_context_tokens) for i in range(config.resident_contexts)]
        self.links: list[Transit | None] = [None] * (config.num_asics - 1)
        self.link_busy_cycles = [0] * (config.num_asics - 1)
        self.fifo_high_watermarks = [0] * config.num_layers
        self.memory_bytes_per_asic = [0] * config.num_asics
        self.pending_sampling: list[tuple[int, WorkItem]] = []
        self.completed_latencies: list[int] = []
        self.measurement_start_cycle: int | None = None
        self.trace_events: list[TraceEvent] = []
        self.next_context = 0

    def _global_memory_components(self, item: WorkItem) -> tuple[int, int, int]:
        cfg = self.config
        compressed_positions = math.ceil(item.position / cfg.compression_ratio)
        index_bytes = math.ceil(compressed_positions * cfg.index_dim * cfg.index_bits / 8)
        selected_positions = cfg.top_blocks * cfg.retrieval_block_size
        kv_positions = cfg.local_window + selected_positions
        kv_bytes = math.ceil(kv_positions * 2 * cfg.head_dim * cfg.kv_element_bytes)
        append_bytes = math.ceil(cfg.index_dim * cfg.index_bits / 8 + 2 * cfg.head_dim * cfg.kv_element_bytes)
        return index_bytes, kv_bytes, append_bytes

    def _timing(self, stage: Stage, item: WorkItem) -> tuple[int, int, int]:
        if not stage.is_global:
            cycles = math.ceil(self.config.recurrent_cycles * self.config.recurrent_weight_scale)
            return cycles, cycles, 0
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
        if (stage_id + 1) % self.config.layers_per_asic != 0:
            return None
        if stage_id == self.config.num_layers - 1:
            return None
        return stage_id // self.config.layers_per_asic

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
            if stage.stage_id == self.config.num_layers - 1:
                self.pending_sampling.append((self.cycle + self.config.sampling_cycles, item))
            else:
                link_index = self._link_index_after(stage.stage_id)
                if link_index is not None:
                    if self.links[link_index] is not None:
                        stage.counters.output_stalled_cycles += 1
                        continue
                    packet_bytes = self.config.activation_bytes + self.config.packet_overhead_bytes
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
            stage.active.append(InFlight(item, service_cycles))
            stage.last_start_cycle = self.cycle
            stage.counters.accepted += 1
            stage.counters.memory_bytes += memory_bytes
            if stage.is_global:
                asic = stage.stage_id // self.config.layers_per_asic
                self.memory_bytes_per_asic[asic] += memory_bytes
            if self.config.trace:
                self.trace_events.append(TraceEvent(
                    "FULL global" if stage.is_global else "recurrent",
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
        )

    def write_chrome_trace(self, path: str | Path) -> None:
        events = [event.chrome_event(self.config.clock_mhz) for event in self.trace_events]
        with Path(path).open("w", encoding="utf-8") as output:
            json.dump({"traceEvents": events}, output, indent=2)
