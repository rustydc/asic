import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from sim.appliance import ApplianceConfig, Simulation, WorkItem


SMALL = ApplianceConfig(
    clock_mhz=100,
    num_asics=2,
    layers_per_asic=4,
    resident_contexts=4,
    tokens_per_context=3,
    initial_context_tokens=16,
    warmup_tokens_per_context=0,
    recurrent_cycles=2,
    global_index_compute_cycles=3,
    global_topk_cycles=1,
    global_attention_cycles=1,
    global_output_cycles=1,
    fifo_depth=1,
    sampling_cycles=2,
    activation_bytes=8,
    packet_overhead_bytes=0,
    link_bytes_per_cycle=8,
    memory_bytes_per_cycle=1_000_000,
    compression_ratio=4,
    retrieval_block_size=2,
    top_blocks=1,
    local_window=2,
    index_dim=4,
    index_bits=4,
    head_dim=2,
    max_cycles=10_000,
)


class SimulationTest(unittest.TestCase):
    def test_completes_every_token_without_context_overlap(self) -> None:
        simulation = Simulation(SMALL)
        result = simulation.run()
        self.assertEqual(result.completed_tokens, 12)
        self.assertEqual([context.generated for context in simulation.contexts], [3] * 4)
        self.assertTrue(all(not context.in_flight for context in simulation.contexts))
        self.assertEqual([stage.counters.accepted for stage in simulation.stages], [12] * 8)

    def test_finite_fifos_create_backpressure_without_loss(self) -> None:
        config = replace(SMALL, recurrent_cycles=1, global_index_compute_cycles=12,
                         global_max_inflight=1, resident_contexts=8)
        result = Simulation(config).run()
        self.assertEqual(result.completed_tokens, 24)
        self.assertTrue(any(stalls > 0 for stalls in result.stage_output_stalls))
        self.assertTrue(all(depth <= config.fifo_depth for depth in result.fifo_high_watermarks))

    def test_longer_context_increases_full_scan_memory(self) -> None:
        short = Simulation(replace(SMALL, initial_context_tokens=16)).run()
        long = Simulation(replace(SMALL, initial_context_tokens=1024)).run()
        self.assertTrue(all(long_bytes > short_bytes for long_bytes, short_bytes
                            in zip(long.memory_bytes_per_asic, short.memory_bytes_per_asic)))

    def test_trace_is_chrome_compatible_json(self) -> None:
        simulation = Simulation(replace(SMALL, tokens_per_context=1, trace=True))
        simulation.run()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            simulation.write_chrome_trace(path)
            events = json.loads(path.read_text(encoding="utf-8"))["traceEvents"]
        self.assertTrue(events)
        self.assertEqual(events[0]["ph"], "X")

    def test_warmup_tokens_are_excluded_from_metrics(self) -> None:
        config = replace(SMALL, warmup_tokens_per_context=2, tokens_per_context=1)
        simulation = Simulation(config)
        result = simulation.run()
        self.assertEqual(result.completed_tokens, config.resident_contexts)
        self.assertEqual([context.generated for context in simulation.contexts], [3] * 4)

    def test_global_phases_overlap_but_share_memory_interval(self) -> None:
        simulation = Simulation(SMALL)
        latency, interval, _ = simulation._timing(
            simulation.stages[3],
            WorkItem(0, 1024, 0, 0),
        )
        self.assertGreater(latency, interval)

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            Simulation(replace(SMALL, fifo_depth=0))


if __name__ == "__main__":
    unittest.main()
