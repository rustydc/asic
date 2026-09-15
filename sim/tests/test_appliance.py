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

    def test_head_asics_extend_the_ring_without_memory(self) -> None:
        config = replace(SMALL, num_head_asics=2, head_cycles=3, head_result_bytes=4)
        simulation = Simulation(config)
        self.assertEqual(len(simulation.stages), 10)
        self.assertEqual(len(simulation.links), 3)
        self.assertTrue(simulation.stages[8].is_head and simulation.stages[9].is_head)
        result = simulation.run()
        self.assertEqual(result.completed_tokens, 12)
        self.assertEqual([stage.counters.accepted for stage in simulation.stages], [12] * 10)
        self.assertEqual(len(result.memory_bytes_per_asic), config.num_asics)
        self.assertEqual(simulation.stages[8].counters.memory_bytes, 0)
        # The head hop carries the hidden vector plus the top-k partial result.
        self.assertGreater(result.link_utilization[2], result.link_utilization[0])
        self.assertGreater(result.mean_token_latency_cycles,
                           Simulation(SMALL).run().mean_token_latency_cycles)

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            Simulation(replace(SMALL, fifo_depth=0))
        with self.assertRaises(ValueError):
            Simulation(replace(SMALL, mac_energy_pj=-1.0))

    def test_energy_model_scales_with_mac_energy_and_throughput(self) -> None:
        config = replace(SMALL, mac_energy_pj=2.0, layer_macs_per_token=1e9, num_head_asics=2,
                         head_macs_per_token=5e8, static_power_w=10.0, memory_energy_pj_per_byte=1.0)
        result = Simulation(config).run()
        # 2 layer ASICs x 1e9 + 2 head ASICs x 5e8 = 3e9 MACs x 2 pJ = 6 mJ per token.
        self.assertAlmostEqual(result.compute_energy_per_token_mj, 6.0)
        self.assertAlmostEqual(result.compute_power_w, 6e-3 * result.aggregate_tokens_per_second)
        self.assertAlmostEqual(result.layer_asic_power_w, 2e-3 * result.aggregate_tokens_per_second)
        self.assertGreater(result.memory_power_w, 0.0)
        self.assertAlmostEqual(result.board_power_w, result.compute_power_w + result.memory_power_w + 10.0)
        doubled = Simulation(replace(config, mac_energy_pj=4.0)).run()
        self.assertAlmostEqual(doubled.compute_power_w, 2 * result.compute_power_w)
        self.assertAlmostEqual(doubled.memory_power_w, result.memory_power_w)

    def test_shipped_configurations_load_and_validate(self) -> None:
        configs = sorted((Path(__file__).resolve().parents[1] / "config").glob("*.json"))
        self.assertGreaterEqual(len(configs), 4)
        loaded = {path.stem: ApplianceConfig.from_json(path) for path in configs}
        for config in loaded.values():
            config.validate()
        hbm = [name for name in loaded if "hbm" in name]
        self.assertEqual(len(hbm), 2)
        for name in hbm:
            # HBM configurations: faster fabric, int4 KV at 16:1, far more bandwidth and contexts.
            self.assertLess(loaded[name].recurrent_cycles, loaded["baseline"].recurrent_cycles)
            self.assertGreater(loaded[name].memory_bytes_per_cycle, 10 * loaded["baseline"].memory_bytes_per_cycle)
            self.assertGreater(loaded[name].resident_contexts, loaded["baseline"].resident_contexts)
            self.assertLess(loaded[name].mac_energy_pj, loaded["baseline"].mac_energy_pj)

    def test_board_power_helper_matches_simulation(self) -> None:
        from sim.appliance import board_power_w, energy_per_token_mj
        config = replace(SMALL, mac_energy_pj=3.0, layer_macs_per_token=866e6, static_power_w=70.0)
        self.assertAlmostEqual(energy_per_token_mj(config), 2 * 866e6 * 3e-12 * 1e3)
        self.assertAlmostEqual(board_power_w(config, 1000.0), 2 * 866e6 * 3e-12 * 1000.0 + 70.0)


if __name__ == "__main__":
    unittest.main()
