import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fabric.tile import (DensityModel, TileSpec, compile_matrix, decompile, emit_vectors, fixed_point_scale,
                         matrix_forward, quantize_matrix, reference_matmul, report_appliance, requantize,
                         tile_forward, tile_spec_for, via_coordinates)

RTL = Path(__file__).parents[1] / "rtl"
SMALL = TileSpec(rows=64, cols=8, rows_per_cycle=2, acc_bits=24)


def random_quantized(rng, in_features, out_features, spec):
    weight = rng.standard_normal((in_features, out_features))
    return quantize_matrix(weight, spec, act_scale=0.05, out_scale=0.5)


class ArithmeticTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)

    def test_tile_forward_equals_integer_matmul(self) -> None:
        for spec in (SMALL, TileSpec()):
            q = random_quantized(self.rng, spec.rows, spec.cols, spec)
            tile = compile_matrix(q, spec).tiles[0]
            x = self.rng.integers(-128, 128, spec.rows)
            psum, y = tile_forward(tile, x, spec)
            np.testing.assert_array_equal(psum, reference_matmul(q, x))
            np.testing.assert_array_equal(y, requantize(reference_matmul(q, x), q.mult, q.shift, spec))

    def test_chained_partial_sums(self) -> None:
        q = random_quantized(self.rng, SMALL.rows, SMALL.cols, SMALL)
        tile = compile_matrix(q, SMALL).tiles[0]
        x = self.rng.integers(-128, 128, SMALL.rows)
        seed = self.rng.integers(-1000, 1000, SMALL.cols)
        psum, _ = tile_forward(tile, x, SMALL, psum_in=seed)
        np.testing.assert_array_equal(psum, seed + reference_matmul(q, x))

    def test_matrix_forward_reduces_row_blocks_and_pads_columns(self) -> None:
        q = random_quantized(self.rng, 150, 20, SMALL)   # 3 row blocks, 3 column blocks
        compiled = compile_matrix(q, SMALL)
        self.assertEqual(len(compiled.tiles), 9)
        x = self.rng.integers(-128, 128, 150)
        psum, y = matrix_forward(compiled, x)
        np.testing.assert_array_equal(psum, reference_matmul(q, x))
        np.testing.assert_array_equal(y, requantize(reference_matmul(q, x), q.mult, q.shift, SMALL))
        self.assertAlmostEqual(compiled.utilization, 150 * 20 / (9 * 64 * 8))

    def test_compile_round_trip_and_symmetric_alphabet(self) -> None:
        q = random_quantized(self.rng, 100, 13, SMALL)
        self.assertGreaterEqual(int(q.weights.min()), -SMALL.weight_max)
        self.assertLessEqual(int(q.weights.max()), SMALL.weight_max)
        np.testing.assert_array_equal(decompile(compile_matrix(q, SMALL)), q.weights)

    def test_requantize_rounds_half_up_and_saturates(self) -> None:
        spec = SMALL
        acc = np.array([5, -5, 1000000, -1000000, 3])
        mult = np.array([1, 1, 1, 1, 3], dtype=np.uint16)
        shift = np.array([1, 1, 0, 0, 2], dtype=np.uint8)
        # 5/2 -> 3 (round half up), -5/2 -> -2 (floor of -2.5 + 0.5), saturation both ways, 9/4 -> 2
        np.testing.assert_array_equal(requantize(acc, mult, shift, spec), np.array([3, -2, 127, -128, 2], dtype=np.int8))

    def test_fixed_point_scale_is_close(self) -> None:
        values = np.array([0.5, 0.01, 3.0, 1e-4])
        mult, shift = fixed_point_scale(values, TileSpec())
        approx = mult.astype(np.float64) / (2.0 ** shift)
        np.testing.assert_allclose(approx, values, rtol=1e-3)

    def test_via_coordinates_match_via_count(self) -> None:
        q = random_quantized(self.rng, SMALL.rows, SMALL.cols, SMALL)
        tile = compile_matrix(q, SMALL).tiles[0]
        coords = via_coordinates(tile, SMALL)
        self.assertEqual(len(coords), tile.via_count(SMALL))
        self.assertEqual(coords.shape[1], 2)


class MappingTest(unittest.TestCase):
    def test_qwen_geometries_fit_the_fabric(self) -> None:
        try:
            from fixed_llm_poc import ASICLMConfig
        except ImportError:  # pragma: no cover
            self.skipTest("PyTorch not installed")
        density = DensityModel()
        cfg9, cfg4 = ASICLMConfig.from_preset("qwen3_5_9b"), ASICLMConfig.from_preset("qwen3_5_4b")
        spec = tile_spec_for(cfg9)
        self.assertEqual((spec.rows, tile_spec_for(cfg4).rows), (4096, 2560))
        nine = report_appliance(cfg9, spec, density)
        four = report_appliance(cfg4, tile_spec_for(cfg4), density)
        self.assertAlmostEqual(nine["layer_die"].coefficients / 1e6, 866, delta=3)
        self.assertGreater(nine["layer_die"].utilization, 0.95)
        self.assertGreater(four["layer_die"].utilization, 0.90)
        # The head slice must fit in the layer die's tile count.
        self.assertLess(nine["head_die"].tiles, nine["layer_die"].tiles)
        self.assertLess(four["head_die"].tiles, four["layer_die"].tiles)
        self.assertLess(four["layer_die"].tiles, nine["layer_die"].tiles)
        # Four sequential passes per layer, four layers per die.
        self.assertEqual(nine["layer_die"].stage_cycles, 4 * 4 * spec.cycles_per_pass)
        self.assertEqual(nine["head_die"].stage_cycles, spec.cycles_per_pass)


@unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "iverilog not installed")
class RtlTest(unittest.TestCase):
    def run_rtl(self, spec: TileSpec, seed: int, with_psum: bool) -> str:
        rng = np.random.default_rng(seed)
        q = random_quantized(rng, spec.rows, spec.cols, spec)
        tile = compile_matrix(q, spec).tiles[0]
        x = rng.integers(-128, 128, spec.rows)
        psum_in = rng.integers(-5000, 5000, spec.cols) if with_psum else None
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            emit_vectors(work, tile, x, spec, psum_in)
            params = [f"-Ptb_fabric_tile.{name}={value}" for name, value in (
                ("ROWS", spec.rows), ("COLS", spec.cols), ("WB", spec.weight_bits), ("AB", spec.act_bits),
                ("P", spec.rows_per_cycle), ("ACC", spec.acc_bits), ("SB", spec.scale_bits), ("SHB", spec.shift_bits))]
            subprocess.run(["iverilog", "-g2012", "-o", "sim.vvp", *params,
                            str(RTL / "fabric_tile.sv"), str(RTL / "tb_fabric_tile.sv")],
                           cwd=work, check=True, capture_output=True, text=True)
            result = subprocess.run(["vvp", "sim.vvp"], cwd=work, check=True, capture_output=True, text=True)
        return result.stdout

    def test_small_tile_bit_exact(self) -> None:
        out = self.run_rtl(SMALL, seed=1, with_psum=True)
        self.assertIn("PASS", out, out)

    def test_wide_tile_bit_exact(self) -> None:
        out = self.run_rtl(TileSpec(rows=256, cols=64, rows_per_cycle=4), seed=2, with_psum=False)
        self.assertIn("PASS", out, out)

    def test_full_depth_tile_bit_exact(self) -> None:
        out = self.run_rtl(TileSpec(rows=4096, cols=16, rows_per_cycle=2), seed=3, with_psum=True)
        self.assertIn("PASS", out, out)


if __name__ == "__main__":
    unittest.main()
