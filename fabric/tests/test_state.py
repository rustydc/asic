import unittest

from fabric import state as ST


class StateWidthTest(unittest.TestCase):
    """The rounding study at a small size: the scaled int8 state tracks the
    float rule where a plain int8 state does not."""

    def test_scaled_int8_tracks_the_float_rule(self) -> None:
        result = ST.study(n=900, k=32, v=32, seed=2)
        for mode in ST.DECAY_MODES:
            self.assertGreater(result[mode][ST.SCHEMES[0]][0], 0.999)          # int16
            self.assertGreater(result[mode][ST.SCHEMES[3]][0], 0.98)           # int8 with a scale
        # A plain int8 state loses the fast decay to rounding; stochastic rounding is noisy.
        self.assertLess(result["fast"][ST.SCHEMES[1]][0], result["fast"][ST.SCHEMES[3]][0] - 0.01)
        self.assertLess(result["fast"][ST.SCHEMES[1]][1], result["fast"][ST.SCHEMES[3]][1] - 0.03)
        self.assertLess(result["fast"][ST.SCHEMES[2]][1], result["fast"][ST.SCHEMES[3]][1])
        self.assertIn("| int8 with a per-head scale (taken) |", ST.report_markdown(result, 900, 32, 32))


if __name__ == "__main__":
    unittest.main()
