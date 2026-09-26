import unittest

from fabric import clock as C
from fabric import sequencer as S
from fabric.memory import MemoryMap
from fabric.tile import TileSpec


def row(unit, ps, library="nangate45", **extra):
    return {"unit": unit, "library": library, "critical_path_ps": ps, "nand2_equiv": 1000.0, **extra}


class ClockTest(unittest.TestCase):
    def test_the_fo4_reading_is_the_one_the_model_was_given(self) -> None:
        # The attention core's 2,127 ps at the middle of 15 to 18 ps is the 585 MHz of Timing.core_mhz.
        self.assertAlmostEqual(C.to_28nm_mhz(2127, 16.5), 585, delta=1)

    def test_a_unit_counts_at_its_best_target_and_not_when_it_timed_a_heavy_input(self) -> None:
        rows = [row("a", 2000, abc_target_ps=1200), row("a", 1800, abc_target_ps=900),
                row("b", 1500), row("b", 900, heavy_inputs={"lane.issue": 853}),
                row("c", 5000, library="asap7"), {"unit": "d", "library": "nangate45", "error": "killed"}]
        best = C.best(rows)
        self.assertEqual(best["a"]["critical_path_ps"], 1800)
        self.assertEqual(best["b"]["critical_path_ps"], 1500)
        self.assertNotIn("c", best)
        self.assertNotIn("d", best)

    def test_the_slowest_unit_sets_the_clock_and_a_missing_one_is_named(self) -> None:
        c = C.clock([row("a", 1800), row("b", 1500)], {"a": "", "b": "", "e": ""})
        self.assertEqual((c.unit, c.path_ps, c.missing), ("a", 1800, ["e"]))
        self.assertLess(c.mhz_range[0], c.mhz)
        self.assertLess(c.mhz, c.mhz_range[1])
        self.assertAlmostEqual(c.mhz_wired, c.mhz / (1 + C.WIRE_MARGIN), delta=0.01)


class LaneRateTest(unittest.TestCase):
    def setUp(self) -> None:
        from fixed_llm_poc import tiny_config
        self.cfg = tiny_config()
        self.spec, self.mm = TileSpec(), MemoryMap.from_config(self.cfg)

    def programs(self, t):
        return (S.recurrent_program(self.cfg, None, self.spec, self.mm, t, slots=S.LANE_SLOTS),
                S.global_program(self.cfg, None, self.spec, self.mm, 5, t))

    def test_one_lane_is_a_token_and_the_link_gap_at_a_time(self) -> None:
        t = S.Timing(core_mhz=500)
        rec, glob = self.programs(t)
        cycles = S.schedule(S.lane_program([rec, rec, rec, glob], 0)).cycles
        r = S.lane_rate(rec, glob, 1, t)
        self.assertAlmostEqual(r.tokens_per_s, 500e6 / (cycles + S.LINK_GAP), delta=1e-6)
        self.assertEqual(r.lane_cycles, cycles + S.LINK_GAP)

    def test_lanes_overlap_tokens(self) -> None:
        t = S.Timing(core_mhz=500)
        rec, glob = self.programs(t)
        one, four = S.lane_rate(rec, glob, 1, t), S.lane_rate(rec, glob, 4, t)
        # The tiny layers are one shared unit's work more than the link's gap,
        # so the overlap has little to hide: 1.16 times here, 1.5 at 9B.
        self.assertGreater(four.tokens_per_s, 1.1 * one.tokens_per_s)
        self.assertGreaterEqual(four.lane_cycles, one.lane_cycles)


if __name__ == "__main__":
    unittest.main()
