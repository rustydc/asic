import unittest

import numpy as np

from hw import si


class SolverTest(unittest.TestCase):
    def test_matched_line_shows_no_reflection_and_the_right_delay(self) -> None:
        net = si.Network()
        root = net.node()
        net.drive(root, si.Z0)
        end = net.line(root, 40.0)
        net.shunt(end, si.Z0, 0.0)
        net.loads["end"] = end
        t, hist, src = si.simulate(net, 8.0)
        # The step divides between the source and line impedances: half amplitude at both ends, flat.
        self.assertAlmostEqual(float(src[(t > 1.0) & (t < 1.9)].mean()), si.VDD / 2, places=2)
        self.assertAlmostEqual(float(hist["end"][(t > 1.3) & (t < 1.9)].mean()), si.VDD / 2, places=2)
        self.assertLess(float(hist["end"][(t > 1.3) & (t < 1.9)].std()), 0.01)
        delay = t[np.argmax(hist["end"] > si.VDD / 4)] - t[np.argmax(src > si.VDD / 4)]
        self.assertAlmostEqual(float(delay), 40.0 / si.V_MM_NS, places=2)

    def test_crossings_interpolate(self) -> None:
        t = np.array([0.0, 1.0, 2.0, 3.0])
        v = np.array([0.0, 1.0, 1.0, 0.0])
        np.testing.assert_allclose(si.crossings(t, v, 0.25, True), [0.25])
        np.testing.assert_allclose(si.crossings(t, v, 0.25, False), [2.75])


class ClockNetTest(unittest.TestCase):
    """The design decision: four devices on one clock fail the edge limit, one per device passes."""

    def test_one_clock_per_device_passes_at_both_placements(self) -> None:
        for trunk in (si.TRUNK_AS_PLACED_MM, si.TRUNK_SOUTH_MM):
            r = si.run(si.point_to_point(trunk), 50.0)
            m = r.loads["d0"]
            self.assertTrue(r.ok)
            self.assertLess(max(m.rise_ns, m.fall_ns), 0.45)
            self.assertLess(m.v_max, si.VDD + 0.05)              # a matched drive: no overshoot
            self.assertTrue(m.clean)
        # The north placement also admits the stronger 33 ohm drive at every device.
        for trunk in (15.0, 55.0):
            self.assertTrue(si.run(si.point_to_point(trunk), 33.0).ok)

    def test_four_on_one_clock_fails_the_edge_limit(self) -> None:
        star = si.run(si.star(si.TRUNK_NORTH_MM, (12.0, 4.0, 4.0, 12.0)), 33.0)
        self.assertGreater(star.worst_edge_ns, si.T_KHKL_MAX_NS)
        self.assertFalse(star.ok)
        fly = si.run(si.fly_by(si.TRUNK_NORTH_MM, si.DEVICE_PITCH_MM, 4), 25.0)
        self.assertGreater(fly.worst_edge_ns, si.T_KHKL_MAX_NS)
        self.assertFalse(fly.ok)
        # And the strong drive that would meet the edge rings through the absolute maximum.
        hard = si.run(si.star(si.TRUNK_NORTH_MM, (12.0, 4.0, 4.0, 12.0)), 12.0)
        self.assertGreater(max(m.v_max for m in hard.loads.values()), si.VDD + si.OVERSHOOT_V)

    def test_two_per_clock_passes_only_in_a_narrow_band(self) -> None:
        self.assertTrue(si.run(si.star(si.TRUNK_NORTH_MM, (4.0, 4.0)), 33.0).ok)
        self.assertFalse(si.run(si.star(si.TRUNK_NORTH_MM, (4.0, 4.0)), 50.0).ok)

    def test_report(self) -> None:
        results = si.study(trunks=(si.TRUNK_NORTH_MM,), r_sweep=(33.0,))
        text = si.report_markdown(results, si.point_to_point_sweep(lengths=(55.0,), drives=(50.0,)))
        self.assertIn("| point-to-point | 33 ohm |", text)
        self.assertIn("| 55 mm | 50 ohm |", text)
        self.assertIn("## Reading", text)


if __name__ == "__main__":
    unittest.main()
