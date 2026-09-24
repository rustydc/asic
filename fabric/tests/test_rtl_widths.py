"""A ratchet on the two width bugs this design keeps growing back.

Neither is a compile error and neither shows up in simulation -- the RTL is
correct either way, and only the critical path knows.  Both were found by
reading a timing report, several times each, so they are counted here
instead: the budget is what is left, and a file that grows one fails.

The first is arithmetic evaluated at 64 bits.  The helpers in
``fabric_fx.svh`` take a ``signed [63:0]``, and a function argument is an
assignment, so every expression handed to one is widened to 64 whatever the
numbers are.  A 48-bit resolve, its variable shift and a 16-bit saturate all
ran at 64 in the state engine; the norm's requantizer and the residual's
round did the same.  Narrowing them to the width the values have was worth
216 ps on the state engine, 480 on the norm and 466 on the residual.

The second is the same thing by hand: a 64-bit temp in the datapath, which
widens everything assigned into it.  ``pr`` in the state engine was one, and
holding the attention core's output scale at 56 bits where the product is 51
cost 40 ps.

Where the count is deliberate the exemption is in the budget, with a reason,
not in the RTL.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

RTL = Path(__file__).resolve().parents[1] / "rtl"

# Calls into the fx_ helpers, which evaluate their argument at 64 bits.
FX = re.compile(r"\bfx_(?:sat|rnd_shr|requant)\s*\(")
# A 64-bit datapath temp, which widens whatever is assigned into it.
W64 = re.compile(r"\breg\s+(?:signed\s+)?\[63:0\]")

# What is left, and why it is allowed to be.
FX_BUDGET = {
    "fabric_engine.sv": 2,       # the engine's top level, not a measured unit
    "fabric_recurrent.sv": 13,   # pass 1 and the reduce; the diff and y stages are narrowed
    "fabric_vector.sv": 2,       # the generic requantizer's reference path
}
W64_BUDGET = {
    "fabric_controller.sv": 1,   # a program-counter product, off the datapath
    "fabric_engine.sv": 2,
    "fabric_memory.sv": 1,
    "fabric_recurrent.sv": 2,    # tr/tn/dl/mag/nsat in pass 1 and the reduce
    "fabric_tile.sv": 1,         # blocking temps whose saturation literals assume 64
}


def _sources() -> list[Path]:
    """The synthesizable RTL.  Testbenches may be as wide as they like."""
    return sorted(p for p in RTL.glob("*.sv") if not p.name.startswith("tb_"))


class RtlWidthRatchetTest(unittest.TestCase):
    def _check(self, pattern: re.Pattern[str], budget: dict[str, int], what: str) -> None:
        over, under = [], []
        for path in _sources():
            found = len(pattern.findall(path.read_text()))
            allowed = budget.get(path.name, 0)
            if found > allowed:
                over.append(f"{path.name}: {found} {what}, budget {allowed}")
            elif found < allowed:
                under.append(f"{path.name}: {found} {what}, budget {allowed}")
        self.assertFalse(over, "\n".join(
            [f"new {what} -- narrow it to the width the numbers have, or raise the "
             f"budget in this test with the reason:"] + over))
        self.assertFalse(under, "\n".join(
            [f"fewer {what} than the budget: lower it so the ratchet holds the ground "
             f"that was won:"] + under))

    def test_no_new_64_bit_fx_calls(self) -> None:
        self._check(FX, FX_BUDGET, "fx_ calls evaluated at 64 bits")

    def test_no_new_64_bit_datapath_temps(self) -> None:
        self._check(W64, W64_BUDGET, "64-bit datapath temps")
