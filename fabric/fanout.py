"""Every register read by more than a handful of loads, over kept netlists.

The mapper buffers a combinational cone freely and cannot buffer a flop's own
output, so a register that a whole lane array reads pays for it in
clock-to-output.  That has been the single most common defect in this design;
this finds the rest of them without waiting for each unit's critical path to
surface one at a time.

    python -m fabric.fanout <min-loads> <netlist-dir>...

The directories are what ``fabric.synth_units --keep`` writes.
"""
import re, sys
from collections import Counter
from pathlib import Path

MIN = int(sys.argv[1])
INST = re.compile(r"^\s*([A-Z][A-Za-z0-9_]*)\s+(\S+)\s*\(([^;]*)\);", re.M | re.S)
CONN = re.compile(r"\.(\w+)\(([^)]*)\)")

rows = []
for d in sys.argv[2:]:
    for f in sorted(Path(d).glob("*_nangate45.v")):
        text = f.read_text()
        loads, driver = Counter(), {}
        for m in INST.finditer(text):
            cell, inst, conns = m.group(1), m.group(2), m.group(3)
            for pin, net in CONN.findall(conns):
                net = net.strip()
                if not net or net in ("clk", "rst_n") or net.startswith(("1'b", "{")):
                    continue
                if pin in ("Q", "QN", "ZN", "Z"):          # an output drives it
                    driver.setdefault(net, (cell, inst))
                else:
                    loads[net] += 1
        for net, (cell, inst) in driver.items():
            if cell.startswith("DFF") and loads[net] >= MIN:
                rows.append((loads[net], f.stem.replace("_nangate45", ""), net))
rows.sort(reverse=True)
seen = set()
print(f"{'loads':>6}  {'unit':22s} register")
for n, unit, net in rows:
    base = re.sub(r"\s*\[\d+\]\s*$", "", net).strip("\\ ")
    if (unit, base) in seen:
        continue
    seen.add((unit, base))
    print(f"{n:6d}  {unit:22s} {base}")
