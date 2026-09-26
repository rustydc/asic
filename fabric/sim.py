"""Running a testbench: Verilator where it is installed, Icarus otherwise.

The engine's testbenches take minutes to tens of minutes under Icarus, most
of it the simulator interpreting structure written for the mapper; Verilator
compiles the same RTL to C++ and runs the tiny engine's token in under a
second, cycle for cycle and bit for bit the same.  Its build is the cost --
a minute or two for an engine -- so builds are cached by what they were
built from: the sources' contents, the top and its parameters.  A test that
repeats a configuration reuses the binary.

``FABRIC_SIM=icarus`` runs everything under Icarus, and a test may ask for
Icarus by name, which is how the cross-checks keep the two honest.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

RTL = Path(__file__).parent / "rtl"
CACHE = Path(os.environ.get("FABRIC_SIM_CACHE", Path(__file__).parent / ".simcache"))
FLAGS = ("--binary", "--timing", "-Wno-fatal", "-Wno-lint", "-Wno-style", "-CFLAGS", "-O1")


def default() -> str:
    """The simulator to use: FABRIC_SIM if set, else Verilator when there is one."""
    chosen = os.environ.get("FABRIC_SIM")
    if chosen:
        return chosen
    return "verilator" if shutil.which("verilator") else "icarus"


def _value(v) -> str:
    """A parameter's value on a command line; Verilator wants a sized literal past 32 bits."""
    if isinstance(v, int) and not -(1 << 31) <= v < (1 << 31):
        return f"64'd{v}"
    return str(v)


def build_verilator(top: str, sources: list[Path], params: dict) -> Path:
    """The Verilator binary for ``top`` over ``sources`` with ``params``,
    built once and cached; returns the executable."""
    h = hashlib.sha256()
    h.update(subprocess.run(["verilator", "--version"], capture_output=True, text=True).stdout.encode())
    h.update(" ".join(FLAGS).encode())
    h.update(top.encode())
    for name, value in sorted(params.items()):
        h.update(f"{name}={_value(value)};".encode())
    for src in sources:
        h.update(Path(src).name.encode())
        h.update(Path(src).read_bytes())
    for inc in sorted(RTL.glob("*.svh")):                    # included, so part of what was built
        h.update(inc.read_bytes())
    key = h.hexdigest()[:24]
    exe = CACHE / key / f"V{top}"
    if exe.exists():
        return exe
    CACHE.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{key}-", dir=CACHE))
    args = [f"-G{name}={_value(value)}" for name, value in params.items()]
    jobs = str(max(1, min(4, (os.cpu_count() or 2) // 2)))
    result = subprocess.run(["verilator", *FLAGS, "-j", jobs, f"-I{RTL}", "--top-module", top, "--Mdir", str(work / "obj"),
                             "-o", f"V{top}", *args, *map(str, sources)], capture_output=True, text=True)
    if result.returncode != 0:
        shutil.rmtree(work, ignore_errors=True)
        raise RuntimeError(f"verilator failed:\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}")
    (work / f"V{top}").write_bytes((work / "obj" / f"V{top}").read_bytes())
    (work / f"V{top}").chmod(0o755)
    shutil.rmtree(work / "obj")
    try:
        work.rename(CACHE / key)                             # another build of the same key may have won
    except OSError:
        shutil.rmtree(work, ignore_errors=True)
    return exe


def run(work: Path, top: str, sources: list[Path], params: dict, simulator: str | None = None) -> str:
    """Simulate ``top`` in ``work`` (where its input files are) and return what it printed."""
    simulator = simulator or default()
    if simulator == "verilator":
        exe = build_verilator(top, sources, params)
        return subprocess.run([str(exe)], cwd=work, check=True, capture_output=True, text=True).stdout
    args = [f"-P{top}.{name}={value}" for name, value in params.items()]
    subprocess.run(["iverilog", "-g2012", "-I", str(RTL), "-s", top, "-o", "sim.vvp", *args, *map(str, sources)],
                   cwd=work, check=True, capture_output=True, text=True)
    return subprocess.run(["vvp", "sim.vvp"], cwd=work, check=True, capture_output=True, text=True).stdout
