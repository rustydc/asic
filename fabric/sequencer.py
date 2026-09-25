"""The token sequencer: the program that runs a layer's tiles, vector
units and DMAs in order for one token, the merge of several tokens'
programs into one stream, and the controller that executes it.

A layer is a fixed dataflow graph over a few dozen on-chip vector buffers:
the residual, the pass input, the pass outputs, the per-head state slots.
This module writes that graph out as a *program*: a list of steps, each a
command to one unit (a tile pass, a norm, a state-engine head, a DMA) with
the buffers it consumes and produces.  Three things are derived from the
same list, so they cannot drift apart:

* ``run_program`` executes the steps on the integer model of ``layer.py``
  and must reproduce ``recurrent_layer_int`` / ``global_layer_int`` bit for
  bit, which proves the program's data flow (slices, head loops, buffer
  reuse) is complete and in a legal order;
* ``schedule`` runs a list scheduler over the steps with each unit's
  throughput and the memory port's bandwidth, giving cycles per token and
  where the time goes, under the controller's exact issue rules so the RTL
  testbench's cycle count must equal it;
* ``emit_program`` writes the steps as the controller's program image.

Dependencies come from buffer names.  A step consumes the buffers it
reads and produces the ones it writes; a write waits for the buffer's
outstanding writers and readers, a read for its outstanding writers.  A
produce marked ``+name`` is a *contribution* to a whole vector by one of
several parallel producers (a head's slice of ``y_norm``): it waits only
for the readers of the previous version, and a consumer of the whole waits
for every contribution.

Several tokens of different contexts run at once by ``stream``: each
token's program has its buffers renamed with a token suffix (the state
slots stay shared, they are the engines' physical buffers), and the
programs are merged into one issue order by simulating which step could
issue first, so one token's memory steps run under another's passes.

The controller (``rtl/fabric_sequencer.sv``) is a microcoded issue engine:
in program order, when the head step's consumed buffers have no
outstanding writer, its produced buffers no outstanding writer or reader
(readers only, for a contribution), and its unit reports the addressed
engine free, it sends the command and moves on; a completing unit returns
the step's tag and the controller releases the step's buffers.  Units are
black boxes to it, which is what lets one testbench check the controller
against stub units of programmed duration.
"""
from __future__ import annotations

import collections
import dataclasses
import functools
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from fabric import hpi
from fabric import layer as L
from fabric.memory import BEAT, MemoryMap, _beats, eligible_blocks
from fabric.tile import TileSpec, write_hex

# --------------------------------------------------------------------------
# The units and their throughput (from the RTL's lanes and latencies)
# --------------------------------------------------------------------------

MAX_CONSUME = 6              # buffers a step may consume (the program word's fields)
MAX_PRODUCE = 2              # buffers a step may produce
MAX_IDS = 255                # buffer ids per program image (0xFF is "none")
UNITS: dict[str, tuple[int, int]] = {   # name -> (id, engines)
    "tiles": (0, 1), "norm": (1, 2), "conv": (2, 1), "gates": (3, 1), "delta": (4, 4),
    "swiglu": (5, 1), "residual": (6, 1), "rotary": (7, 2), "attn": (8, 4), "mem": (9, 1),
}
NE = max(engines for _, engines in UNITS.values())    # engine ports a unit has in the controller's port map
RELEASES = 1                 # completions the controller drains a cycle (rtl/fabric_sequencer.sv NREL)
# Heads whose state the recurrent program reads ahead of the head it is on.
# The controller issues in order, so a head's read placed after the previous
# head's write-back waits out that head's update, and the port idles for it
# every head: 12,300 cycles of a 104,751-cycle token at the 9B geometry.
# Read ahead, the update runs under the next reads.  At most seven: the four
# engines have two slots each, and the eighth head ahead is this head's slot.
# Two or more gives the single token the whole saving (92,462 cycles).  Over
# tokens 3-6 of a stream of contexts, three is the best of the depths at
# 73,943 cycles (two to seven: 73,943-77,668), against 77,712 in order (0).
STATE_READ_AHEAD = 3
SHARED_PREFIX = "s_slot"     # buffers shared by every token in flight: the state engines' slots
MEM_PREFIX = "m_"            # names of buffers in the memory image (the rest live in the vector buffer)

# Operand conventions of the layer engine's adapters (rtl/fabric_engine.sv).
HIST_REC = 4                 # bytes per channel of the conv history in the vector buffer (kernel - 1 used)
SLOT_HEADER = BEAT           # the state slot: one beat of scale, exponent, peak, saturated count, then the rows
MEM_RD, MEM_WR = 0, 1        # the memory unit's operations (arg[3:0]): memory to vector buffer, vector buffer to memory
# A read of a context's state flagged fresh is, on a FIRST token, a fill instead:
# zeros for the conv history, and a state slot's header beat with the scale at
# 1.0 then zero rows.  The flags are in every program; the token's FIRST decides.
MEM_FRESH_ZERO, MEM_FRESH_SLOT = 1 << 4, 1 << 5
ADDR_BITS = 30               # the program word's address operands (src, dst, a2, a3)
# Vector-buffer ports each adapter has.  A step names every buffer it consumes,
# but a one-port unit reads them in turn -- the state engine takes its slot, its
# unit vector, the conv output and the gates through one port -- so those cannot
# want the same bank in the same cycle.  These must match ``R_*`` and ``W_*`` in
# rtl/fabric_engine.sv.
RD_PORTS = {"tiles": 1, "norm": 2, "conv": 2, "gates": 2, "delta": 1,
            "swiglu": 2, "residual": 2, "rotary": 1, "attn": 1, "mem": 1}
WR_PORTS = {"tiles": 1, "norm": 1, "conv": 2, "gates": 1, "delta": 1,
            "swiglu": 1, "residual": 1, "rotary": 1, "attn": 1, "mem": 1}
MEM_APPEND, MEM_SCAN, MEM_ROWS = 2, 3, 4   # the global layer's: append the token, scan the index, stream a head's rows
MEM_PAGE_SHIFT = 7           # a memory page (the map's alignment) in beats: the context base travels as a page number
MAX_TOKENS = 4               # tokens in flight the engine holds a slot for
SLOT_TOKEN_SHIFT = 28        # a memory command's token in flight, in a3's top bits
ROT_TABLE, ROT_HEAD = 0, 1   # the rotary unit's operations (arg[3:0]); arg[7:4] the head kind, 0 q and 1 k
NORM_INT16 = 1 << 8          # the norm's input elements are int16 (else int8)
NORM_GATED = 1 << 9          # the norm's gain is silu of the requantized int8 vector at arg[31:16]
NORM_RESIDUAL, NORM_UNIT, NORM_GATE, NORM_FFN = 0, 1, 2, 3    # the norm's constant sets


@dataclasses.dataclass(frozen=True)
class Timing:
    """Lanes and latencies of the units, and the memory they share.

    The lane counts are ``rtl/fabric_engine.sv``'s own parameters -- ``NL``
    for the vector units, ``CL`` for the conv, ``ATT_L`` for the attention
    cores and the record reader, ``P`` and ``COLS`` for a tile -- and not a
    second estimate of them.  The programs' beat counts come from the same
    fields, so a machine with wider units is one ``Timing`` and one
    elaboration rather than two machines that disagree about which one they
    are; that disagreement is what made this model half the engine's real
    cycle count for as long as nothing compared them.

    The latencies are each adapter's own, read off its state machine and
    checked against the RTL command by command by ``EngineCycleTest``.  A
    command costs its beats plus the adapter's latency, and the two are
    separate fields so that a wider machine keeps the latency.
    """
    # The core clock, from the measured units rather than a round number.
    # The slowest of them (``fabric/results/synth_units.json``) is the
    # attention core at 2,127 ps on NanGate 45, and that library's FO4 is
    # 20.54 ps on a twenty-stage fanout-of-four inverter chain, so the stage
    # is 103.5 FO4 deep.  A 28 nm FO4 of 15 to 18 ps puts it at 1.55 to 1.86
    # ns, 537 to 644 MHz; this is the middle of that.  Pre-layout, one
    # corner, and the 28 nm FO4 is an assumption -- no 28 nm library here
    # measures it.  See the clock section of the README.
    core_mhz: float = 585.0
    # Lanes, as the RTL is elaborated.
    # A buffer beat is sixteen bytes, so a unit's lanes are capped by what its
    # operands weigh: eight for the norm and the residual, whose vectors are
    # int16, sixteen for the units whose operands are int8.
    lanes: int = 8                   # NL: the norm and the residual
    l_vec: int = 16                  # SW_L: SwiGLU
    l_conv: int = 4                  # CL: the causal conv
    l_attn: int = 16                 # ATT_L at most: a beat of the buffer, as int8
    # `P` and `COLS` are the tile's, and come from the TileSpec the pass is compiled for.
    # Adapter latencies: what a command costs beyond its beats.
    norm_latency: int = 34           # the front pipeline, the sum pass's tail and the inverse square root
    tile_block_latency: int = 13     # a row block: its start cycle and the requantizer walk's tail
    tile_done_latency: int = 2       # the write walk's last tile and the report
    conv_latency: int = 14
    gates_latency: int = 13
    delta_latency: int = 29          # three passes over K rows, then the gated output
    swiglu_latency: int = 12
    residual_latency: int = 6
    rotary_latency: int = 3
    rotary_table_latency: int = 8
    attn_row_stall: int = 9          # in_ready drops while the core exponentiates a key row
    attn_out_latency: int = 29
    attn_start_latency: int = 3
    # The memory the unit talks to.  ``port_*`` is one request of the port
    # itself; the rest is what each operation does around it.
    port_request: int = 7            # accept, read latency and turnaround
    port_read_beat: int = 1          # a read beat a cycle
    port_write_beat: int = 1         # the mover's addresses run a beat ahead of its data
    mem_read_latency: int = 8
    fill_latency: int = 4            # FIRST: a fresh read's fill, two beats a cycle with no request on the port
    append_first_saving: int = 4     # FIRST: the append's block sums filled rather than read
    mem_write_latency: int = 6
    reader_latency: int = 8
    reader_arrive: int = 5           # a request taken to its first beat in the ring (the testbench memory: 2 cycles)
    reader_gap: int = 2              # a record's emission: the cycle it starts and the one it retires
    reader_reissue: int = 2          # the rows loop: one request taken to the next offered
    scan_latency: int = 19           # the query in and its codes, before any record is read
    scan_request: int = 5
    append_beat: int = 3             # the append's own write path, a beat at a time
    append_latency: int = 58
    append_index_latency: int = 35   # the block's index projection, its codes and its record
    # The memory behind the port.  ``devices`` of 0 is the testbench's own
    # model, a beat a cycle after a short latency, which is what the engine
    # tests run against and what the constants above are measured on.  With
    # devices it is the HPI path to that many APS512XXN parts at their
    # datasheet timing (``hpi.PathModel``): the stripe map sends each 1 KB of
    # a request to the next device, so a request of a stripe or less gets one
    # device however many there are, and each device's chunk is a burst of
    # its own -- the command, the latency, two words a clock and tCPH.
    # ``memory_mode`` is the controller as built or with its faults fixed;
    # ``pushout`` the refresh push-out a read burst takes (None: the
    # testbench device model's mean).
    devices: int = 0
    memory_mode: str = "asbuilt"
    pushout: float | None = None
    # The scan and the record reader keep one request in flight and wait for
    # it, so a page of records reaches one or two devices of the sixteen.
    # ``deep_requests`` is those two units asking ahead, which only the
    # pipelined path can take: their requests then overlap across devices.
    deep_requests: bool = False
    stripe_beats: int = hpi.STRIPE_BYTES // BEAT
    burst_clocks: int = 20           # a burst's command, latency and tCPH, at 250 MHz
    beat_clocks: int = 4             # x16 DDR: four clocks to a sixteen-byte beat
    controller_mhz: float = 250.0    # the HPI controller's own clock

    # Both of these are the memory measured against the core clock, so they
    # follow it rather than repeating a number it might no longer have.
    @property
    def port_bytes_per_cycle(self) -> float:
        """Sixteen devices at 250 MHz DDR x16, per core cycle."""
        return 16e9 / (self.core_mhz * 1e6)

    @property
    def ctrl_ratio(self) -> float:
        """Core cycles to a controller clock."""
        return self.core_mhz / self.controller_mhz

    def beats(self, n: int) -> int:
        return -(-n // self.lanes)

    def head_lanes(self, head_dim: int) -> int:
        """The lanes of the units that stream a head: the attention cores, the
        record reader and the rotary.  They take ``HD / L`` beats with no
        ragged last one, so the lane count has to divide the head, and their
        operands are int8, so a buffer beat holds sixteen of them."""
        return max(l for l in (1, 2, 4, 8, 16) if l <= self.l_attn and head_dim % l == 0)

    def norm(self, d: int) -> int:
        return 2 * self.beats(d) + self.norm_latency

    def tile_pass(self, spec: TileSpec, row_blocks: int, tokens: int, walk: int) -> int:
        """A pass over ``row_blocks`` row blocks for ``tokens`` tokens.

        Every row block streams the tile's rows ``rows_per_cycle`` at a time
        and then waits out the columns' requantizer walk, which is a column a
        cycle for each token.  ``walk`` is the write phase, which visits
        *every* tile of the array whether or not it is in this pass -- a
        cycle for one that is not, and a beat of output per token for one
        that is.  That walk is the array's size rather than the pass's work,
        and it is what this model used to charge nothing for.
        """
        block = spec.rows // spec.rows_per_cycle + spec.cols * tokens + self.tile_block_latency
        return row_blocks * block + walk + self.tile_done_latency

    def attention(self, rows: int, group: int, head_dim: int) -> int:
        """One core's command: the queries and the gates in, then a key row
        and a value row each, then the group's outputs.  A beat a cycle: the
        adapter asks the buffer for the next beat while the core takes this
        one (rtl/fabric_engine.sv).  It used to ask and then present, two
        cycles a beat, and a row cost twice its beats."""
        beats = head_dim // self.head_lanes(head_dim)
        return (self.attn_start_latency + 2 * group * beats
                + rows * (2 * beats + self.attn_row_stall) + group * beats + self.attn_out_latency)

    @property
    def path(self) -> "hpi.PathModel | None":
        if not self.devices:
            return None
        return _path(self.devices, self.memory_mode, self.core_mhz, self.controller_mhz, self.pushout)

    def transfer(self, beats: int) -> int:
        """Core cycles the memory needs for one transfer of ``beats``.

        With the testbench's model that is a beat a cycle.  Over the HPI
        devices the stripe map cuts the transfer into ``stripe_beats``
        chunks and hands them round the devices, so the chunks of different
        devices go at once and a device's own chunks go in turn.  Aggregate
        only: the beat mover's writes are posted, so a write command looks
        almost free and the read after it pays, and no per-command number
        here means anything on that path."""
        if not self.devices:
            return beats
        stripe = self.stripe_beats
        sizes = [min(stripe, beats - i * stripe) for i in range(max(1, -(-beats // stripe)))]
        load = [0] * min(self.devices, len(sizes))
        for i, size in enumerate(sizes):                 # the map hands the chunks round in turn
            load[i % len(load)] += self.burst_clocks + self.beat_clocks * size
        return int(max(load) * self.ctrl_ratio)

    def move(self, beats: int, write: bool) -> int:
        """The beat mover: memory to the vector buffer, or back, two beats a
        transfer -- the port's wide requests and the buffer's wide port.
        Over the devices it is one request on the path, and a write is
        charged its completion: the mover's writes are posted, but the next
        memory step waits for the path, and the steps take the memory unit
        in turn."""
        if self.path is not None:
            if write and self.path.mode == "asbuilt":
                # Posted: the bridge's queues take 64 beats at once, the
                # rest as the path feeds them to the devices.
                taken = self.path.taken(beats - 64) if beats > 64 else 0
                return max(taken, -(-beats // 2)) + self.mem_write_latency
            return self.path.cost(write, beats) + (self.mem_write_latency if write else self.mem_read_latency)
        transfers = -(-beats // 2)
        return (self.port_write_beat * transfers + self.mem_write_latency) if write else (transfers + self.mem_read_latency)

    def port(self, beats: int, write: bool) -> tuple[int, bool]:
        """A mover request's hold on the path as built, and whether it is a
        posted write: see ``Step.port``."""
        if self.path is None or self.path.mode != "asbuilt":
            return 0, False
        return self.path.request(write, beats), write

    def read_records(self, requests: list[int], record_beats: int, head_dim: int, maxr: int) -> int:
        """The record reader over ``requests`` (the records in each), cycle for
        cycle (rtl/fabric_memory.sv).  Two stages with a ring of ``2 * maxr``
        records between them: a request's records arrive ``reader_arrive``
        after it is taken and then a record's beats apart, and the next is
        taken once they have all arrived and the ring has room for it; a
        record goes out when it has arrived and the one before it is out,
        its two rows a beat a cycle and ``reader_gap`` more.  Which stage
        binds depends on the geometry: at the 9B one the rows out (34
        cycles a record) are twice the beats in, and at the tests' small
        ones a request's arrival is the longer.

        This used to charge every record its beats in and its rows out one
        after the other, and a request's latency on top.  The tests' small
        geometries cannot tell that from the reader -- a record's two beats
        in there are exactly the two cycles its emission starts and retires
        in -- but at the 9B one it was 48 cycles a record where the reader
        takes 34, and it counted a window of 512 records as one request
        where the engine takes it a page of eight at a time."""
        out = 2 * (head_dim // self.head_lanes(head_dim)) + self.reader_gap
        cap = 2 * maxr
        path = self.path
        if path is not None and self.deep_requests and path.mode == "pipelined":
            # Every request asked for ahead: the records arrive at the rate
            # the devices give them, and go out at the reader's.
            got = path.sequence([(False, n * record_beats, i * n * record_beats * BEAT) for i, n in enumerate(requests)])
            return self.reader_latency + path.first_beat(record_beats) + max(got, sum(requests) * out)
        taken, prev, done, arrived = 0, None, [], 0
        for count in requests:
            if prev is not None:
                # Taken when the last request is in and the ring has room.
                taken = max(prev, taken + self.reader_reissue)
                need = arrived + count - cap                  # records that must be out first
                if need > 0:
                    taken = max(taken, done[need - 1] + 1)
            lead, spacing = self.reader_arrive, record_beats
            if path is not None:
                # Over the devices the request's records come back as the
                # path has them: its first beat, then the rest at the rate its
                # devices give -- after the whole of a chunk, as built, since
                # a chunk is drained only once it is filled.
                first = path.first_beat(count * record_beats)
                lead = first - record_beats
                spacing = max(record_beats, (path.request(False, count * record_beats) - first) / count)
            for k in range(count):
                arrive = taken + lead + (k + 1) * spacing
                done.append(max(done[-1] if done else 0, arrive) + out)
            arrived += count
            prev = taken + lead + count * spacing
        return self.reader_latency + int(math.ceil(done[-1] if done else 0))

    def scan(self, records: int, record_beats: int, selected: int, per_request: int = 0) -> int:
        """The index scan: the query's codes, then every eligible record read in
        one burst and scored, then the chosen ids out a cycle each.  Over the
        devices the records come a page (``per_request``) a request, each
        request's first beat the path's latency after it and then a beat a
        cycle into the scorer."""
        if not records:
            return self.scan_latency
        if self.path is not None and per_request and self.deep_requests and self.path.mode == "pipelined":
            got = self.path.sequence([(False, records * record_beats, 0)])
            return self.scan_latency + self.path.first_beat(record_beats) + max(got, record_beats * records) + selected - 1
        if self.path is not None and per_request:
            pages = [min(per_request, records - i) for i in range(0, records, per_request)]
            each = sum(max(self.path.request(False, n * record_beats), self.path.first_beat(n * record_beats) + n * record_beats)
                       for n in pages)
            return self.scan_latency + each + selected - 1
        return self.scan_latency + self.scan_request + record_beats * records + selected - 1

    def write_record(self, record_beats: int) -> int:
        """A record the append writes.  It has its own path to the port, not
        the mover's, and still takes three cycles a beat: the same address,
        data, present that the mover used to.  Over the devices, its request
        on the path."""
        if self.path is not None:
            return max(self.path.cost(True, record_beats), record_beats * self.append_beat)
        return self.port_request + record_beats * self.append_beat

    def fill(self, beats: int) -> int:
        """A fresh read on a FIRST token: the buffer filled, two beats a cycle."""
        return -(-beats // 2) + self.fill_latency

    def append(self, heads: int, record_beats: int, block_end: bool, first: bool = False, sums_beats: int = 0,
               index_beats: int = 0) -> int:
        """The token's window records, and at a block's end its block means and
        index record.  Over the devices the block sums' read and write and the
        index record are requests on the path too."""
        one = heads * self.write_record(record_beats)
        extra = 0
        if self.path is not None and sums_beats:
            ideal = 2 * -(-sums_beats // 2)
            extra = self.path.cost(False, sums_beats) + self.path.cost(True, sums_beats) - ideal
            if block_end and index_beats:
                extra += self.path.cost(True, index_beats)
        return (self.append_latency - (self.append_first_saving if first else 0) + one
                + (one + self.append_index_latency if block_end else 0) + extra)

    def memory(self, nbytes: int, burst_bytes: int) -> int:
        """Cycles the port is busy moving nbytes in bursts of burst_bytes, at the HPI burst efficiency."""
        return int(math.ceil(nbytes / (self.port_bytes_per_cycle * hpi.efficiency(max(1, burst_bytes // BEAT)))))


POSTED_SLACK = 40        # core cycles of a posted write the bridge's queues hold while the path is busy (64 beats)


@functools.lru_cache(maxsize=None)
def _path(devices: int, mode: str, core_mhz: float, f_mhz: float, pushout: float | None) -> "hpi.PathModel":
    return hpi.PathModel(devices, mode, core_mhz, f_mhz, pushout)


# --------------------------------------------------------------------------
# Steps, dependencies, programs
# --------------------------------------------------------------------------

@dataclasses.dataclass
class Step:
    name: str
    unit: str
    engine: int
    src: tuple[str, ...]             # buffers consumed
    dst: tuple[str, ...]             # buffers produced; "+name" contributes to a whole vector
    cycles: int
    func: Callable[[dict], None] | None = None
    nbytes: int = 0                  # memory traffic, for the report
    token: int | None = None         # set on a stream's steps: their private buffers carry the token suffix
    deps: list[int] = dataclasses.field(default_factory=list)
    ops: dict | None = None          # the command's operands for the layer engine (see ``operands``)
    # The memory path as built: ``port`` is the cycles the step's request
    # holds it, and a ``posted`` write's step ends when its data is taken,
    # before the path has written it.  A later step waits for the path only
    # for its own request (``schedule``).  Both are zero on the testbench's
    # memory and on the pipelined path, whose steps are charged what they
    # occupy.
    port: int = 0
    posted: bool = False

    def __post_init__(self) -> None:
        self.cycles = max(1, int(self.cycles))       # a command occupies its unit for at least a cycle
        assert len(self.src) <= MAX_CONSUME and len(self.dst) <= MAX_PRODUCE, self.name


def _plain(name: str) -> str:
    return name[1:] if name.startswith("+") else name


def window_requests(pos: int, window: int, page: int) -> list[int]:
    """The reader requests one head's window records take, as the engine
    issues them: the window is a ring of ``window`` slots written at
    ``pos % window``, read as runs that stop at its wrap, each a page of
    ``page`` records at a time."""
    p, out = max(0, pos - window + 1), []
    while p <= pos:
        count = min(pos - p + 1, window - p % window, page)
        out.append(count)
        p += count
    return out


def pass_matrices(cfg, recurrent: bool) -> list[list[tuple[int, int, bool]]]:
    """Each pass's matrices as ``(in_features, out_features, raw)``.

    ``engine._passes`` compiles the same list from the weights; only the
    shapes matter for the timing, and they follow from the config, so the
    model can have them without a compiled layer.  ``engine._tiles`` checks
    that the two agree."""
    nh, nkv, hd, idim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.index_dim
    nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    hk, hv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    d = cfg.hidden_size
    layer = 0 if recurrent else cfg.global_layer_offset
    ffn = cfg.layer_intermediate_size(layer)
    common = [[(d, ffn, False), (d, ffn, False)], [(ffn, d, False)]]     # gate and up, then down
    if recurrent:
        kd, vd = nk * hk, nv * hv
        return [[(d, 2 * kd + vd, False), (d, vd, False), (d, nv, True), (d, nv, True)], [(vd, d, False)]] + common
    return [[(d, nh * 2 * hd, False), (d, nkv * hd, False), (d, nkv * hd, False),
             (d, idim, False), (d, idim, False)], [(nh * hd, d, False)]] + common


def pass_walk(cfg, spec: TileSpec, recurrent: bool) -> list[int]:
    """What the write phase of each pass costs, per token of a chunk.

    The pass adapter's write phase steps over *every* tile of the array, one
    cycle for a tile that is not in this pass and one beat of its output per
    token for one that is, so the array's size is part of every pass's cost.
    Returned per pass as ``fixed + per_token``, packed as a pair."""
    passes = pass_matrices(cfg, recurrent)
    total = sum(-(-i // spec.rows) * -(-o // spec.cols) for p in passes for i, o, _ in p)
    out = []
    for matrices in passes:
        beats, hits = 0, 0
        for i, o, raw in matrices:
            for cb in range(-(-o // spec.cols)):
                valid = min(spec.cols, o - cb * spec.cols)
                beats += -(-(valid * (4 if raw else 1)) // BEAT)
                hits += 1
        out.append((total - hits, beats))
    return out


def operands(**fields) -> dict:
    """The command fields of a step for the layer engine: the address
    operands ``src``, ``dst``, ``a2`` and ``a3`` (30 bits), the argument
    ``arg`` (32 bits) and ``len`` (16 bits), each an int, a reference
    ``(buffer, byte_offset)`` resolved against the engine's layout (a third
    element shifts the address right, for a page number), or a list of
    ``(shift, value)`` parts OR-ed together.  Vector-buffer references are
    byte addresses, memory references (``m_...``) beat addresses."""
    return {k: v for k, v in fields.items() if v is not None}


def _rename_value(value, token: int):
    if isinstance(value, tuple):
        return (_renamed(value[0], token),) + value[1:]
    if isinstance(value, list):
        return [(sh, _rename_value(v, token)) for sh, v in value]
    return value


def resolve_value(value, layout) -> int:
    """An operand value as an integer, references looked up in ``layout.address``."""
    if value is None:
        return 0
    if isinstance(value, tuple):
        lookup = getattr(layout, "operand", layout.address)      # a memory name: its offset in the token's slot
        return (lookup(value[0]) + value[1]) >> (value[2] if len(value) > 2 else 0)
    if isinstance(value, list):
        return sum(resolve_value(v, layout) << sh for sh, v in value)
    return int(value)


class Linker:
    """The dependency rules, applied step by step: per buffer the writers of
    its current version and the readers since."""

    def __init__(self) -> None:
        self.writers: dict[str, list[int]] = {}
        self.readers: dict[str, list[int]] = {}

    def deps_for(self, step: Step) -> list[int]:
        deps: set[int] = set()
        for name in step.src:
            deps.update(self.writers.get(name, []))
        for name in step.dst:
            plain = _plain(name)
            deps.update(self.readers.get(plain, []))
            if not name.startswith("+"):
                deps.update(self.writers.get(plain, []))
        return sorted(deps)

    def commit(self, step: Step, index: int) -> None:
        for name in step.src:
            self.readers.setdefault(name, []).append(index)
        for name in step.dst:
            plain = _plain(name)
            if name.startswith("+") and not self.readers.get(plain):
                self.writers.setdefault(plain, []).append(index)     # another contribution to the current version
            else:
                self.writers[plain] = [index]                        # a new version
                self.readers[plain] = []


def link(steps: list[Step]) -> None:
    """Fill each step's dependencies in program order."""
    lk = Linker()
    for i, step in enumerate(steps):
        step.deps = lk.deps_for(step)
        lk.commit(step, i)


def _slot(engine: int, k: int) -> str:
    """The physical state buffer of an engine: two per engine, so the next head's read overlaps this one."""
    return f"{SHARED_PREFIX}[{engine * 2 + k % 2}]"


def _chunk_env(chunk: int):
    """Accessors of a chunk's per-token values in the environment: with one
    token the plain arrays, with more the token's slice of a ``[T, ...]``
    array (or entry of a list)."""
    def get(e, name, t):
        return e[name][t] if chunk > 1 else e[name]

    def put(e, name, t, value):
        if chunk == 1:
            e[name] = value
            return
        try:
            holder = e[name]
        except KeyError:
            holder = np.zeros((chunk,) + np.shape(value), dtype=np.int64) if isinstance(value, np.ndarray) else [None] * chunk
            e[name] = holder
        holder[t] = value
    return get, put


def _contrib(name: str, chunk: int) -> str:
    """A per-token step's produce: the whole buffer for one token, a contribution to it in a chunk."""
    return name if chunk == 1 else "+" + name


def recurrent_program(cfg, c: L.RecurrentConsts | None, spec: TileSpec, mm: MemoryMap, t: Timing = Timing(),
                      chunk: int = 1, read_ahead: int = STATE_READ_AHEAD, first: bool = False) -> list[Step]:
    """One token through a recurrent layer, or a chunk of ``chunk``
    consecutive tokens of one context (prefill): the passes carry the whole
    chunk at once, the vector units and state engines take the tokens in
    turn, and each head's state is read and written once per chunk.

    Inputs in the environment: ``x`` (int16 residual; ``[chunk, d]`` for a
    chunk), ``s_mem[h]`` (the state rows of each head) and, for the int8
    state, ``scale_mem[h]`` (its scale beat), ``hist_mem`` (the conv
    history).  With ``c`` None the program has its shape and timing but
    cannot be run.  Every step carries its operands for the layer engine
    (``recurrent_layout``)."""
    nk, nv, hk, hv = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    d, kd, vd = cfg.hidden_size, nk * hk, nv * hv
    conv_dim, ffn = 2 * kd + vd, cfg.layer_intermediate_size(0)
    n_delta = UNITS["delta"][1]
    repeat = nv // nk
    int8_state = mm.state_bits == 8
    head_bytes = hk * hv * mm.state_bits // 8 + (BEAT if int8_state else 0)   # the rows and the scale beat
    lay = recurrent_layout(cfg, spec, mm, chunk)
    off_z, off_b, off_a, p1 = lay["off_z"], lay["off_b"], lay["off_a"], lay["p1"]
    get, put = _chunk_env(chunk)
    T = chunk
    steps: list[Step] = []

    def add(name, unit, src, dst, cycles, func=None, engine=0, nbytes=0, ops=None, port=(0, False)):
        steps.append(Step(name, unit, engine, tuple(src), tuple(dst), int(cycles), func, nbytes, ops=ops, port=port[0], posted=port[1]))

    def fabric(q):
        return lambda x: L._fabric(q, x, spec)

    def hist_name(i: int) -> str:
        return "hist" if i == 0 else ("hist_next" if i == T else f"hist[{i}]")

    def tok(name: str, i: int) -> str:                    # a per-token step's name
        return name if chunk == 1 else f"{name}<{i}>"

    hist_bytes = _beats(conv_dim * (cfg.linear_conv_kernel - 1)) * BEAT
    hist_beats = lay["sizes"]["hist"] // BEAT
    # With ``first`` the token is FIRST: the history and every head's state
    # are fills rather than reads (the program image is the same; only what
    # the token does, and how long it takes, differ).
    def hist_rd(e):
        e["hist"] = np.zeros_like(e["hist_mem"]) if first else e["hist_mem"]
    add("dma.hist_rd", "mem", (), ("hist",), t.fill(hist_beats) if first else t.move(hist_beats, write=False), hist_rd,
        nbytes=0 if first else hist_bytes,
        ops=operands(src=("m_hist", 0), dst=("hist", 0), arg=MEM_RD | MEM_FRESH_ZERO, len=hist_beats),
        port=(0, False) if first else t.port(hist_beats, False))
    # The heads over the state engines, each engine alternating two state
    # slots; a chunk's tokens run on the slot in turn.  The first reads go
    # here, under the norm, the input pass and the conv, which they do not
    # depend on; each later one after the write-back of the head
    # STATE_READ_AHEAD before it.
    slot_beats = lay["sizes"][_slot(0, 0)] // BEAT

    def s_read(h: int) -> None:
        slot = _slot(h % n_delta, h // n_delta)

        def s_rd(e, h=h, slot=slot):                       # the slot holds the rows and, for int8, the scale beat
            rows = np.zeros_like(e["s_mem"][h]) if first else e["s_mem"][h]
            if int8_state:
                e[slot] = (rows, (L.ONE_U, 0, 0, 0) if first else tuple(int(x) for x in e["scale_mem"][h]))
            else:
                e[slot] = rows
        add(f"dma.s_rd[{h}]", "mem", (), (slot,), t.fill(slot_beats) if first else t.move(slot_beats, write=False), s_rd,
            nbytes=0 if first else head_bytes,
            ops=operands(src=(f"m_s[{h}]", 0), dst=(slot, 0), arg=MEM_RD | MEM_FRESH_SLOT, len=slot_beats),
            port=(0, False) if first else t.port(slot_beats, False))
    for h in range(min(read_ahead, nv)):
        s_read(h)
    for i in range(T):
        add(tok("norm.h", i), "norm", ("x",), (_contrib("A", chunk),), t.norm(d), lambda e, i=i: put(e, "A", i, L._norm(get(e, "x", i), c.norm)),
            ops=operands(src=("x", i * 2 * d), dst=("A", i * d), arg=NORM_RESIDUAL | NORM_INT16, len=d // t.lanes))

    def in_proj(e):                                     # one buffer P1 per token: qkv | z | b, a accumulators
        for i in range(T):
            a = get(e, "A", i)
            put(e, "qkv", i, fabric(c.in_proj_qkv)(a)[1])
            put(e, "z", i, fabric(c.in_proj_z)(a)[1])
            put(e, "b_acc", i, fabric(c.in_proj_b)(a)[0])
            put(e, "a_acc", i, fabric(c.in_proj_a)(a)[0])
    walk = pass_walk(cfg, spec, recurrent=True)
    add("pass.in_proj", "tiles", ("A",), ("P1",), t.tile_pass(spec, -(-d // spec.rows), T, walk[0][0] + T * walk[0][1]), in_proj,
        ops=operands(src=("A", 0), dst=("P1", 0), arg=[(0, 0), (8, -(-d // spec.rows)), (16, T)], a2=d, a3=p1))

    for i in range(T):
        def conv(e, i=i):
            y, nxt = L.conv_silu_int(e[hist_name(i)], get(e, "qkv", i), c.conv_w, c.conv_mult_in, c.conv_sh_in,
                                     c.conv_mult_out, c.conv_sh_out)
            put(e, "conv", i, y)
            e[hist_name(i + 1)] = nxt
        add(tok("conv", i), "conv", ("P1", hist_name(i)), (_contrib("conv", chunk), hist_name(i + 1)),
            -(-conv_dim // t.l_conv) + t.conv_latency, conv,
            ops=operands(src=("P1", i * p1), dst=("conv", i * conv_dim), a2=(hist_name(i), 0), a3=(hist_name(i + 1), 0), len=conv_dim // t.l_conv))
    add("dma.hist_wr", "mem", ("hist_next",), (), t.move(hist_beats, write=True), lambda e: e.__setitem__("hist_mem", e["hist_next"]), nbytes=hist_bytes,
        ops=operands(src=("hist_next", 0), dst=("m_hist", 0), arg=MEM_WR, len=hist_beats), port=t.port(hist_beats, True))
    for i in range(T):
        def gates(e, i=i):
            decay, beta = L.head_gates_int(get(e, "a_acc", i)[:nv], get(e, "b_acc", i)[:nv], c.gate_mult_a, c.gate_sh_a,
                                           c.gate_mult_b, c.gate_sh_b, c.a_coef, c.dt_bias)
            put(e, "decay", i, decay)
            put(e, "beta", i, beta)
        add(tok("gates", i), "gates", ("P1",), (_contrib("gates", chunk),), nv + t.gates_latency, gates,
            ops=operands(src=("P1", i * p1 + off_b), a2=("P1", i * p1 + off_a), dst=("gates", i * lay["gates"]), len=nv))
    for i in range(T):
        for j in range(nk):
            for which, off in (("q", 0), ("k", kd)):
                def unit_norm(e, i=i, j=j, off=off, which=which):
                    put(e, f"{which}_unit[{j}]", i, L._norm(get(e, "conv", i)[off + j * hk:off + (j + 1) * hk], c.unit_norm))
                add(tok(f"norm.{which}[{j}]", i), "norm", ("conv",), (f"+qk_unit[{j}]",), t.norm(hk), unit_norm,
                    engine=j % UNITS["norm"][1],
                    ops=operands(src=("conv", i * conv_dim + off + j * hk), dst=(f"qk_unit[{j}]", i * 2 * hk + (hk if which == "k" else 0)),
                                 arg=NORM_UNIT, len=hk // t.lanes))
    for h in range(nv):
        e_id, k = h % n_delta, h // n_delta
        slot = _slot(e_id, k)
        if not read_ahead:
            s_read(h)
        for i in range(T):
            def delta(e, h=h, slot=slot, i=i):
                v = get(e, "conv", i)[2 * kd + h * hv:2 * kd + (h + 1) * hv]
                k_unit, q_unit = get(e, f"k_unit[{h // repeat}]", i), get(e, f"q_unit[{h // repeat}]", i)
                decay, beta = int(get(e, "decay", i)[h]), int(get(e, "beta", i)[h])
                if int8_state:
                    t_new, *scale_new, y = L.delta_state_int8(e[slot][0], k_unit, v, q_unit, decay, beta, *e[slot][1])
                    e[slot] = (t_new, tuple(scale_new))
                else:
                    e[slot], y = L.delta_state_int(e[slot], k_unit, v, q_unit, decay, beta)
                put(e, f"y[{h}]", i, y)
            add(tok(f"delta[{h}]", i), "delta", (slot, f"qk_unit[{h // repeat}]", "conv", "gates"),
                (slot, _contrib(f"y[{h}]", chunk)), 3 * hk + t.delta_latency, delta, engine=e_id,
                ops=operands(src=(f"qk_unit[{h // repeat}]", i * 2 * hk), dst=(f"y[{h}]", i * 2 * hv), a2=("conv", i * conv_dim + 2 * kd + h * hv),
                             a3=(slot, 0), arg=("gates", i * lay["gates"] + 4 * h)))

        def s_wr(e, h=h, slot=slot):
            if int8_state:
                e["s_mem"][h], e["scale_mem"][h] = e[slot]
            else:
                e["s_mem"][h] = e[slot]
        add(f"dma.s_wr[{h}]", "mem", (slot,), (), t.move(slot_beats, write=True), s_wr, nbytes=head_bytes,
            ops=operands(src=(slot, 0), dst=(f"m_s[{h}]", 0), arg=MEM_WR, len=slot_beats), port=t.port(slot_beats, True))
        if read_ahead and h + read_ahead < nv:
            s_read(h + read_ahead)
        for i in range(T):
            def gnorm(e, h=h, i=i):
                gate = L.silu_fixed(L.requant(get(e, "z", i)[h * hv:(h + 1) * hv], c.z_mult, c.z_shift, 16))
                put(e, f"y_norm[{h}]", i, L._norm(get(e, f"y[{h}]", i), c.gated_norm, gain=gate))
            add(tok(f"gnorm[{h}]", i), "norm", (f"y[{h}]", "P1"), ("+y_norm",), t.norm(hv), gnorm,
                engine=h % UNITS["norm"][1],
                ops=operands(src=(f"y[{h}]", i * 2 * hv), dst=("y_norm", i * vd + h * hv), arg=NORM_GATE | NORM_INT16 | NORM_GATED,
                             a2=("P1", i * p1 + off_z + h * hv), len=hv // t.lanes))

    def out_proj(e):
        for i in range(T):
            y_norm = np.concatenate([get(e, f"y_norm[{h}]", i) for h in range(nv)])
            put(e, "mixer", i, fabric(c.out_proj)(y_norm)[1])
    add("pass.out_proj", "tiles", ("y_norm",), ("mixer",), t.tile_pass(spec, -(-vd // spec.rows), T, walk[1][0] + T * walk[1][1]), out_proj,
        ops=operands(src=("y_norm", 0), dst=("mixer", 0), arg=[(0, 1), (8, -(-vd // spec.rows)), (16, T)], a2=vd, a3=d))
    for i in range(T):
        add(tok("residual.1", i), "residual", ("x", "mixer"), (_contrib("x1", chunk),), t.beats(d) + t.residual_latency,
            lambda e, i=i: put(e, "x1", i, L.residual_int(get(e, "x", i), get(e, "mixer", i), c.res_mult, c.res_shift)),
            ops=operands(src=("x", i * 2 * d), a2=("mixer", i * d), arg=0, dst=("x1", i * 2 * d), len=d // t.lanes))
    _ffn_steps(add, c.ffn if c is not None else None, spec, d, ffn, t, walk, chunk)
    link(steps)
    return steps


def recurrent_layout(cfg, spec: TileSpec, mm: MemoryMap, chunk: int = 1) -> dict:
    """The recurrent program's buffers as the layer engine holds them: byte
    sizes of every vector-buffer name and memory-image name (``m_...``), the
    offsets inside one token's pass-1 output ``P1`` (``qkv | z | b words | a
    words``) and its size ``p1``, and the bytes of one token's gates.  A
    chunk's buffers hold its tokens' vectors in turn.  The int8 state's slot
    is a header beat then ``K`` rows of ``V`` bytes; the conv history keeps
    ``HIST_REC`` bytes per channel, one buffer per token boundary."""
    nk, nv, hk, hv = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    d, kd, vd = cfg.hidden_size, nk * hk, nv * hv
    conv_dim, ffn = 2 * kd + vd, cfg.layer_intermediate_size(0)
    assert cfg.linear_conv_kernel - 1 <= HIST_REC
    align = lambda n: -(-n // BEAT) * BEAT
    off_z = 2 * kd + vd
    off_b = align(off_z + vd)
    off_a = align(off_b + 4 * nv)
    p1 = align(off_a + 4 * nv)
    gates = align(4 * nv)
    slot = (SLOT_HEADER + hk * hv) if mm.state_bits == 8 else 2 * hk * hv
    T = chunk
    sizes = {"x": T * 2 * d, "A": T * d, "P1": T * p1, "hist": align(conv_dim * HIST_REC), "conv": T * conv_dim,
             "hist_next": align(conv_dim * HIST_REC), "gates": T * gates, "y_norm": T * vd, "mixer": T * d, "x1": T * 2 * d,
             "A2": T * d, "GU": T * 2 * ffn, "act": T * ffn, "ffn": T * d, "x2": T * 2 * d, "m_hist": align(conv_dim * HIST_REC)}
    for i in range(1, T):
        sizes[f"hist[{i}]"] = align(conv_dim * HIST_REC)
    for i in range(nk):
        sizes[f"qk_unit[{i}]"] = T * 2 * hk                    # a token's unit q then its unit k
    for j in range(2 * UNITS["delta"][1]):
        sizes[_slot(j // 2, j % 2)] = slot
    for h in range(nv):
        sizes[f"y[{h}]"] = T * 2 * hv
        sizes[f"m_s[{h}]"] = slot
    return {"sizes": sizes, "off_z": off_z, "off_b": off_b, "off_a": off_a, "p1": p1, "gates": gates}


def _ffn_steps(add, f: L.FfnConsts | None, spec: TileSpec, d: int, ffn: int, t: Timing, walk: list, chunk: int = 1) -> None:
    get, put = _chunk_env(chunk)
    T = chunk

    def tok(name: str, i: int) -> str:
        return name if chunk == 1 else f"{name}<{i}>"

    for i in range(T):
        add(tok("norm.h2", i), "norm", ("x1",), (_contrib("A2", chunk),), t.norm(d), lambda e, i=i: put(e, "A2", i, L._norm(get(e, "x1", i), f.norm)),
            ops=operands(src=("x1", i * 2 * d), dst=("A2", i * d), arg=NORM_FFN | NORM_INT16, len=d // t.lanes))

    def gate_up(e):
        for i in range(T):
            put(e, "gate_ffn", i, L._fabric(f.gate_proj, get(e, "A2", i), spec)[1])
            put(e, "up_ffn", i, L._fabric(f.up_proj, get(e, "A2", i), spec)[1])
    add("pass.gate_up", "tiles", ("A2",), ("GU",), t.tile_pass(spec, -(-d // spec.rows), T, walk[2][0] + T * walk[2][1]), gate_up,
        ops=operands(src=("A2", 0), dst=("GU", 0), arg=[(0, 2), (8, -(-d // spec.rows)), (16, T)], a2=d, a3=2 * ffn))
    for i in range(T):
        add(tok("swiglu", i), "swiglu", ("GU",), (_contrib("act", chunk),), -(-ffn // t.l_vec) + t.swiglu_latency,
            lambda e, i=i: put(e, "act", i, L.swiglu_int(get(e, "gate_ffn", i), get(e, "up_ffn", i), f.mult_g, f.sh_g, f.mult_o, f.sh_o)),
            ops=operands(src=("GU", i * 2 * ffn), a2=("GU", i * 2 * ffn + ffn), arg=0, dst=("act", i * ffn), len=ffn // t.l_vec))

    def down(e):
        for i in range(T):
            put(e, "ffn", i, L._fabric(f.down_proj, get(e, "act", i), spec)[1])
    add("pass.down", "tiles", ("act",), ("ffn",), t.tile_pass(spec, -(-ffn // spec.rows), T, walk[3][0] + T * walk[3][1]), down,
        ops=operands(src=("act", 0), dst=("ffn", 0), arg=[(0, 3), (8, -(-ffn // spec.rows)), (16, T)], a2=ffn, a3=d))
    for i in range(T):
        add(tok("residual.2", i), "residual", ("x1", "ffn"), (_contrib("x2", chunk),), t.beats(d) + t.residual_latency,
            lambda e, i=i: put(e, "x2", i, L.residual_int(get(e, "x1", i), get(e, "ffn", i), f.res_mult, f.res_shift)),
            ops=operands(src=("x1", i * 2 * d), a2=("ffn", i * d), arg=1, dst=("x2", i * 2 * d), len=d // t.lanes))


def global_program(cfg, c: L.GlobalConsts | None, spec: TileSpec, mm: MemoryMap, pos: int, t: Timing = Timing(),
                   chunk: int = 1, first: bool = False) -> list[Step]:
    """One token through a global layer at position ``pos``, or a chunk of
    ``chunk`` consecutive tokens from ``pos`` (prefill): the passes carry
    the chunk, everything else takes the tokens in turn.  Inputs in the
    environment: ``x`` (``[chunk, d]`` for a chunk), and the memory side as
    ``k_rows``/``v_rows`` ``[kv_heads, N, hd]`` (the window then the
    retrieved blocks, as the record reader would stream them; a list per
    token for a chunk).  Every step carries its operands for the layer
    engine (``global_layout``); the memory steps name the context's memory
    image ``m_ctx`` by page and the position."""
    nh, nkv, hd, rd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.rotary_dim
    d, ffn, group, idim = cfg.hidden_size, cfg.layer_intermediate_size(cfg.global_layer_offset), nh // nkv, cfg.index_dim
    lay = global_layout(cfg, spec, mm, chunk)
    off_k, off_v, off_iq, off_ik, p1 = lay["off_k"], lay["off_v"], lay["off_iq"], lay["off_ik"], lay["p1"]
    rows_bytes, sel_bytes = lay["rows"], lay["sel"]
    ctx = ("m_ctx", 0, MEM_PAGE_SHIFT)                     # the context's memory page, in a3
    rec_beats = mm.kv_record_bytes // BEAT                 # a key or value record's beats
    get, put = _chunk_env(chunk)
    T = chunk
    steps: list[Step] = []

    def add(name, unit, src, dst, cycles, func=None, engine=0, nbytes=0, ops=None, port=(0, False)):
        steps.append(Step(name, unit, engine, tuple(src), tuple(dst), int(cycles), func, nbytes, ops=ops, port=port[0], posted=port[1]))

    def tok(name: str, i: int) -> str:
        return name if chunk == 1 else f"{name}<{i}>"

    for i in range(T):
        add(tok("norm.h", i), "norm", ("x",), (_contrib("A", chunk),), t.norm(d), lambda e, i=i: put(e, "A", i, L._norm(get(e, "x", i), c.norm)),
            ops=operands(src=("x", i * 2 * d), dst=("A", i * d), arg=NORM_RESIDUAL | NORM_INT16, len=d // t.lanes))

    def qkv(e):                                         # one buffer P1 per token: q, gate | k | v | index_q | index_k
        for i in range(T):
            a = get(e, "A", i)
            qg = L._fabric(c.q_proj, a, spec)[1].reshape(nh, 2 * hd)
            put(e, "q_raw", i, qg[:, :hd])
            put(e, "gate", i, qg[:, hd:])
            put(e, "k_raw", i, L._fabric(c.k_proj, a, spec)[1].reshape(nkv, hd))
            put(e, "v", i, L._fabric(c.v_proj, a, spec)[1].reshape(nkv, hd))
            put(e, "index_q", i, L._fabric(c.index_q, a, spec)[1])
            put(e, "index_k", i, L._fabric(c.index_k, a, spec)[1])
    walk = pass_walk(cfg, spec, recurrent=False)
    add("pass.qkv", "tiles", ("A",), ("P1",), t.tile_pass(spec, -(-d // spec.rows), T, walk[0][0] + T * walk[0][1]), qkv,
        ops=operands(src=("A", 0), dst=("P1", 0), arg=[(0, 0), (8, -(-d // spec.rows)), (16, T)], a2=d, a3=p1))
    for i in range(T):
        add(tok("rotary.table", i), "rotary", (), (_contrib("rot", chunk),), rd // 2 + t.rotary_table_latency,
            lambda e, i=i: put(e, "rot", i, L.rotary_table_int(pos + i, c.inv_freq)),
            ops=operands(dst=("rot", i * 2 * rd), arg=ROT_TABLE, a3=pos + i, len=rd // 2))
        add(tok("norm.index_q", i), "norm", ("P1",), (_contrib("iq", chunk),), t.norm(idim),
            lambda e, i=i: put(e, "index_q_unit", i, L._norm(get(e, "index_q", i), c.unit_norm)),
            ops=operands(src=("P1", i * p1 + off_iq), dst=("iq", i * idim), arg=NORM_UNIT, len=idim // t.lanes))
    rot_l = t.head_lanes(hd)                               # the rotary streams a head, like the cores
    rot_cycles = 2 * (hd // rot_l) + t.norm_latency + hd // rot_l + t.rotary_latency
    for i in range(T):
        for n in range(nkv):
            def krot(e, n=n, i=i):
                put(e, f"k[{n}]", i, L.rotary_int(L._norm(get(e, "k_raw", i)[n], c.k_norm), *get(e, "rot", i), rd, c.rot_mult_k, c.rot_sh_k))
            add(tok(f"rotary.k[{n}]", i), "rotary", ("P1", "rot"), ("+k",), rot_cycles, krot, engine=n % UNITS["rotary"][1],
                ops=operands(src=("P1", i * p1 + off_k + n * hd), dst=("k", i * nkv * hd + n * hd), arg=ROT_HEAD | (1 << 4),
                             a2=("rot", i * 2 * rd), len=hd // rot_l))
        for h in range(nh):
            def qrot(e, h=h, i=i):
                put(e, f"q[{h}]", i, L.rotary_int(L._norm(get(e, "q_raw", i)[h], c.q_norm), *get(e, "rot", i), rd, c.rot_mult_q, c.rot_sh_q))
            add(tok(f"rotary.q[{h}]", i), "rotary", ("P1", "rot"), (f"+qg[{h // group}]",), rot_cycles, qrot, engine=h % UNITS["rotary"][1],
                ops=operands(src=("P1", i * p1 + h * 2 * hd), dst=(f"qg[{h // group}]", i * group * hd + (h % group) * hd), arg=ROT_HEAD,
                             a2=("rot", i * 2 * rd), len=hd // rot_l))
    # The memory side, token by token: append this token's records, scan the index, then stream rows to the cores.
    for i in range(T):
        p = pos + i
        append_bytes = nkv * mm.kv_record_bytes + (nkv * mm.kv_record_bytes + mm.index_record_bytes) // mm.block + 2 * mm.sums_bytes
        add(tok("mem.append", i), "mem", ("k", "P1"), (), t.append(nkv, rec_beats, (p + 1) % mm.block == 0, first and p == 0,
                                                         -(-mm.sums_bytes // BEAT), -(-mm.index_record_bytes // BEAT)), nbytes=append_bytes,
            ops=operands(src=("k", i * nkv * hd), dst=("P1", i * p1 + off_ik), a2=("P1", i * p1 + off_v), a3=ctx,
                         arg=[(0, MEM_APPEND), (4, p)]))
        # The scan reads the index a page of records per request; the rows are the
        # window (head-major, page bursts) and one mean record per selected block.
        eligible = eligible_blocks(p, mm.local_window, mm.block)
        scan_bytes = eligible * mm.index_record_bytes
        add(tok("mem.scan", i), "mem", ("iq",), (_contrib("sel", chunk),), t.scan(eligible, mm.index_record_bytes // BEAT, min(cfg.top_blocks, eligible), mm.index_burst_records),
            lambda e, i=i: put(e, "selected", i, None), nbytes=scan_bytes,
            ops=operands(src=("iq", i * idim), dst=("sel", i * sel_bytes), a3=ctx, arg=[(0, MEM_SCAN), (4, p)]))
        n_window, n_blocks = min(p + 1, mm.local_window), min(cfg.top_blocks, eligible)
        rows = n_window + n_blocks
        for n in range(nkv):
            heads = list(range(n * group, (n + 1) * group))
            window_bytes, block_bytes = n_window * mm.kv_record_bytes, n_blocks * mm.kv_record_bytes

            def mem_rows(e, n=n, i=i):
                k_rows, v_rows = (e["k_rows"][i], e["v_rows"][i]) if chunk > 1 else (e["k_rows"], e["v_rows"])
                put(e, f"rows[{n}]", i, (k_rows[n], v_rows[n]))
            add(tok(f"mem.rows[{n}]", i), "mem", ("sel",), (_contrib(f"rows[{n}]", chunk),),
                t.read_records(window_requests(p, mm.local_window, mm.window_burst_records) + [1] * n_blocks, rec_beats, hd,
                               mm.window_burst_records),
                mem_rows, nbytes=window_bytes + block_bytes,
                ops=operands(src=("sel", i * sel_bytes), dst=(f"rows[{n}]", i * rows_bytes), a2=n, a3=ctx, arg=[(0, MEM_ROWS), (4, p)], len=rows))

            def attn(e, n=n, heads=heads, i=i):
                q = np.stack([get(e, f"q[{h}]", i) for h in heads])
                k_rows, v_rows = get(e, f"rows[{n}]", i)
                put(e, f"att[{n}]", i, L.attention_int(q, get(e, "gate", i)[heads], k_rows, v_rows, mult_s=c.mult_s, sh_s=c.sh_s,
                                                       mult_gate=c.mult_gate, sh_gate=c.sh_gate, mult_o=c.mult_o, sh_o=c.sh_o))
            add(tok(f"attn[{n}]", i), "attn", (f"qg[{n}]", "P1", f"rows[{n}]"), ("+att",),
                t.attention(rows, group, hd), attn, engine=n % UNITS["attn"][1],
                ops=operands(src=(f"qg[{n}]", i * group * hd), dst=("att", i * nh * hd + n * group * hd),
                             a2=("P1", i * p1 + n * group * 2 * hd + hd), a3=(f"rows[{n}]", i * rows_bytes), len=rows))

    def o_proj(e):
        for i in range(T):
            att = np.concatenate([get(e, f"att[{n}]", i) for n in range(nkv)]).reshape(nh * hd)
            put(e, "mixer", i, L._fabric(c.o_proj, att, spec)[1])
    add("pass.o_proj", "tiles", ("att",), ("mixer",), t.tile_pass(spec, -(-(nh * hd) // spec.rows), T, walk[1][0] + T * walk[1][1]), o_proj,
        ops=operands(src=("att", 0), dst=("mixer", 0), arg=[(0, 1), (8, -(-(nh * hd) // spec.rows)), (16, T)], a2=nh * hd, a3=d))
    for i in range(T):
        add(tok("residual.1", i), "residual", ("x", "mixer"), (_contrib("x1", chunk),), t.beats(d) + t.residual_latency,
            lambda e, i=i: put(e, "x1", i, L.residual_int(get(e, "x", i), get(e, "mixer", i), c.res_mult, c.res_shift)),
            ops=operands(src=("x", i * 2 * d), a2=("mixer", i * d), arg=0, dst=("x1", i * 2 * d), len=d // t.lanes))
    _ffn_steps(add, c.ffn if c is not None else None, spec, d, ffn, t, walk, chunk)
    link(steps)
    return steps


def global_layout(cfg, spec: TileSpec, mm: MemoryMap, chunk: int = 1) -> dict:
    """The global program's buffers as the layer engine holds them: byte
    sizes of every vector-buffer name and of the context's memory image
    ``m_ctx`` (the map's window, block store, index and block sums), the
    offsets inside one token's pass-1 output ``P1`` (``q, gate per head | k
    | v | index_q | index_k``) and its size ``p1``, and one token's bytes of
    ``sel`` and of a head's ``rows``.  ``rot`` holds the sines then the
    cosines of the rotary frequencies as int16; ``sel`` the count then the
    ids of the selected blocks as int16; ``rows[n]`` the head's key and
    value records as int8.  A chunk's buffers hold its tokens' vectors in
    turn."""
    nh, nkv, hd, rd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.rotary_dim
    d, ffn, group, idim = cfg.hidden_size, cfg.layer_intermediate_size(cfg.global_layer_offset), nh // nkv, cfg.index_dim
    align = lambda n: -(-n // BEAT) * BEAT
    off_k = nh * 2 * hd
    off_v = off_k + nkv * hd
    off_iq = off_v + nkv * hd
    off_ik = off_iq + idim
    p1 = align(off_ik + idim)
    sel = 2 * (1 + cfg.top_blocks)
    rows = (mm.local_window + cfg.top_blocks) * 2 * hd
    T = chunk
    sizes = {"x": T * 2 * d, "A": T * d, "P1": T * p1, "rot": T * 2 * rd, "iq": T * idim, "k": T * nkv * hd,
             "sel": T * sel, "att": T * nh * hd, "mixer": T * d, "x1": T * 2 * d, "A2": T * d, "GU": T * 2 * ffn, "act": T * ffn,
             "ffn": T * d, "x2": T * 2 * d, "m_ctx": mm.context_bytes}
    for n in range(nkv):
        sizes[f"qg[{n}]"] = T * group * hd
        sizes[f"rows[{n}]"] = T * rows
    return {"sizes": sizes, "off_k": off_k, "off_v": off_v, "off_iq": off_iq, "off_ik": off_ik, "p1": p1, "sel": sel, "rows": rows}


# --------------------------------------------------------------------------
# Streams of tokens
# --------------------------------------------------------------------------

def _shared(name: str) -> bool:
    return _plain(name).startswith(SHARED_PREFIX)


def _renamed(name: str, token: int) -> str:
    return name if _shared(name) else f"{name}@{token}"


def retarget(steps: list[Step], token: int) -> list[Step]:
    """A copy of a token's program with its private buffers renamed for that token."""
    out = []
    for s in steps:
        ops = None if s.ops is None else {k: _rename_value(v, token) for k, v in s.ops.items()}
        out.append(Step(s.name, s.unit, s.engine, tuple(_renamed(n, token) for n in s.src),
                        tuple(_renamed(n, token) for n in s.dst), s.cycles, s.func, s.nbytes, token, ops=ops, port=s.port, posted=s.posted))
    return out


class _View(dict):
    """The environment as one token sees it: its private names carry the token suffix."""

    def __init__(self, env: dict, token: int) -> None:
        super().__init__()
        self.env, self.token = env, token

    def _key(self, name: str) -> str:
        return _renamed(name, self.token)

    def __getitem__(self, name: str):
        return self.env[self._key(name)]

    def __setitem__(self, name: str, value) -> None:
        self.env[self._key(name)] = value

    def update(self, pairs) -> None:                    # type: ignore[override]
        for name, value in dict(pairs).items():
            self[name] = value


def interleave(programs: list[list[Step]]) -> list[Step]:
    """Merge token programs (already retargeted) into one issue order: at
    each pick, the program whose next step could issue earliest under the
    controller's rules goes next, the older token on a tie.  The merged
    list carries its dependencies.

    The state slots are one set of physical buffers for every program.
    Dependencies only order a step against what is already merged, so
    without more one program's read into a slot can land between another's
    read of it and that program's update: its update then runs on the wrong
    context's state.  In the in-order program at the 9B geometry a stream
    of three did this three times a token, and the small configurations the
    tests run never did.  A program holds a slot from its read into it to
    its write-back, and reads into a slot only when no older program will
    read into it again.  An older program then never waits on a younger
    one, so the oldest always moves and nothing deadlocks; a hold per slot
    alone deadlocks once programs read ahead."""
    lk = Linker()
    merged: list[Step] = []
    issue: list[int] = []
    end: list[int] = []
    free: dict[tuple[str, int], int] = {}
    ptr = [0] * len(programs)

    def slots(names) -> list[str]:
        return [_plain(n) for n in names if _plain(n).startswith(SHARED_PREFIX)]

    def reads_into(step: Step) -> list[str]:
        return slots(step.dst) if step.unit == "mem" else []

    def writes_back(step: Step) -> list[str]:
        return slots(step.src) if step.unit == "mem" else []
    future = [collections.Counter(n for st in prog for n in reads_into(st)) for prog in programs]
    owner: dict[str, int] = {}
    while any(p < len(prog) for p, prog in zip(ptr, programs)):
        best = None
        for j, prog in enumerate(programs):
            if ptr[j] >= len(prog):
                continue
            step = prog[ptr[j]]
            if any(owner.get(n, j) != j or any(future[i][n] for i in range(j)) for n in reads_into(step)):
                continue
            deps = lk.deps_for(step)
            t0 = issue[-1] + 1 if issue else 0
            for d in deps:
                t0 = max(t0, end[d] + 1)
            t0 = max(t0, free.get((step.unit, step.engine), 0))
            if best is None or (t0, j) < (best[0], best[1]):
                best = (t0, j, step, deps)
        t0, j, step, deps = best
        step.deps = deps
        merged.append(step)
        issue.append(t0)
        end.append(t0 + step.cycles)
        free[(step.unit, step.engine)] = t0 + step.cycles + 1
        lk.commit(step, len(merged) - 1)
        for n in reads_into(step):
            owner[n] = j
        future[j].subtract(reads_into(step))
        for n in writes_back(step):
            owner.pop(n, None)
        ptr[j] += 1
    return merged


def stream(program: list[Step], tokens: int) -> list[Step]:
    """`tokens` consecutive tokens of one layer, of different contexts, as one merged program."""
    return interleave([retarget(program, k) for k in range(tokens)])


def run_program(steps: list[Step], inputs: dict) -> dict:
    """Execute the steps in program order on the integer model; returns the
    environment.  A stream's inputs carry the token suffix (``x@0``)."""
    env = dict(inputs)
    views: dict[int, _View] = {}
    for step in steps:
        if step.func is not None:
            step.func(env if step.token is None else views.setdefault(step.token, _View(env, step.token)))
    return env


# --------------------------------------------------------------------------
# Scheduling
# --------------------------------------------------------------------------

@dataclasses.dataclass
class Schedule:
    issue: list[int]
    end: list[int]
    steps: list[Step]
    release: list[int] = dataclasses.field(default_factory=list)   # the cycle each step's buffers came back

    @property
    def cycles(self) -> int:
        """What the engine counts: ``start`` to the cycle the controller
        reports done, which is the cycle after the last release drained --
        and the drain is registered, so one more after that."""
        return max(self.release) + 2 if self.release else 0

    @property
    def last_done(self) -> int:
        """The cycle the last command's completion arrives, counted from the
        first issue: what a testbench over stub units sees."""
        return max(self.end) + 1 if self.end else 0

    def busy(self, unit: str) -> int:
        return sum(s.cycles for s in self.steps if s.unit == unit)

    def report(self, t: Timing = Timing()) -> str:
        total = self.cycles
        lines = [f"{total} cycles per token ({total / t.core_mhz:.1f} us at {t.core_mhz:.0f} MHz)",
                 "| Unit | Busy cycles | Share |", "| --- | ---: | ---: |"]
        for unit in UNITS:
            b = self.busy(unit)
            if b:
                lines.append(f"| {unit} | {b} | {100 * b / total / UNITS[unit][1]:.0f}% |")
        traffic = sum(s.nbytes for s in self.steps)
        lines.append(f"\nmemory traffic {traffic / 1024:.0f} KB")
        return "\n".join(lines)


def schedule(steps: list[Step], releases: int = RELEASES) -> Schedule:
    """The controller's issue rules, cycle for cycle (rtl/fabric_sequencer.sv).

    In program order: the head step issues at the earliest cycle after the
    previous issue at which every dependency's buffers have been released
    and the addressed engine is free.  A unit that stops working at cycle
    ``e`` reports done at ``e + 1``, and the controller *drains* at most
    ``releases`` completions a cycle, lowest engine port first.  The drain is
    registered and the counters are registers, so a release picked at ``c`` is
    applied at ``c + 1``, readable at ``c + 2``, and a dependant issues there
    rather than in the drain cycle: picking the port, reading its ids and
    moving the counters by them did not fit in one cycle
    (rtl/fabric_sequencer.sv).  The port itself is free at ``c + 1``, because
    the ids were registered out of its slot when the drain was picked.  With
    nothing
    else waiting a step issues one cycle after its last dependency ended,
    which is what an unbounded release gave.  A port whose release has not
    drained holds the buffers it must return and can be given no new
    command, which is what makes the bound safe.

    The loop steps from event to event rather than cycle by cycle: state
    changes only when a unit reports done, when a held release drains, or
    when the head step issues.
    """
    n = len(steps)
    issue, end = [0] * n, [0] * n
    release: list[int | None] = [None] * n                   # the cycle each step's buffers came back
    port_free: dict[int, int] = {}                           # engine port -> the cycle it may be given a command again
    port_of = [UNITS[s.unit][0] * NE + s.engine for s in steps]
    running: dict[int, int] = {}                             # engine port -> the step on it
    pending: dict[int, int] = {}                             # engine port -> a step whose release is held
    cycle, i = 0, 0
    path_free = 0                                            # the memory path as built: the cycle it is idle
    while i < n or running or pending:
        # A unit that ended at cycle - 1 reports done now; the drains this
        # cycle are the lowest ports of what is held and what just arrived.
        want = dict(pending)
        for port, s in list(running.items()):
            if end[s] + 1 == cycle:
                want[port] = s
                del running[port]
        for port in sorted(want)[:releases]:
            release[want.pop(port)] = cycle
            # The drain applies the cycle after it is picked, and the port is
            # free in that cycle -- the ids it returns were registered out of
            # the slot when it was picked, so the slot may be written then.
            port_free[port] = cycle + 1
        pending = want
        if (i < n and (i == 0 or cycle > issue[i - 1])
                and port_of[i] not in pending and port_of[i] not in running
                and cycle >= port_free.get(port_of[i], 0)
                and all(release[d] is not None and release[d] + 2 <= cycle for d in steps[i].deps)):
            # ``cycles`` is what the engine's own spans measure: the cycle the
            # command issued to the cycle its completion arrived.  The unit
            # therefore stops working one before that, and reports done at it.
            issue[i], end[i] = cycle, cycle + steps[i].cycles - 1
            st = steps[i]
            if st.port:
                # The memory path as built takes one request at a time: a
                # read waits for the one before it, and a posted write's data
                # is taken once the path is free (bar what its queues hold).
                start = max(cycle, path_free)
                if st.posted:
                    # A write the bridge's queues hold is taken at once; a
                    # longer one is taken as the path frees.
                    if st.port > 0 and st.cycles > POSTED_SLACK:
                        end[i] = max(end[i], start + st.cycles - 1 - POSTED_SLACK)
                    path_free = start + st.port
                else:
                    end[i] = start + st.cycles - 1
                    path_free = end[i] + 1
            running[port_of[i]] = i
            i += 1
        # The next cycle anything can happen: a drain, a completion, or --
        # when the head waits only on its turn -- the cycle after the last issue.
        ahead = [cycle + 1] if pending else []
        if running:
            ahead.append(min(end[s] for s in running.values()) + 1)
        if (i < n and port_of[i] not in pending and port_of[i] not in running
                and all(release[d] is not None for d in steps[i].deps)):
            # Every candidate here must be past `cycle`, or the loop below
            # stops advancing.  A dependency that released *at* this cycle is
            # not yet visible -- the drain is registered -- so the earliest
            # the head can go is the cycle after the last of them.
            earliest = cycle + 1 if i == 0 else issue[i - 1] + 1
            earliest = max(earliest, port_free.get(port_of[i], 0))
            for d in steps[i].deps:
                earliest = max(earliest, release[d] + 2)
            ahead.append(earliest)
        cycle = min(ahead) if ahead else cycle + 1
    return Schedule(issue, end, steps, [r if r is not None else 0 for r in release])


def token_interval(program: list[Step], tokens: int = 3) -> int:
    """The steady-state cycles per token of a stream: the last token's cost on top of the others."""
    a = schedule(stream(program, tokens)).cycles
    b = schedule(stream(program, tokens - 1)).cycles
    return a - b


# --------------------------------------------------------------------------
# The program image for the controller
# --------------------------------------------------------------------------

def buffer_ids(steps: list[Step]) -> dict[str, int]:
    ids: dict[str, int] = {}
    for step in steps:
        for name in step.src + tuple(_plain(n) for n in step.dst):
            if name not in ids:
                ids[name] = len(ids)
    assert len(ids) <= MAX_IDS, f"{len(ids)} buffers, the program word holds {MAX_IDS}"
    return ids


def encode(steps: list[Step], layout=None) -> list[int]:
    """Each step as a 256-bit word: unit, engine, last, the length the unit
    is given, a 32-bit argument, four 30-bit address operands (source,
    destination, two more), then up to six consumed buffer ids, two
    produced ids and their contribution bits (0xFF is no buffer).  Without
    a layout the length is the step's cycles and the operand fields are
    zero (the stub units); with one (``engine.Layout``) the fields are the
    step's operands.  A stub unit takes its duration from ``arg`` rather
    than the length, because a full-size command can run longer than the
    length field is wide -- one attention core streams tens of thousands of
    context rows -- while a real command's length is always its beats."""
    ids = buffer_ids(steps)
    words = []
    for i, step in enumerate(steps):
        assert step.cycles < (1 << 32) and step.engine < 16
        ops = step.ops if layout is not None and step.ops is not None else {}
        length = resolve_value(ops["len"], layout) if "len" in ops else min(step.cycles, 0xFFFF)
        fields = {k: resolve_value(ops.get(k), layout) for k in ("arg", "src", "dst", "a2", "a3")}
        if layout is None:
            fields["arg"] = step.cycles
        elif step.unit == "mem":
            # Which token in flight the command is for: the engine adds that
            # token's slot page to its memory addresses.
            token = step.token or 0
            assert token < MAX_TOKENS and fields["a3"] < (1 << SLOT_TOKEN_SHIFT), step.name
            fields["a3"] |= token << SLOT_TOKEN_SHIFT
        assert length < (1 << 16) and fields["arg"] < (1 << 32), step.name
        assert all(fields[k] < (1 << ADDR_BITS) for k in ("src", "dst", "a2", "a3")), step.name
        w = UNITS[step.unit][0] | (step.engine << 4) | (int(i == len(steps) - 1) << 8) | (length << 16)
        w |= fields["arg"] << 32
        for k, name in enumerate(("src", "dst", "a2", "a3")):
            w |= fields[name] << (64 + ADDR_BITS * k)
        consume = [ids[n] for n in step.src] + [0xFF] * (MAX_CONSUME - len(step.src))
        produce = [ids[_plain(n)] for n in step.dst] + [0xFF] * (MAX_PRODUCE - len(step.dst))
        contrib = sum(int(n.startswith("+")) << k for k, n in enumerate(step.dst))
        base = 64 + 4 * ADDR_BITS
        for k, c in enumerate(consume):
            w |= c << (base + 8 * k)
        for k, p in enumerate(produce):
            w |= p << (base + 8 * MAX_CONSUME + 8 * k)
        w |= contrib << (base + 8 * MAX_CONSUME + 8 * MAX_PRODUCE)
        assert w < (1 << 256)
        words.append(w)
    return words


def emit_program(directory: Path, steps: list[Step]) -> dict:
    """Write ``program.hex`` and the schedule the RTL must reproduce; returns the testbench parameters."""
    write_hex(directory / "program.hex", encode(steps), 256)
    sched = schedule(steps)
    (directory / "expected_issue.txt").write_text("".join(f"{i} {a} {b}\n" for i, (a, b) in enumerate(zip(sched.issue, sched.end))))
    params = {"N": len(steps), "EXPECTED_CYCLES": sched.last_done}
    for name, (uid, engines) in UNITS.items():
        params[f"E{uid}"] = engines
    (directory / "params.json").write_text(json.dumps(params))
    return params


def check_trace(steps: list[Step], trace: str) -> list[str]:
    """Check a testbench trace (``tag unit engine issue done`` per line)
    against the program: every dependency ended before the issue, no engine
    ran two steps at once, and the program order was kept.  Tags are the
    step index modulo 256; the trace is in completion order."""
    rows = [tuple(int(v) for v in line.split()) for line in trace.strip().splitlines() if line.strip()]
    problems = []
    if len(rows) != len(steps):
        return [f"{len(rows)} trace rows for {len(steps)} steps"]
    # Issue order recovers the absolute index: the k-th issue is step k.
    by_issue = sorted(rows, key=lambda r: r[3])
    by_index = {}
    for k, (tag, unit, engine, t_issue, t_done) in enumerate(by_issue):
        if tag != k % 256:
            problems.append(f"issue {k} carried tag {tag}")
        by_index[k] = (unit, engine, t_issue, t_done)
    for i, step in enumerate(steps):
        unit, engine, t_issue, t_done = by_index[i]
        if unit != UNITS[step.unit][0] or engine != step.engine:
            problems.append(f"step {i} ran on unit {unit} engine {engine}")
        for d in step.deps:
            if by_index[d][3] > t_issue:
                problems.append(f"step {i} issued at {t_issue} before dependency {d} ended at {by_index[d][3]}")
        if i and by_index[i - 1][2] > t_issue:
            problems.append(f"step {i} issued before step {i - 1}")
    busy: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for tag, unit, engine, t_issue, t_done in rows:
        busy.setdefault((unit, engine), []).append((t_issue, t_done))
    for key, spans in busy.items():
        spans.sort()
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            if b0 < a1:
                problems.append(f"unit {key[0]} engine {key[1]} overlapped at {b0} < {a1}")
    return problems


# --------------------------------------------------------------------------
# The full-size picture
# --------------------------------------------------------------------------

def report_markdown(cfg, mm: MemoryMap, pos: int, t: Timing = Timing(), chunk: int = 8) -> str:
    """The two layers' schedules at a model's geometry, one token at a time, as a stream, and as a prefill chunk."""
    spec = TileSpec()
    out = [f"# Token sequencer schedule: hidden {cfg.hidden_size}, position {pos}", ""]
    for title, steps, chunked in (("Recurrent layer", recurrent_program(cfg, None, spec, mm, t), recurrent_program(cfg, None, spec, mm, t, chunk)),
                                  ("Global layer", global_program(cfg, None, spec, mm, pos, t), global_program(cfg, None, spec, mm, pos, t, chunk))):
        sched = schedule(steps)
        interval = token_interval(steps)
        pre = schedule(chunked)
        out += [f"## {title}: {len(steps)} steps", "", sched.report(t), "",
                f"stream of tokens: {interval} cycles per token ({interval / t.core_mhz:.1f} us), "
                f"{100 * sched.busy('mem') / interval:.0f}% of the memory port", "",
                f"prefill, chunk of {chunk}: {pre.cycles // chunk} cycles per token ({pre.cycles / chunk / t.core_mhz:.1f} us), "
                f"{sum(s.nbytes for s in chunked) // chunk // 1024} KB of memory traffic per token, {len(chunked)} steps", ""]
    return "\n".join(out)


def main() -> None:
    from fixed_llm_poc import ASICLMConfig
    cfg = ASICLMConfig.qwen3_5_9b()
    mm = MemoryMap.from_config(cfg)
    print(report_markdown(cfg, mm, mm.context_tokens - 1))
    print(report_markdown(cfg, mm, 4095))


if __name__ == "__main__":
    main()
