"""A layer die's front end: what ``rtl/fabric_die_link.sv`` does with the ring.

Packets come in from the die before, each a work item of one token or of a
chunk of a prompt (``controller.py``), and each takes a lane of the engine
as it arrives: the engine runs up to ``lanes`` of them at once, a lane each,
their steps interleaved as they are ready (``sequencer.schedule_lanes``).
There is no gathering: a packet waits only for a free lane, and for a lane
of its own context to finish -- two tokens of one context cannot run at once
on its state, and a prompt's packets come back to back from one slot.  A
packet whose CRC fails gives its lane back, and one the die has no program
for -- not a work item, a token count other than one or the chunk, a length
other than the vectors -- is dropped; both are counted.  Each packet leaves
with the engine's output vectors in place of its input, the header as it
came, when its lane is done: packets of different contexts may overtake one
another, a context's own never do.

The engine is given, per packet, a run for each of the die's layers in
turn, pushed to the packet's lane: the program for the layer's kind and the
packet's shape (its first step and length, from a table the die holds), the
layer, and where the layer's part of the slot starts; and with the first,
the lane's token: its slot as a page of the die's memory -- the packet's
context field is the controller's slot number, and a slot is ``slot_pages``
pages -- its FIRST flag and its position.

This is the model the gateware is checked against: which packets run,
what each one's lane is given, and what leaves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from fabric import controller as C


@dataclass(frozen=True)
class Push:
    """One run pushed to a lane: the program, the layer and its part of the
    slot, and the lane's token -- its FIRST, slot and position."""
    pc: int
    steps: int
    layer: int
    page: int
    first: bool
    slot: int
    position: int


@dataclass
class DieLink:
    d: int                          # hidden elements a token
    chunk: int = 1                  # the die's chunk, besides a single token
    lanes: int = 4
    slot_pages: int = 16
    page_base: int = 0
    layers: int = 1                 # a token's layers on this die, a run each
    crc_errors: int = 0
    malformed: int = 0

    def shape(self, packet: bytes) -> int | None:
        """The packet's token count if the die has a program for it, else None."""
        kind, _, _, _, tokens, length = C._unheader(packet)
        if kind != C.KIND_ITEM or tokens not in (1, self.chunk) or length != tokens * 2 * self.d:
            return None
        return tokens

    def admit(self, packets: Sequence[bytes]) -> list[C.WorkItem]:
        """The work items that run, in the order they arrive; the rest are counted."""
        out = []
        for packet in packets:
            if self.shape(packet) is None:
                self.malformed += 1
                continue
            try:
                item, _ = C.unpack_item(packet, self.d)
            except ValueError:
                self.crc_errors += 1
                continue
            out.append(item)
        return out

    def slot_page(self, item: C.WorkItem) -> int:
        return self.page_base + item.context * self.slot_pages

    def pushes(self, item: C.WorkItem, table: Sequence[int], layer_table: Sequence[int] | None = None) -> list[Push]:
        """What the item's lane is given: a run a layer.  ``table`` is the
        program table, an entry per ``{kind, chunked}``; ``layer_table`` has
        each layer's kind in bit 0 and its part of the slot above it."""
        layer_table = layer_table or [0] * self.layers
        chunked = int(np.ndim(item.hidden) == 2)
        out = []
        for layer in range(self.layers):
            entry = table[((layer_table[layer] & 1) << 1) | chunked]
            out.append(Push(entry & 0xFFFF, (entry >> 16) & 0xFFFF, layer, layer_table[layer] >> 1, bool(item.flags & C.FLAG_FIRST),
                            self.slot_page(item), item.position))
        return out

    def forward(self, item: C.WorkItem, layer: Callable[[C.WorkItem, int, np.ndarray], np.ndarray]) -> bytes:
        """The item's packet as it leaves: its vectors through ``layer(item,
        k, hidden)`` for each of the die's layers k, the header unchanged."""
        hidden = np.asarray(item.hidden)
        for k in range(self.layers):
            hidden = np.asarray(layer(item, k, hidden), dtype=np.int64)
        return C.pack_item(C.WorkItem(item.context, item.position, hidden.astype(np.int16), item.flags))


def packets_of(words: Sequence[int]) -> list[bytes]:
    """Split a stream of link words into packets, by each header's length."""
    raw = b"".join(int(w).to_bytes(4, "little") for w in words)
    out, i = [], 0
    while i < len(raw):
        length = C._unheader(raw[i:])[5]
        n = C.HEADER_BYTES + length + C.TRAILER_BYTES
        out.append(raw[i:i + n])
        i += n
    return out


def order_problems(sent: Sequence[bytes], got: Sequence[bytes]) -> list[str]:
    """What left against what the model says leaves: the same packets, each
    context's in the order it came."""
    problems = []
    if sorted(sent) != sorted(got):
        problems.append(f"{len(got)} packets out, {len(sent)} expected, {len(set(got) ^ set(sent))} differ")
    for ctx in {C._unheader(p)[2] for p in sent}:
        mine = lambda ps: [p for p in ps if C._unheader(p)[2] == ctx]
        if mine(sent) != mine(got):
            problems.append(f"context {ctx}: its packets out of order")
    return problems


# --------------------------------------------------------------------------
# Vectors for tb_die_link
# --------------------------------------------------------------------------

def _stub_layer(entry: int):
    """What tb_die_link's stand-in engine does to a lane for a run: adds a
    constant that says which program ran and which layer it was."""
    def layer(item: C.WorkItem, k: int, hidden: np.ndarray) -> np.ndarray:
        e = entry(item, k)
        return ((hidden.astype(np.int64) + 1 + 7 * e + 3 * k + 32768) % 65536) - 32768
    return layer


def table(chunk: int, d: int, lanes: int = 4) -> list[int]:
    """A program table for the testbench, an entry per {kind, chunked}: each
    entry's first step and length distinct, and a lane's vectors in place --
    its input and output one buffer, the same in the entries of a shape, as
    the die's contract has it."""
    out = []
    for kind in (0, 1):
        for chunked in (0, 1):
            tokens = chunk if chunked else 1
            stride = -(-tokens * 2 * d // 16) * 16 + 16          # a beat apart, so a lane that overruns shows
            base = 0x1000 + chunked * 0x4000
            word = (100 * chunked + 10 * kind + 3) | ((5 + kind + 2 * chunked) << 16)
            for k in range(4):
                word |= (base + k * stride) << (32 + 24 * k)
                word |= (base + k * stride) << (128 + 24 * k)
            out.append(word)
    return out


def emit_die_link_vectors(directory, rng: np.random.Generator, d: int = 16, chunk: int = 3, lanes: int = 4, layers: int = 1) -> dict:
    """Packets for tb_die_link and what the model says of them: the items
    that run, each one's pushes, and the packets that leave."""
    from pathlib import Path
    from fabric.tile import write_hex
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    link = DieLink(d, chunk, lanes, slot_pages=37, page_base=5, layers=layers)
    layer_table = [((k % 2) | ((11 * k) << 1)) for k in range(layers)]      # kinds alternating, each its part of the slot
    tab = table(chunk, d)
    used: set[tuple[int, int]] = set()

    def fresh(context: int | None = None) -> tuple[int, int]:
        while True:                                         # a context and position no other packet has
            c = int(rng.integers(0, 200)) if context is None else context
            p = int(rng.integers(0, 1 << 20))
            if (c, p) not in used:
                used.add((c, p))
                return c, p

    def item(tokens: int, flags: int | None = None, context: int | None = None) -> bytes:
        hidden = rng.integers(-32768, 32767, size=(tokens, d) if tokens > 1 else d).astype(np.int16)
        f = int(rng.integers(0, 8)) if flags is None else flags
        c, p = fresh(context)
        return C.pack_item(C.WorkItem(c, p, hidden, f))

    def bad_crc(packet: bytes) -> bytes:
        raw = bytearray(packet)
        raw[C.HEADER_BYTES + 3] ^= 0x40
        return bytes(raw)

    def not_item(packet: bytes) -> bytes:
        raw = bytearray(packet[:-C.TRAILER_BYTES])
        raw[0] = 0x58
        return bytes(raw) + C.crc32(bytes(raw)).to_bytes(4, "little")

    packets = ([item(1, C.FLAG_FIRST)]                                        # a lone token
               + [item(1) for _ in range(6)]                                  # more than there are lanes
               + [item(1), item(chunk), item(1), item(chunk)]                 # shapes side by side
               + [item(1), bad_crc(item(1)), item(1, C.FLAG_FIRST | C.FLAG_SAMPLE)]     # a CRC failure gives its lane back
               + [not_item(item(1)), item(2 if chunk != 2 else 4), item(chunk, C.FLAG_FIRST)]   # dropped, dropped, a chunk
               + [item(chunk) for _ in range(5)]
               + [bad_crc(item(chunk)), item(1)]
               + [item(1, 0, 9), item(1, 0, 9), item(1, 0, 4), item(1, 0, 9)])  # one slot's prompt: one at a time
    items = link.admit(packets)
    chunked = lambda it: int(np.ndim(it.hidden) == 2)
    entry = lambda it, k: ((layer_table[k] & 1) << 1) | chunked(it)
    outs = [link.forward(it, _stub_layer(entry)) for it in items]
    ins = [int.from_bytes(p[j:j + 4], "little") for p in packets for j in range(0, len(p), 4)]
    expected = []                                  # {pc, steps, layer, page, first, slot, position} a push, per item in order
    for it in items:
        for p in link.pushes(it, tab, layer_table):
            expected.append(p.pc | (p.steps << 16) | (p.layer << 32) | (p.page << 34) | (int(p.first) << 55) | (p.slot << 56)
                            | (p.position << 77))
    write_hex(directory / "die_table.hex", tab, 256)
    write_hex(directory / "die_layers.hex", layer_table, 32)
    write_hex(directory / "in.hex", ins, 32)
    write_hex(directory / "pk.hex", [len(p) // 4 for p in packets], 16)
    (directory / "model.txt").write_text("".join(f"{w:x}\n" for w in expected))
    from fabric.controller import _write_params
    params = _write_params(directory, D=d, CHUNK=chunk, LANES=lanes, SLOT_PAGES=37, PAGE_BASE=5, LAYERS=layers, PACKETS=len(packets),
                           IWORDS=len(ins), OWORDS=sum(len(p) // 4 for p in outs), CRC_ERRORS=link.crc_errors, MALFORMED=link.malformed)
    return {"params": params, "items": items, "outs": outs, "pushes": expected}


def check_die_link_run(directory, model: dict) -> list[str]:
    """tb_die_link's record against the model: every item's pushes, in its
    lane, while no other lane held its context; and every packet out."""
    from pathlib import Path
    directory = Path(directory)
    problems = []
    rows = [line.split() for line in (directory / "pushes.txt").read_text().splitlines() if line.strip()]
    lanes: dict[int, list] = {}
    runs: list[tuple[int, int, int, int]] = []           # (lane, first push cycle, done cycle, slot)
    got = []
    for r in rows:
        if r[0] == "P":
            cycle, lane, word = int(r[1]), int(r[2]), int(r[3], 16)
            lanes.setdefault(lane, []).append((cycle, word))
        else:
            cycle, lane = int(r[1]), int(r[2])
            pushed = lanes.pop(lane, [])
            if not pushed:
                problems.append(f"lane {lane} done at {cycle} with nothing pushed")
                continue
            runs.append((lane, pushed[0][0], cycle, (pushed[0][1] >> 56) & ((1 << 21) - 1)))
            got.append([w for _, w in pushed])
    for lane, pushed in lanes.items():
        problems.append(f"lane {lane} never done")
    # Each item's pushes, as the model has them; the items in their lanes' order of starting.
    want = [model["pushes"][i:i + len(model["pushes"]) // max(1, len(model["items"]))]
            for i in range(0, len(model["pushes"]), len(model["pushes"]) // max(1, len(model["items"])))]
    order = sorted(range(len(runs)), key=lambda i: runs[i][1])
    if sorted(map(tuple, got)) != sorted(map(tuple, want)):
        problems.append(f"{len(got)} lanes' runs, {len(want)} expected; they differ")
    elif [tuple(got[i]) for i in order] != [tuple(w) for w in want]:
        problems.append("the items did not start in the order they came")
    for i, (la, a0, a1, sa) in enumerate(runs):
        for lb, b0, b1, sb in runs[i + 1:]:
            if sa == sb and a0 < b1 and b0 < a1:
                problems.append(f"slot page {sa} in lanes {la} and {lb} at once")
    words = [int(w, 16) for w in (directory / "out_words.hex").read_text().split()]
    problems += order_problems(model["outs"], packets_of(words))
    return problems
