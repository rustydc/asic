"""A layer die's front end: what ``rtl/fabric_die_link.sv`` does with the ring.

Packets come in from the die before, each a work item of one token or of a
chunk of a prompt (``controller.py``), and are gathered into batches of up
to ``lanes``: the engine runs a program over a batch, its packets' tokens
interleaved.  A batch is one shape, a token a lane or a chunk a lane, since
a program is compiled for one; it closes when it is full, when a packet of
the other shape arrives, or when the link goes quiet.  A packet whose CRC
fails takes no lane, and one the die has no program for -- not a work item,
a token count other than one or the chunk, a length other than the vectors
-- is dropped; both are counted.  Each lane's packet leaves with the
engine's output vectors in place of its input, the header as it came.

The engine is given, per batch, the program for the batch's shape and size
(its first step and length, from a table the die holds), each lane's slot
as a page of the die's memory -- the packet's context field is the
controller's slot number, and a slot is ``slot_pages`` pages -- and each
lane's FIRST flag.

This is the model the gateware is checked against: which packets make
which batch, what each batch starts the engine with, and what leaves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from fabric import controller as C


@dataclass
class Batch:
    chunked: bool
    items: list[C.WorkItem] = field(default_factory=list)

    @property
    def entry(self) -> int:
        """The program table's entry: {chunked, lanes - 1}."""
        return (int(self.chunked) << 2) | (len(self.items) - 1)


@dataclass
class DieLink:
    d: int                          # hidden elements a token
    chunk: int = 1                  # the die's chunk, besides a single token
    lanes: int = 4
    slot_pages: int = 16
    page_base: int = 0
    crc_errors: int = 0
    malformed: int = 0

    def shape(self, packet: bytes) -> int | None:
        """The packet's token count if the die has a program for it, else None."""
        kind, _, _, _, tokens, length = C._unheader(packet)
        if kind != C.KIND_ITEM or tokens not in (1, self.chunk) or length != tokens * 2 * self.d:
            return None
        return tokens

    def batches(self, groups: Sequence[Sequence[bytes]]) -> list[Batch]:
        """The batches a stream of packets makes.  Each group is sent back to
        back and followed by quiet, which closes whatever batch is open."""
        out: list[Batch] = []
        for group in groups:
            cur: Batch | None = None
            for packet in group:
                tokens = self.shape(packet)
                if tokens is None:
                    self.malformed += 1
                    continue
                chunked = tokens != 1
                # The header decides: a packet that cannot join closes the batch, whatever its CRC.
                if cur is not None and cur.items and (len(cur.items) == self.lanes or cur.chunked != chunked):
                    out.append(cur)
                    cur = None
                try:
                    item, _ = C.unpack_item(packet, self.d)
                except ValueError:
                    self.crc_errors += 1
                    continue
                if cur is None or not cur.items:
                    cur = Batch(chunked)
                cur.items.append(item)
            if cur is not None and cur.items:
                out.append(cur)
        return out

    def slot_page(self, item: C.WorkItem) -> int:
        return self.page_base + item.context * self.slot_pages

    def first_mask(self, batch: Batch) -> int:
        return sum(1 << k for k, item in enumerate(batch.items) if item.flags & C.FLAG_FIRST)

    def forward(self, batch: Batch, layer: Callable[[Batch, int, np.ndarray], np.ndarray]) -> list[bytes]:
        """The batch's packets as they leave: lane k's vectors through
        ``layer(batch, k, hidden)``, the header otherwise unchanged."""
        out = []
        for k, item in enumerate(batch.items):
            hidden = np.asarray(layer(batch, k, np.asarray(item.hidden)), dtype=np.int64)
            out.append(C.pack_item(C.WorkItem(item.context, item.position, hidden.astype(np.int16), item.flags)))
        return out


# --------------------------------------------------------------------------
# Vectors for tb_die_link
# --------------------------------------------------------------------------

def _stub_layer(batch: Batch, k: int, hidden: np.ndarray) -> np.ndarray:
    """What tb_die_link's stand-in engine does to a lane: adds a constant
    that says which program ran and which lane it was."""
    return ((hidden.astype(np.int64) + 1 + 3 * k + 7 * batch.entry + 32768) % 65536) - 32768


def table(chunk: int, d: int, lanes: int = 4) -> list[int]:
    """A program table for the testbench: each entry's first step and length
    distinct, and a lane's input and output the same in every entry of a
    shape, as the die's contract has it."""
    out = []
    for chunked in (0, 1):
        tokens = chunk if chunked else 1
        stride = -(-tokens * 2 * d // 16) * 16 + 16          # a beat apart, so a lane that overruns shows
        base_in, base_out = 0x1000 + chunked * 0x4000, 0x2000 + chunked * 0x4000
        for n in range(1, 5):
            word = (100 * chunked + 10 * n) | ((5 + n) << 16)
            for k in range(4):
                word |= (base_in + k * stride) << (32 + 24 * k)
                word |= (base_out + k * stride) << (128 + 24 * k)
            out.append(word)
    return out


def emit_die_link_vectors(directory, rng: np.random.Generator, d: int = 16, chunk: int = 3, lanes: int = 4) -> dict:
    """Groups of packets for tb_die_link, the batches the model makes of
    them, the engine starts and the packets that leave."""
    from pathlib import Path
    from fabric.tile import write_hex
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    link = DieLink(d, chunk, lanes, slot_pages=37, page_base=5)

    def item(tokens: int, flags: int | None = None) -> bytes:
        hidden = rng.integers(-32768, 32767, size=(tokens, d) if tokens > 1 else d).astype(np.int16)
        f = int(rng.integers(0, 8)) if flags is None else flags
        return C.pack_item(C.WorkItem(int(rng.integers(0, 200)), int(rng.integers(0, 1 << 20)), hidden, f))

    def bad_crc(packet: bytes) -> bytes:
        raw = bytearray(packet)
        raw[C.HEADER_BYTES + 3] ^= 0x40
        return bytes(raw)

    def not_item(packet: bytes) -> bytes:
        raw = bytearray(packet[:-C.TRAILER_BYTES])
        raw[0] = 0x58
        return bytes(raw) + C.crc32(bytes(raw)).to_bytes(4, "little")

    groups = [
        [item(1, C.FLAG_FIRST)],                                   # a lone token: the quiet closes it
        [item(1) for _ in range(6)],                               # a full batch, then two
        [item(1), item(1), item(chunk), item(chunk)],              # the shape changes: two batches
        [item(1), bad_crc(item(1)), item(1, C.FLAG_FIRST | C.FLAG_SAMPLE)],   # a CRC failure takes no lane
        [not_item(item(1)), item(2 if chunk != 2 else 4), item(chunk, C.FLAG_FIRST)],   # dropped, dropped, a chunk
        [item(chunk) for _ in range(5)],                           # chunks: four and one
        [bad_crc(item(chunk)), item(1)],                           # a failed chunk opens no batch
    ]
    batches = link.batches(groups)

    ins, packets, outs, starts, after = [], [], [], [], []
    done_batches = 0
    for g, group in enumerate(groups):
        for i, packet in enumerate(group):
            words = [int.from_bytes(packet[j:j + 4], "little") for j in range(0, len(packet), 4)]
            ins += words
            packets.append((len(words), i == len(group) - 1))
        # The packets out once this group's batches have run.
        model = DieLink(d, chunk, lanes, 37, 5)
        done_batches = len(model.batches(groups[:g + 1]))
        after.append(done_batches)
    for b in batches:
        for packet in link.forward(b, _stub_layer):
            outs += [int.from_bytes(packet[j:j + 4], "little") for j in range(0, len(packet), 4)]
        pages = [link.slot_page(it) for it in b.items] + [link.page_base] * (4 - len(b.items))
        word = table(chunk, d)[b.entry] & 0xFFFFFFFF
        word |= link.first_mask(b) << 32
        for k, page in enumerate(pages):
            word |= page << (36 + 21 * k)
        starts.append(word)
    out_counts = []
    for g in range(len(groups)):
        out_counts.append(sum(len(C.pack_item(it)) // 4 for b in batches[:after[g]] for it in b.items))
    write_hex(directory / "die_table.hex", table(chunk, d), 256)
    write_hex(directory / "in.hex", ins, 32)
    write_hex(directory / "pk.hex", [n | (int(last) << 16) for n, last in packets], 20)
    write_hex(directory / "gout.hex", out_counts, 32)
    write_hex(directory / "out.hex", outs, 32)
    write_hex(directory / "starts.hex", starts, 128)
    from fabric.controller import _write_params
    return _write_params(directory, D=d, CHUNK=chunk, LANES=lanes, SLOT_PAGES=37, PAGE_BASE=5, PACKETS=len(packets),
                         GROUPS=len(groups), IWORDS=len(ins), OWORDS=len(outs), BATCHES=len(batches),
                         CRC_ERRORS=link.crc_errors, MALFORMED=link.malformed)
