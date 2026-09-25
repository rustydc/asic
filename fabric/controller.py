"""The controller: what the FPGA at the head of the ring does.

The appliance is a ring of dies behind one FPGA.  A work item -- one token
of one context: its hidden vector, the context's slot and the position --
enters the ring at the first layer die and comes back from the last head
die with two partial top-k lists appended, one from each half of the LM
head.  Everything between a host's request and the next token it gets is
the controller's: the embedding of the token into the hidden vector, the
packet that carries it, which context goes next, the merge of the two lists
and the draw from them, and the table that says which context holds which
slot in the dies' memories.  This module is the model of that, the way
``sequencer.py`` is the model of the token sequencer: the gateware will be
checked against it bit for bit where it is arithmetic (the sampler, the
CRC, the packet) and step for step where it is control (the scheduler).

Three contracts are fixed here that the dies must honour, since they are
the ring's protocol rather than the controller's alone:

* the packet: a 16-byte header, the hidden vector as int16 at the residual
  scale, and, after the head dies, their lists, all under one CRC-32;
* the head list: K candidates of (row, logit) with the logit a signed
  fixed-point value of ``LOGIT_FRAC`` fraction bits, and the log-sum-exp of
  the die's whole half in the same format, so the two halves can be merged
  into one distribution;
* the FIRST flag: the first token of a context in a slot that another
  context had, which tells a layer die to start that slot's state from
  zero rather than from what the slot holds.

The sampler is integer arithmetic on the exponential table the attention
core already uses (``layer.exp_neg_fixed``), so the same LUT serves both,
and the draw is defined on the random word, not on a distribution, so the
gateware's generator and this model give the same token for the same word.
"""

from __future__ import annotations

import math
import struct
import zlib
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from fabric.layer import FF, exp_neg_fixed

# --------------------------------------------------------------------------
# The packet
# --------------------------------------------------------------------------

KIND_ITEM = 0x57                 # a work item
# The CRC is a trailer, not a header field, because the link is a stream: a
# head die appending its list, and the sender of a packet whose payload is
# 8 KB, would both have to hold the whole thing to fill a header CRC in.
HEADER = struct.Struct("<BBHIHH")    # kind, flags, context, position, length, reserved
HEADER_BYTES = HEADER.size      # 12; with the 4-byte trailer, 16 of overhead
TRAILER_BYTES = 4
FLAG_SAMPLE = 0x01               # the controller wants a token from this item
FLAG_FIRST  = 0x02               # first token of a new context in this slot: zero the state
FLAG_LAST   = 0x04               # the context is done after this item: the slot may be reused

LOGIT_FRAC = FF                  # logits and log-sum-exps: signed, 2^-10 per LSB
LIST_HEADER = struct.Struct("<BBbBi")   # die, k, reserved, reserved, lse
ENTRY = struct.Struct("<Ii")            # row, logit


@dataclass(frozen=True)
class WorkItem:
    context: int
    position: int
    hidden: np.ndarray               # int16, the residual at scale s_h
    flags: int = 0


@dataclass(frozen=True)
class HeadList:
    """One head die's answer: its K best rows and the log-sum-exp of all of
    its rows, both as fixed point with LOGIT_FRAC fraction bits.  ``rows``
    are global row numbers: die d's local row r is r + d * rows_per_die."""
    die: int
    rows: np.ndarray                 # uint32
    logits: np.ndarray               # int32
    lse: int


def crc32(data: bytes) -> int:
    """The Ethernet CRC-32, as zlib computes it."""
    return zlib.crc32(data) & 0xFFFFFFFF


def pack_item(item: WorkItem, lists: Sequence[HeadList] = ()) -> bytes:
    """A work item as it travels the ring: header, hidden vector, and the
    lists the head dies have appended so far."""
    hidden = np.asarray(item.hidden, dtype="<i2").tobytes()
    tail = b"".join(pack_head_list(hl) for hl in lists)
    payload = hidden + tail
    head = HEADER.pack(KIND_ITEM, item.flags, item.context, item.position, len(payload), 0)
    return head + payload + struct.pack("<I", crc32(head + payload))


def pack_head_list(hl: HeadList) -> bytes:
    body = LIST_HEADER.pack(hl.die, len(hl.rows), 0, 0, int(hl.lse))
    return body + b"".join(ENTRY.pack(int(r), int(l)) for r, l in zip(hl.rows, hl.logits))


def append_head_list(packet: bytes, hl: HeadList) -> bytes:
    """What a head die does to a passing item: its list on the end, the length
    and the CRC brought up to date, nothing else touched."""
    kind, flags, context, position, length, _ = HEADER.unpack(packet[:HEADER_BYTES])
    payload = packet[HEADER_BYTES:HEADER_BYTES + length] + pack_head_list(hl)
    head = HEADER.pack(kind, flags, context, position, len(payload), 0)
    return head + payload + struct.pack("<I", crc32(head + payload))


def unpack_item(packet: bytes, d: int) -> tuple[WorkItem, list[HeadList]]:
    """The item and its lists, or a ValueError if the CRC does not hold."""
    kind, flags, context, position, length, _ = HEADER.unpack(packet[:HEADER_BYTES])
    payload = packet[HEADER_BYTES:HEADER_BYTES + length]
    if kind != KIND_ITEM:
        raise ValueError(f"not a work item: kind {kind:#x}")
    crc, = struct.unpack("<I", packet[HEADER_BYTES + length:HEADER_BYTES + length + TRAILER_BYTES])
    if crc32(packet[:HEADER_BYTES + length]) != crc:
        raise ValueError("CRC mismatch")
    hidden = np.frombuffer(payload[:2 * d], dtype="<i2").astype(np.int64)
    lists, at = [], 2 * d
    while at < len(payload):
        die, k, _, _, lse = LIST_HEADER.unpack(payload[at:at + LIST_HEADER.size])
        at += LIST_HEADER.size
        entries = [ENTRY.unpack(payload[at + i * ENTRY.size:at + (i + 1) * ENTRY.size]) for i in range(k)]
        at += k * ENTRY.size
        lists.append(HeadList(die, np.array([e[0] for e in entries], dtype=np.uint32),
                              np.array([e[1] for e in entries], dtype=np.int32), lse))
    return WorkItem(context, position, hidden, flags), lists


# --------------------------------------------------------------------------
# The head die's arithmetic, so a model of one can be exact
# --------------------------------------------------------------------------

def to_fixed(x) -> np.ndarray:
    """Float logits to the list's fixed point."""
    return np.rint(np.asarray(x, dtype=np.float64) * (1 << LOGIT_FRAC)).astype(np.int64)


def lse_fixed(logits_fixed: np.ndarray) -> int:
    """log-sum-exp of fixed-point logits, in the same fixed point."""
    x = np.asarray(logits_fixed, dtype=np.float64) / (1 << LOGIT_FRAC)
    m = float(x.max())
    return int(round((m + math.log(np.exp(x - m).sum())) * (1 << LOGIT_FRAC)))


def head_list(die: int, logits_fixed: np.ndarray, k: int, row_base: int = 0) -> HeadList:
    """What a head die returns for its half: the K largest, rows made global."""
    order = np.argsort(-logits_fixed, kind="stable")[:k]
    return HeadList(die, (order + row_base).astype(np.uint32), logits_fixed[order].astype(np.int32),
                    lse_fixed(logits_fixed))


# --------------------------------------------------------------------------
# Merge and sample
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SamplingParams:
    """Temperature as the reciprocal in F16 (``inv_t = 2^FF / T``); top_k the
    candidates kept; top_p in U16 (65535 is all of them).  Integers, because
    the gateware takes them as such."""
    inv_t: int = 1 << FF
    top_k: int = 32
    top_p: int = 0xFFFF

    @staticmethod
    def of(temperature: float = 1.0, top_k: int = 32, top_p: float = 1.0) -> "SamplingParams":
        return SamplingParams(int(round((1 << FF) / max(temperature, 1e-6))), top_k, min(0xFFFF, int(top_p * 0xFFFF)))


def merge_lists(lists: Sequence[HeadList]) -> tuple[np.ndarray, np.ndarray, int]:
    """The union of the head dies' candidates in descending order, and the
    log-sum-exp of the whole vocabulary from the halves' own."""
    rows = np.concatenate([hl.rows.astype(np.int64) for hl in lists])
    logits = np.concatenate([hl.logits.astype(np.int64) for hl in lists])
    order = np.lexsort((rows, -logits))                     # by logit descending, rows to break ties
    lses = [hl.lse / (1 << LOGIT_FRAC) for hl in lists]
    m = max(lses)
    total = int(round((m + math.log(sum(math.exp(v - m) for v in lses))) * (1 << LOGIT_FRAC)))
    return rows[order], logits[order], total


def weights(logits_desc: np.ndarray, inv_t: int) -> np.ndarray:
    """Unnormalised softmax weights over candidates in descending order:
    ``exp(-(max - l) / T)`` through the F16 exponential table, U16 each."""
    l = np.asarray(logits_desc, dtype=np.int64)
    t = ((l[0] - l) * int(inv_t)) >> FF                    # F16, >= 0
    return exp_neg_fixed(t).astype(np.int64)


def sample(rows_desc: np.ndarray, logits_desc: np.ndarray, params: SamplingParams, rnd: int) -> tuple[int, int]:
    """The token, and the index among the candidates it came from.

    ``rnd`` is a 32-bit word; the draw is ``(rnd * total) >> 32`` over the
    cumulative weights of the candidates top-p keeps, which is what the
    gateware does with a multiplier and a scan, so the same word gives the
    same token here and there."""
    k = min(params.top_k, len(rows_desc))
    w = weights(logits_desc[:k], params.inv_t)
    cum = np.cumsum(w)
    total = int(cum[-1])
    if total == 0:                                          # every weight underflowed but the first, which cannot: guard anyway
        return int(rows_desc[0]), 0
    keep = k
    if params.top_p < 0xFFFF:                               # the shortest prefix holding top_p of the mass
        keep = int(np.searchsorted(cum * 0x10000, params.top_p * total, side="left")) + 1
        keep = max(1, min(keep, k))
    total = int(cum[keep - 1])
    r = ((int(rnd) & 0xFFFFFFFF) * total) >> 32
    i = int(np.searchsorted(cum[:keep], r, side="right"))
    return int(rows_desc[i]), i


def logprob(logit_fixed: int, lse_total_fixed: int) -> float:
    """The sampled token's log-probability under the full softmax, for scoring."""
    return (int(logit_fixed) - int(lse_total_fixed)) / (1 << LOGIT_FRAC)


# --------------------------------------------------------------------------
# The embedding table
# --------------------------------------------------------------------------

class EmbeddingTable:
    """Token to hidden vector, int16 at the residual scale: the DDR4 behind the
    FPGA, ``vocab x hidden x 2`` bytes (2.03 GB for the 9B), one row a lookup."""

    def __init__(self, rows: np.ndarray) -> None:
        self.rows = np.asarray(rows, dtype=np.int16)

    @classmethod
    def quantize(cls, weight: np.ndarray, s_h: float) -> "EmbeddingTable":
        q = np.clip(np.rint(np.asarray(weight, dtype=np.float64) / s_h), -32768, 32767)
        return cls(q.astype(np.int16))

    @property
    def vocab(self) -> int:
        return self.rows.shape[0]

    @property
    def hidden(self) -> int:
        return self.rows.shape[1]

    @property
    def nbytes(self) -> int:
        return self.rows.nbytes

    def lookup(self, token: int) -> np.ndarray:
        return self.rows[int(token)].astype(np.int64)


# --------------------------------------------------------------------------
# Contexts and their slots
# --------------------------------------------------------------------------

@dataclass
class Context:
    id: int
    tokens: list[int]                # the prompt, then what was generated
    prompt_len: int
    max_new: int
    params: SamplingParams
    slot: int | None = None
    next_pos: int = 0                # the next position to inject
    in_flight: bool = False
    fresh: bool = True               # the slot has not seen this context yet
    done: bool = False
    logprobs: list[float] = field(default_factory=list)
    seed: int = 0
    stops: tuple[int, ...] = ()      # tokens that end a turn when sampled
    keep: bool = False               # resident between turns: its slot is kept until closed or evicted


class ContextTable:
    """Which context holds which slot of the dies' memories.  A slot is one
    resident context's state on every die; there are as many as the memory
    map gives.  Allocation takes a free slot, then the least recently used
    idle one, whose context is evicted: its next token, if it has one, will
    have to start over, which is the host's problem to avoid and the table's
    to report."""

    def __init__(self, slots: int) -> None:
        self.slots = slots
        self.holder: dict[int, int] = {}                 # slot -> context id
        self.lru: OrderedDict[int, None] = OrderedDict()  # slots by last use, oldest first
        self.evicted: list[tuple[int, int]] = []          # (context, slot)
        self.kept = False                                 # the last acquire gave the context the slot it already held

    def acquire(self, ctx: int, busy: Callable[[int], bool]) -> int | None:
        """A slot for ``ctx``: its own if it has one, else a free one, else the
        oldest whose holder is not in flight.  None if every slot is busy."""
        self.kept = False
        for slot, holder in self.holder.items():
            if holder == ctx:
                self.lru.move_to_end(slot)
                self.kept = True
                return slot
        free = [s for s in range(self.slots) if s not in self.holder]
        if free:
            slot = free[0]
        else:
            slot = next((s for s in self.lru if not busy(self.holder[s])), None)
            if slot is None:
                return None
            self.evicted.append((self.holder[slot], slot))
        self.holder[slot] = ctx
        self.lru[slot] = None
        self.lru.move_to_end(slot)
        return slot

    def release(self, ctx: int) -> None:
        for slot, holder in list(self.holder.items()):
            if holder == ctx:
                del self.holder[slot]
                self.lru.pop(slot, None)


# --------------------------------------------------------------------------
# The ring, as the controller sees it
# --------------------------------------------------------------------------

class Ring:
    """A pipeline of dies, one item each, advancing a stage per step: the
    layer dies transform the hidden vector, the head dies append their lists.
    ``layer`` and ``head`` are the dies' models, so a test can be exact."""

    def __init__(self, n_layer_dies: int, layer: Callable[[WorkItem], np.ndarray],
                 heads: Sequence[Callable[[WorkItem], HeadList]]) -> None:
        self.layer, self.heads = layer, list(heads)
        self.stages: list[bytes | None] = [None] * (n_layer_dies + len(heads))
        self.n_layer = n_layer_dies

    @property
    def free(self) -> bool:
        return self.stages[0] is None

    def inject(self, packet: bytes) -> None:
        assert self.free
        self.stages[0] = packet

    def step(self, d: int) -> bytes | None:
        """Every item one die on; the one leaving the last head die comes back."""
        out = self.stages[-1]
        for i in range(len(self.stages) - 1, 0, -1):
            self.stages[i] = self.stages[i - 1]
        self.stages[0] = None
        for i, packet in enumerate(self.stages):
            if packet is None:
                continue
            item, lists = unpack_item(packet, d)
            if i < self.n_layer:
                self.stages[i] = pack_item(WorkItem(item.context, item.position, self.layer(item), item.flags), lists)
            else:
                self.stages[i] = append_head_list(packet, self.heads[i - self.n_layer](item))
        return out


# --------------------------------------------------------------------------
# The controller
# --------------------------------------------------------------------------

class Controller:
    """Requests in, tokens out.  Each step injects one item if the ring's first
    die is free and some context has a token to send -- round robin over the
    contexts, so every die works on a different context's token and one
    conversation sees the ring's latency -- and takes the item leaving the
    ring: a prompt token's lists are dropped, a sampled one's are merged and
    drawn from, the token appended, and the context goes back in the queue."""

    def __init__(self, embedding: EmbeddingTable, ring: Ring, slots: int, seed: int = 1,
                 on_token: Callable[[int, int, float, int, bool], None] | None = None) -> None:
        self.embedding, self.ring, self.table = embedding, ring, ContextTable(slots)
        self.contexts: dict[int, Context] = {}
        self.queue: deque[int] = deque()
        self.rng = np.random.default_rng(seed)
        self.next_id = 0
        self.steps = 0
        # Each sampled token as it is drawn: (context, token, log-probability,
        # its position, whether it ends the turn).  The host's completions.
        self.on_token = on_token

    def submit(self, prompt: Sequence[int], max_new: int, params: SamplingParams = SamplingParams()) -> int:
        """A one-turn context: the prompt, max_new tokens, then its slot is given back."""
        cid = self.open(params)
        self.append(cid, prompt, max_new)
        self.contexts[cid].keep = False
        return cid

    def open(self, params: SamplingParams = SamplingParams(), stops: Sequence[int] = ()) -> int:
        """A context that stays resident between turns: its state is kept on
        the dies until it is closed or its slot is taken for another."""
        ctx = Context(self.next_id, [], 0, 0, params, stops=tuple(stops), keep=True, done=True)
        self.contexts[ctx.id] = ctx
        self.next_id += 1
        return ctx.id

    def append(self, cid: int, tokens: Sequence[int], max_new: int) -> None:
        """A turn: tokens after what the context has seen, then up to max_new
        sampled.  The last token sampled before, if any, goes in first -- it
        was drawn but never fed through the layers."""
        ctx = self.contexts[cid]
        if not ctx.done:
            raise ValueError("the context is mid-turn")
        ctx.tokens += list(tokens)
        ctx.prompt_len, ctx.max_new, ctx.done = len(ctx.tokens), max_new, False
        if cid not in self.queue:
            self.queue.append(cid)

    def cancel(self, cid: int) -> None:
        """End the turn after the token in flight; the context stays open."""
        ctx = self.contexts[cid]
        if not ctx.done:
            ctx.max_new = len(ctx.tokens) - ctx.prompt_len + (1 if ctx.in_flight and ctx.next_pos >= ctx.prompt_len - 1 else 0)
            if not ctx.in_flight:
                self._end_turn(ctx)

    def close(self, cid: int) -> None:
        """Forget the context and give its slot back."""
        ctx = self.contexts[cid]
        ctx.done, ctx.keep = True, False
        if not ctx.in_flight:
            self.table.release(cid)

    def _end_turn(self, ctx: Context) -> None:
        ctx.done = True
        if not ctx.keep:
            self.table.release(ctx.id)

    def _busy(self, ctx: int) -> bool:
        """A slot is not taken from a context in flight or in the middle of a
        turn -- evicted mid-turn it would start over, and two contexts sharing
        one slot would each keep evicting the other.  Only a resident context
        between turns is taken, and its next turn starts from its first token."""
        c = self.contexts[ctx]
        return c.in_flight or not c.done

    def _inject(self) -> bool:
        for _ in range(len(self.queue)):
            cid = self.queue.popleft()
            ctx = self.contexts[cid]
            if ctx.done or ctx.in_flight:
                self.queue.append(cid)
                continue
            slot = self.table.acquire(cid, self._busy)
            if slot is None:
                self.queue.append(cid)                      # every slot is mid-turn: it waits, the holders go on
                continue
            if ctx.slot is not None and not self.table.kept:
                ctx.fresh, ctx.next_pos = True, 0            # evicted meanwhile, even if given the same slot back: start over
            if ctx.fresh:
                ctx.next_pos = 0
            ctx.slot = slot
            pos = ctx.next_pos
            flags = 0
            if ctx.fresh:
                flags |= FLAG_FIRST
            if pos >= ctx.prompt_len - 1:
                flags |= FLAG_SAMPLE
            item = WorkItem(slot, pos, self.embedding.lookup(ctx.tokens[pos]), flags)
            self.ring.inject(pack_item(item))
            ctx.in_flight, ctx.fresh = True, False
            self.queue.append(cid)
            return True
        return False

    def _retire(self, packet: bytes) -> None:
        item, lists = unpack_item(packet, self.embedding.hidden)
        cid = self.table.holder.get(item.context)
        ctx = self.contexts[cid]
        ctx.in_flight = False
        ctx.next_pos = item.position + 1
        if ctx.done:                                        # closed while in flight
            if not ctx.keep:
                self.table.release(cid)
            return
        if item.flags & FLAG_SAMPLE:
            rows, logits, total = merge_lists(lists)
            rnd = int(self.rng.integers(0, 1 << 32))
            token, i = sample(rows, logits, ctx.params, rnd)
            ctx.tokens.append(token)
            ctx.logprobs.append(logprob(int(logits[i]), total))
            last = len(ctx.tokens) - ctx.prompt_len >= ctx.max_new or token in ctx.stops
            if self.on_token is not None:
                self.on_token(cid, token, ctx.logprobs[-1], item.position + 1, last)
            if last:
                self._end_turn(ctx)

    def step(self) -> None:
        out = self.ring.step(self.embedding.hidden)
        if out is not None:
            self._retire(out)
        if self.ring.free:
            self._inject()
        self.steps += 1

    def run(self, max_steps: int = 100_000) -> None:
        for _ in range(max_steps):
            if all(c.done for c in self.contexts.values()) and all(s is None for s in self.ring.stages):
                return
            self.step()
        raise RuntimeError("the ring did not drain")

    def generated(self, cid: int) -> list[int]:
        ctx = self.contexts[cid]
        return ctx.tokens[ctx.prompt_len:]


# --------------------------------------------------------------------------
# Vectors for the gateware's testbenches
# --------------------------------------------------------------------------

def _write_params(directory, **params) -> dict:
    import json
    (directory / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    return params


def emit_sampler_vectors(directory, rng: np.random.Generator, k: int, cases: int) -> dict:
    """``cases`` items for tb_sampler: two lists each, the parameters and the
    random word, and the token and index the model draws."""
    from pathlib import Path
    from fabric import layer as L
    from fabric.tile import write_hex
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    L.write_luts(directory)
    lists0, lists1, params, expected = [], [], [], []
    for c in range(cases):
        vocab = 2 * k * 4
        logits = to_fixed(rng.standard_normal(vocab) * rng.uniform(0.5, 6.0))
        if c % 3 == 1:
            logits[3] = logits[7]                                   # a tie across the halves' order
        half = vocab // 2
        n0, n1 = (k, k) if c % 4 else (rng.integers(1, k + 1), rng.integers(1, k + 1))
        l0, l1 = head_list(0, logits[:half], int(n0), 0), head_list(1, logits[half:], int(n1), half)
        temperature = float(np.exp(rng.uniform(np.log(0.2), np.log(4.0))))     # warm enough that the draw matters
        p = SamplingParams(int(round((1 << FF) / temperature)), int(rng.integers(1, 2 * k + 1)),
                           0xFFFF if c % 2 else int(rng.integers(1000, 0xFFFF)))
        rnd = int(rng.integers(0, 1 << 32))
        rows, merged, _ = merge_lists([l0, l1])
        token, index = sample(rows, merged, p, rnd)
        lists0.append(l0)
        lists1.append(l1)
        params.append((p, rnd))
        expected.append((token, index))
    # Entries as 64-bit words: row in the high half, logit low; a list padded to K with a count word first.
    def words(hl):
        out = [len(hl.rows)]
        for i in range(k):
            r = int(hl.rows[i]) if i < len(hl.rows) else 0
            l = int(hl.logits[i]) & 0xFFFFFFFF if i < len(hl.rows) else 0
            out.append((r << 32) | l)
        return out
    write_hex(directory / "list0.hex", [w for hl in lists0 for w in words(hl)], 64)
    write_hex(directory / "list1.hex", [w for hl in lists1 for w in words(hl)], 64)
    write_hex(directory / "params.hex", [(p.inv_t << 48) | (p.top_k << 40) | (p.top_p << 24) | 0 for p, _ in params], 64)
    write_hex(directory / "rnd.hex", [r for _, r in params], 32)
    write_hex(directory / "expected.hex", [(t << 8) | i for t, i in expected], 40)
    return _write_params(directory, K=k, CASES=cases)


def emit_crc_vectors(directory, rng: np.random.Generator, cases: int) -> dict:
    """Packets of random items through the CRC: the words, the byte counts,
    the model's CRC."""
    from pathlib import Path
    from fabric.tile import write_hex
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    words, lengths, expected, d = [], [], [], 8
    for c in range(cases):
        hidden = rng.integers(-32768, 32767, size=d).astype(np.int16)
        packet = pack_item(WorkItem(c, c * 7, hidden, c & 7))
        if c % 2:
            packet = append_head_list(packet, HeadList(1, np.arange(c % 5 + 1, dtype=np.uint32),
                                                       rng.integers(-1000, 1000, size=c % 5 + 1).astype(np.int32), c))
        body = packet[:-TRAILER_BYTES]                              # the CRC is over everything before it
        lengths.append(len(body))
        padded = body + bytes(-len(body) % 4)
        words += [int.from_bytes(padded[i:i + 4], "little") for i in range(0, len(padded), 4)]
        expected.append(crc32(body))
    write_hex(directory / "words.hex", words, 32)
    write_hex(directory / "lengths.hex", lengths, 16)
    write_hex(directory / "expected.hex", expected, 32)
    return _write_params(directory, CASES=cases, WORDS=len(words))


def emit_ring_vectors(directory, rng: np.random.Generator, cases: int, d: int = 8) -> dict:
    """Packets for tb_ring: the header fields and payload words a sender is
    given, and the words the model says the link carries."""
    from pathlib import Path
    from fabric.tile import write_hex
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    heads, payloads, packets, counts = [], [], [], []
    for c in range(cases):
        hidden = rng.integers(-32768, 32767, size=d).astype(np.int16)
        flags = int(rng.integers(0, 8))
        item = WorkItem(int(rng.integers(0, 1 << 16)), int(rng.integers(0, 1 << 20)), hidden, flags)
        lists = []
        for die in range(int(rng.integers(0, 3))):
            n = int(rng.integers(1, 5))
            lists.append(HeadList(die, rng.integers(0, 1 << 20, size=n).astype(np.uint32),
                                  rng.integers(-5000, 5000, size=n).astype(np.int32), int(rng.integers(-1000, 1000))))
        packet = pack_item(item, lists)
        payload = packet[HEADER_BYTES:-TRAILER_BYTES]
        heads.append((KIND_ITEM, flags, item.context, item.position, len(payload)))
        payloads += [int.from_bytes(payload[i:i + 4], "little") for i in range(0, len(payload), 4)]
        packets += [int.from_bytes(packet[i:i + 4], "little") for i in range(0, len(packet), 4)]
        counts.append(len(payload) // 4)
    write_hex(directory / "hdr.hex",
              [(k) | (f << 8) | (ctx << 16) | (pos << 32) | (ln << 64) for k, f, ctx, pos, ln in heads], 80)
    write_hex(directory / "payload.hex", payloads, 32)
    write_hex(directory / "packet.hex", packets, 32)
    return _write_params(directory, CASES=cases, PWORDS=len(payloads), KWORDS=len(packets))


def emit_top_vectors(directory, rng: np.random.Generator, cases: int, d: int = 8, k: int = 8) -> dict:
    """Requests for tb_controller_top: an embedding table, the requests the
    soft side makes, the packets the model says go out, the replies that come
    back with the head dies' lists, and the tokens the model draws."""
    from pathlib import Path
    from fabric import layer as L
    from fabric.tile import write_hex
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    L.write_luts(directory)
    vocab = 64
    table = EmbeddingTable(rng.integers(-3000, 3000, size=(vocab, d)).astype(np.int16))
    reqs, outs, reps, tokens, rep_counts = [], [], [], [], []
    for c in range(cases):
        token = int(rng.integers(0, vocab))
        slot = int(rng.integers(0, 1 << 12))
        position = int(rng.integers(0, 1 << 16))
        temperature = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
        p = SamplingParams(int(round((1 << FF) / temperature)), int(rng.integers(1, 2 * k + 1)),
                           0xFFFF if c % 2 else int(rng.integers(2000, 0xFFFF)))
        rnd = int(rng.integers(0, 1 << 32))
        # What goes out: the token's row, at the request's slot and position.
        item = WorkItem(slot, position, table.lookup(token), FLAG_SAMPLE)
        outs += [int.from_bytes(pack_item(item)[i:i + 4], "little") for i in range(0, len(pack_item(item)), 4)]
        # What comes back: some other hidden vector, and the two dies' lists.
        logits = to_fixed(rng.standard_normal(2 * k * 4) * rng.uniform(0.5, 4.0))
        half = len(logits) // 2
        lists = [head_list(0, logits[:half], k, 0), head_list(1, logits[half:], k, half)]
        reply = pack_item(WorkItem(slot, position, rng.integers(-1000, 1000, size=d).astype(np.int16), FLAG_SAMPLE), lists)
        reps += [int.from_bytes(reply[i:i + 4], "little") for i in range(0, len(reply), 4)]
        rep_counts.append(len(reply) // 4)
        rows, merged, _ = merge_lists(lists)
        row, index = sample(rows, merged, p, rnd)
        reqs.append((slot, position, token, p, rnd))
        tokens.append((row, index, slot))
    write_hex(directory / "emb.hex", [int(np.uint16(v)) | (int(np.uint16(w)) << 16)
                                      for row in table.rows for v, w in zip(row[0::2], row[1::2])], 32)
    write_hex(directory / "req.hex",
              [(slot) | (pos << 16) | (tok << 48) | (p.inv_t << 56) | (p.top_k << 72) | (p.top_p << 80) | (rnd << 96)
               for slot, pos, tok, p, rnd in reqs], 128)
    write_hex(directory / "out.hex", outs, 32)
    write_hex(directory / "reply.hex", reps, 32)
    write_hex(directory / "rcount.hex", rep_counts, 16)
    write_hex(directory / "token.hex", [(row << 24) | (index << 16) | slot for row, index, slot in tokens], 64)
    return _write_params(directory, D=d, K=k, CASES=cases, EWORDS=vocab * d // 2,
                         OWORDS=len(outs), RWORDS=len(reps))
