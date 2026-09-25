"""The host's queues: how software on the host talks to the controller.

The appliance is a PCIe device.  The host drives it the way it drives an
NVMe drive, through a pair of rings in its own memory and two doorbells:

* the **submission queue**, SQ_SIZE entries of 64 bytes, which the host
  fills and the device fetches by DMA.  The host writes the index after its
  last new entry to the SQ tail doorbell;
* the **completion queue**, CQ_SIZE entries of 16 bytes, which the device
  fills by DMA and the host reads.  An entry is new when its phase bit is
  the pass's: the device writes 1 on its first pass round the ring, 0 on its
  second, and so on, so the host needs no count from the device.  The host
  writes the index after its last consumed entry to the CQ head doorbell,
  which is what lets the device reuse the slots.  The device never writes an
  entry the host has not consumed: completions wait on the device instead.

Every entry is little-endian.  A command:

    0   opcode   u8     IDENTIFY, OPEN, APPEND, CANCEL, CLOSE
    1   flags    u8     (none yet)
    2   cmd_id   u16    the host's, echoed in every completion the command makes
    4   ctx      u32    the context, for APPEND, CANCEL and CLOSE
    8   addr     u64    host memory: the tokens for APPEND (u32 each), the page for IDENTIFY
    16  count    u32    tokens at addr
    20  max_new  u32    tokens to sample after them (APPEND; at least one)
    24  inv_t    u32    OPEN: 1 / temperature in F16, as the sampler takes it
    28  top_k    u16
    30  top_p    u16    in U16: 0xFFFF is 1.0
    32  n_stops  u16    OPEN: tokens that end a turn when sampled, up to four
    34  reserved u16
    36  stops    4 x u32
    52  reserved

A completion:

    0   token    u32    the sampled token (TOKEN set)
    4   logprob  i32    its log-probability, LOGPROB_FRAC fraction bits
    8   ctx      u16    the context (OPEN's new one)
    10  cmd_id   u16
    12  sq_head  u16    the SQ entries the device has fetched: the host's room
    14  status   u8     OK, or why not
    15  flags    u8     PHASE, TOKEN (a token is in the entry), LAST (the command is finished)

IDENTIFY, OPEN, CANCEL and CLOSE each complete once, LAST set.  APPEND
completes once per sampled token, the turn's last with LAST -- a token as
soon as it is drawn, which is the stream a chat interface shows -- or once
with an error.  A context is resident between turns: its state stays on
the dies, so the next APPEND carries only what is new.  Resident contexts
keep their slots until the slots are needed; one that lost its slot starts
its next turn from its first token, which the controller does by itself
(it keeps every context's tokens), so the host sees latency, not an error.

The model below is the device's side (``Device``) and a driver's
(``Driver``), over a byte array standing in for host memory; the gateware
and the Linux driver are checked against it.
"""

from __future__ import annotations

import math
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

from fabric import controller as C

# Opcodes.
IDENTIFY, OPEN, APPEND, CANCEL, CLOSE = 0x01, 0x02, 0x03, 0x04, 0x05
# Status.
OK, BAD_OPCODE, BAD_CONTEXT, BUSY, BAD_ARGUMENT, NO_CONTEXTS = 0, 1, 2, 3, 4, 5
# Completion flags.
PHASE, TOKEN, LAST, CANCELLED = 0x01, 0x02, 0x04, 0x08

SQ_ENTRY, CQ_ENTRY = 64, 16
LOGPROB_FRAC = 16
MAX_STOPS = 4
IDENTIFY_BYTES = 4096
MAGIC = 0x43495341                  # "ASIC"
VERSION = 1

# Registers, 32 bits each, in the device's BAR 0.
REG_ID, REG_VERSION, REG_CONTEXTS, REG_VOCAB = 0x000, 0x004, 0x008, 0x00C
REG_SQ_BASE_LO, REG_SQ_BASE_HI, REG_SQ_SIZE = 0x010, 0x014, 0x018
REG_CQ_BASE_LO, REG_CQ_BASE_HI, REG_CQ_SIZE = 0x01C, 0x020, 0x024
REG_ENABLE, REG_STATUS = 0x028, 0x02C
REG_SQ_TAIL, REG_CQ_HEAD = 0x100, 0x104          # the doorbells
MAX_QUEUE = 4096

_SQ = struct.Struct("<BBHIQIIIHHH2x4I12x")
_CQ = struct.Struct("<IiHHHBB")
assert _SQ.size == SQ_ENTRY and _CQ.size == CQ_ENTRY


@dataclass
class Command:
    opcode: int
    cmd_id: int = 0
    ctx: int = 0
    addr: int = 0
    count: int = 0
    max_new: int = 0
    inv_t: int = 0
    top_k: int = 0
    top_p: int = 0xFFFF
    stops: tuple[int, ...] = ()
    flags: int = 0

    def pack(self) -> bytes:
        if len(self.stops) > MAX_STOPS:
            raise ValueError(f"at most {MAX_STOPS} stop tokens")
        stops = list(self.stops) + [0] * (MAX_STOPS - len(self.stops))
        return _SQ.pack(self.opcode, self.flags, self.cmd_id, self.ctx, self.addr, self.count, self.max_new,
                        self.inv_t, self.top_k, self.top_p, len(self.stops), *stops)

    @classmethod
    def unpack(cls, raw: bytes) -> "Command":
        (op, flags, cmd_id, ctx, addr, count, max_new, inv_t, top_k, top_p, n_stops, *stops) = _SQ.unpack(raw)
        return cls(op, cmd_id, ctx, addr, count, max_new, inv_t, top_k, top_p, tuple(stops[:min(n_stops, MAX_STOPS)]), flags)


@dataclass
class Completion:
    ctx: int
    cmd_id: int
    status: int = OK
    flags: int = 0
    token: int = 0
    logprob: int = 0                 # fixed point, LOGPROB_FRAC fraction bits
    sq_head: int = 0

    def pack(self) -> bytes:
        return _CQ.pack(self.token, self.logprob, self.ctx, self.cmd_id, self.sq_head, self.status, self.flags)

    @classmethod
    def unpack(cls, raw: bytes) -> "Completion":
        token, logprob, ctx, cmd_id, sq_head, status, flags = _CQ.unpack(raw)
        return cls(ctx, cmd_id, status, flags, token, logprob, sq_head)


def logprob_fixed(lp: float) -> int:
    return max(-(1 << 31), int(round(lp * (1 << LOGPROB_FRAC))))


def sampling_fields(params: C.SamplingParams) -> dict:
    return {"inv_t": params.inv_t, "top_k": params.top_k, "top_p": params.top_p}


class HostMemory:
    """The host's memory as the device's DMA sees it."""

    def __init__(self, size: int) -> None:
        self.data = bytearray(size)

    def read(self, addr: int, n: int) -> bytes:
        return bytes(self.data[addr:addr + n])

    def write(self, addr: int, raw: bytes) -> None:
        self.data[addr:addr + len(raw)] = raw


class Device:
    """The controller's host side: its registers, the fetch of commands and
    the posting of completions, over the controller model."""

    def __init__(self, ctl: C.Controller, memory: HostMemory, max_contexts: int = 1024) -> None:
        self.ctl, self.mem, self.max_contexts = ctl, memory, max_contexts
        self.regs = {REG_ID: MAGIC, REG_VERSION: VERSION, REG_CONTEXTS: max_contexts, REG_VOCAB: ctl.embedding.vocab,
                     REG_SQ_BASE_LO: 0, REG_SQ_BASE_HI: 0, REG_SQ_SIZE: 0, REG_CQ_BASE_LO: 0, REG_CQ_BASE_HI: 0,
                     REG_CQ_SIZE: 0, REG_ENABLE: 0, REG_STATUS: 0}
        self.sq_head = self.sq_tail = 0
        self.cq_tail = self.cq_head = 0
        self.phase = 1
        self.pending: deque[Completion] = deque()           # completions waiting for room in the CQ
        self.turn_cmd: dict[int, int] = {}                   # context -> the APPEND its tokens complete
        self.cancelled: set[int] = set()
        self.interrupts = 0
        ctl.on_token = self._token

    # --- registers
    def mmio_write(self, reg: int, value: int) -> None:
        value &= 0xFFFFFFFF
        if reg == REG_SQ_TAIL:
            self.sq_tail = value % max(self.regs[REG_SQ_SIZE], 1)
        elif reg == REG_CQ_HEAD:
            self.cq_head = value % max(self.regs[REG_CQ_SIZE], 1)
            self._flush()
        elif reg == REG_ENABLE:
            self.regs[REG_ENABLE] = value & 1
            if value & 1:
                for r in (REG_SQ_SIZE, REG_CQ_SIZE):
                    if not 2 <= self.regs[r] <= MAX_QUEUE or self.regs[r] & (self.regs[r] - 1):
                        raise ValueError("queue sizes are powers of two from 2 to 4096")
                self.sq_head = self.sq_tail = self.cq_tail = self.cq_head = 0
                self.phase = 1
            self.regs[REG_STATUS] = value & 1
        elif reg in (REG_SQ_BASE_LO, REG_SQ_BASE_HI, REG_SQ_SIZE, REG_CQ_BASE_LO, REG_CQ_BASE_HI, REG_CQ_SIZE):
            self.regs[reg] = value
        else:
            raise ValueError(f"register {reg:#x} is read-only")

    def mmio_read(self, reg: int) -> int:
        return self.regs[reg]

    def _base(self, lo: int, hi: int) -> int:
        return self.regs[lo] | (self.regs[hi] << 32)

    # --- commands
    def service(self) -> None:
        """Fetch and execute every command up to the tail."""
        if not self.regs[REG_ENABLE]:
            return
        size = self.regs[REG_SQ_SIZE]
        while self.sq_head != self.sq_tail:
            raw = self.mem.read(self._base(REG_SQ_BASE_LO, REG_SQ_BASE_HI) + self.sq_head * SQ_ENTRY, SQ_ENTRY)
            self.sq_head = (self.sq_head + 1) % size
            self._execute(Command.unpack(raw))

    def _execute(self, cmd: Command) -> None:
        ctl = self.ctl
        done = lambda status=OK, ctx=cmd.ctx: self._post(Completion(ctx, cmd.cmd_id, status, LAST))
        if cmd.opcode == IDENTIFY:
            page = struct.pack("<IIIIII", MAGIC, VERSION, self.max_contexts, ctl.embedding.vocab,
                               ctl.table.slots, ctl.embedding.hidden)
            self.mem.write(cmd.addr, page + bytes(IDENTIFY_BYTES - len(page)))
            done()
        elif cmd.opcode == OPEN:
            open_now = sum(1 for c in ctl.contexts.values() if c.keep)
            if open_now >= self.max_contexts:
                done(NO_CONTEXTS, 0)
                return
            params = C.SamplingParams(cmd.inv_t, cmd.top_k, cmd.top_p)
            done(OK, ctl.open(params, cmd.stops))
        elif cmd.opcode in (APPEND, CANCEL, CLOSE):
            ctx = ctl.contexts.get(cmd.ctx)
            if ctx is None or not ctx.keep:
                done(BAD_CONTEXT)
            elif cmd.opcode == APPEND:
                if not ctx.done:
                    done(BUSY)
                elif cmd.max_new < 1:
                    done(BAD_ARGUMENT)
                else:
                    raw = self.mem.read(cmd.addr, 4 * cmd.count)
                    tokens = list(struct.unpack(f"<{cmd.count}I", raw))
                    if any(t >= ctl.embedding.vocab for t in tokens):
                        done(BAD_ARGUMENT)
                        return
                    self.turn_cmd[cmd.ctx] = cmd.cmd_id
                    self.cancelled.discard(cmd.ctx)
                    ctl.append(cmd.ctx, tokens, cmd.max_new)
            elif cmd.opcode == CANCEL:
                if not ctx.done:
                    self.cancelled.add(cmd.ctx)
                    ctl.cancel(cmd.ctx)
                done()
            else:
                ctl.close(cmd.ctx)
                done()
        else:
            done(BAD_OPCODE)

    # --- completions
    def _token(self, cid: int, token: int, lp: float, position: int, last: bool) -> None:
        flags = TOKEN | (LAST if last else 0) | (CANCELLED if last and cid in self.cancelled else 0)
        self._post(Completion(cid, self.turn_cmd.get(cid, 0), OK, flags, token, logprob_fixed(lp)))

    def _post(self, cmp: Completion) -> None:
        self.pending.append(cmp)
        self._flush()

    def _flush(self) -> None:
        size = self.regs[REG_CQ_SIZE]
        posted = False
        while self.pending and (self.cq_tail + 1) % size != self.cq_head:
            cmp = self.pending.popleft()
            cmp.flags = (cmp.flags & ~PHASE) | self.phase
            cmp.sq_head = self.sq_head
            self.mem.write(self._base(REG_CQ_BASE_LO, REG_CQ_BASE_HI) + self.cq_tail * CQ_ENTRY, cmp.pack())
            self.cq_tail = (self.cq_tail + 1) % size
            if self.cq_tail == 0:
                self.phase ^= 1
            posted = True
        if posted:
            self.interrupts += 1


class Driver:
    """What a driver does: the queues in host memory, the doorbells, and the
    completions read back by their phase."""

    def __init__(self, dev: Device, memory: HostMemory, sq_entries: int = 64, cq_entries: int = 256,
                 base: int = 0x1000) -> None:
        self.dev, self.mem = dev, memory
        self.sq_size, self.cq_size = sq_entries, cq_entries
        self.sq_base = base
        self.cq_base = self.sq_base + sq_entries * SQ_ENTRY
        self.buf_base = self.cq_base + cq_entries * CQ_ENTRY
        self.buf_next = self.buf_base
        self.sq_tail = self.sq_head = 0
        self.cq_head = 0
        self.phase = 1
        self.next_id = 0
        if dev.mmio_read(REG_ID) != MAGIC:
            raise RuntimeError("not the device")
        self.mem.write(self.cq_base, bytes(cq_entries * CQ_ENTRY))
        for reg, value in ((REG_SQ_BASE_LO, self.sq_base & 0xFFFFFFFF), (REG_SQ_BASE_HI, self.sq_base >> 32),
                           (REG_SQ_SIZE, sq_entries), (REG_CQ_BASE_LO, self.cq_base & 0xFFFFFFFF),
                           (REG_CQ_BASE_HI, self.cq_base >> 32), (REG_CQ_SIZE, cq_entries), (REG_ENABLE, 1)):
            dev.mmio_write(reg, value)

    def _buffer(self, raw: bytes) -> int:
        addr = self.buf_next
        self.mem.write(addr, raw)
        self.buf_next += (len(raw) + 63) // 64 * 64
        return addr

    def submit(self, cmd: Command) -> int:
        if (self.sq_tail + 1) % self.sq_size == self.sq_head:
            raise RuntimeError("the submission queue is full")
        cmd.cmd_id = self.next_id
        self.next_id = (self.next_id + 1) & 0xFFFF
        self.mem.write(self.sq_base + self.sq_tail * SQ_ENTRY, cmd.pack())
        self.sq_tail = (self.sq_tail + 1) % self.sq_size
        self.dev.mmio_write(REG_SQ_TAIL, self.sq_tail)
        return cmd.cmd_id

    def poll(self) -> list[Completion]:
        out = []
        while True:
            cmp = Completion.unpack(self.mem.read(self.cq_base + self.cq_head * CQ_ENTRY, CQ_ENTRY))
            if (cmp.flags & PHASE) != self.phase:
                break
            out.append(cmp)
            self.sq_head = cmp.sq_head
            self.cq_head = (self.cq_head + 1) % self.cq_size
            if self.cq_head == 0:
                self.phase ^= 1
        if out:
            self.dev.mmio_write(REG_CQ_HEAD, self.cq_head)
        return out

    # The commands.
    def identify(self) -> int:
        self.identify_page = self.buf_next
        self.buf_next += IDENTIFY_BYTES
        return self.submit(Command(IDENTIFY, addr=self.identify_page))

    def open(self, params: C.SamplingParams = C.SamplingParams(), stops: Sequence[int] = ()) -> int:
        return self.submit(Command(OPEN, stops=tuple(stops), **sampling_fields(params)))

    def append(self, ctx: int, tokens: Sequence[int], max_new: int) -> int:
        addr = self._buffer(struct.pack(f"<{len(tokens)}I", *tokens))
        return self.submit(Command(APPEND, ctx=ctx, addr=addr, count=len(tokens), max_new=max_new))

    def cancel(self, ctx: int) -> int:
        return self.submit(Command(CANCEL, ctx=ctx))

    def close(self, ctx: int) -> int:
        return self.submit(Command(CLOSE, ctx=ctx))


def run(dev: Device, drv: Driver, until, max_steps: int = 200_000) -> list[Completion]:
    """Step the device and the ring, draining completions, until ``until(completions so far)``."""
    got: list[Completion] = []
    for _ in range(max_steps):
        dev.service()
        got += drv.poll()
        if until(got):
            return got
        dev.ctl.step()
    raise RuntimeError("never finished")
