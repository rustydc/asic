"""The block diagrams and interface contracts, one source for docs/BLOCKS.md and a standalone HTML page.

    python -m fabric.docs.blocks_doc            # writes fabric/docs/BLOCKS.md
    python -m fabric.docs.blocks_doc page.html  # also writes the HTML page with the diagrams inline

The contracts are transcribed from the RTL port lists (fabric_engine.sv,
fabric_sequencer.sv, fabric_memory.sv, fabric_hpi.sv, fabric_cdc.sv,
fabric_tile.sv and the vector units) and from fabric/sequencer.py.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

HERE = Path(__file__).parent

# A section is (title, [blocks]); a block is a paragraph string, ("diagram", name),
# ("table", [headers], [rows]) or ("list", [items]).
SECTIONS = [
    ("The appliance", [
        ("diagram", "appliance"),
        "One board: a controller FPGA on PCIe and a ring of ten ASICs of one design. Eight run four consecutive layers "
        "each (three Gated DeltaNet recurrent layers and one gated-attention global layer), two run in head mode and hold "
        "half of the 248K-row LM head each. A work item (one token of one context: the hidden vector, the context id, the "
        "position) enters at die 0, passes through every die, and leaves the head dies with two partial top-k lists that "
        "the FPGA merges and samples. Every die works on a different context's token at once, so the appliance's "
        "throughput is one die's and a single conversation sees the ring's latency.",
        ("table", ["Interface", "Carries", "Contract"], [
            ["Host link", "PCIe Gen4 x4 to x8 over a SlimSAS cable", "The FPGA terminates PCIe and owns the host protocol, scheduling, sampling, context allocation and telemetry."],
            ["Ring link (die to die)", "Work items: hidden vector (4096 x 16 bit), context id, position; head dies append their partial lists",
             "Source-synchronous ready/valid packets with framing and CRC; candidate 32 data bits at 250 MHz DDR, 2 GB/s raw. A die forwards the item unchanged and replaces the hidden vector (layer mode) or appends its list (head mode)."],
            ["Die memory", "Per-context state of the die's four layers (see the memory map)", "16 AP Memory APS512XXN PSRAMs in HPI x16 mode per layer die, 1 GB, 16 GB/s at 250 MHz, in 1 KB stripes across the devices; head dies have none."],
            ["Management SPI", "Boot: the ~80 KB of per-column requantizer constants and unit constants; the mode strap", "Loaded into the constants SRAM before the first work item."],
        ]),
    ]),
    ("A layer die", [
        ("diagram", "layer_die"),
        "The die is a token sequencer, a vector buffer, the units behind their adapters, and one memory path. The "
        "sequencer runs up to four lanes, each a context's token through the die's layer programs (lists of unit commands "
        "with the buffers each consumes and produces) in its own order, and each cycle issues the oldest lane's next "
        "command whose buffers' scoreboards are clear and whose engine is free; the engine returns the command's tag "
        "when its last write has landed. Every unit reads its operands from and writes its results to "
        "the vector buffer at byte addresses the program carries, so the buffer is the only coupling between units. The "
        "memory unit is the one requester of the die's memory port: the state and history moves of the recurrent layers, "
        "the append, the index scan and the record reads of the global layer.",
        ("table", ["Block", "Role", "Instances (9B die)"], [
            ["Token sequencer", "Microcoded issue engine of four lanes: a program store of 256-bit words, a run queue and writer and reader counters per buffer id a lane; tag table", "1"],
            ["Vector buffer", "Byte-addressed SRAM, 16-byte beats, one read port per unit stream and 19 write ports", "1, 312 KB for a recurrent token, 1.2 MB for a global token"],
            ["Tile array (pass adapter)", "The four passes of a layer (in, out+gates, FFN gate/up, FFN down) over NT tiles of 4096 x 64 via-ROM coefficients", "3306 tiles"],
            ["Norm", "RMS norm with a gain; the L2 normaliser and the gated norm by its arguments", "2 engines"],
            ["Conv", "4-tap causal convolution with SiLU over the q, k, v channels, history in and out", "1"],
            ["Gates", "Per-head decay and beta from the pass's one-column accumulators", "1"],
            ["State engine", "The int8 Gated DeltaNet update of one head (128 x 128 rows) with its scale header", "4 engines"],
            ["SwiGLU", "silu(gate) * up, requantized", "1"],
            ["Residual", "h + y * mult >> shift in int16", "1"],
            ["Rotary", "Table: sin/cos of the position; head: head norm, rotation of the first RD dims, int8", "2 engines"],
            ["Attention", "Online-softmax attention of a query group over the head's window and block rows, gated output", "4 cores (one per KV head)"],
            ["Memory unit", "Mover, append, index scan with top-K, record reader, behind a 4-way arbiter", "1"],
            ["Memory path", "Bridge (clock crossing), stripe unit, 16 channel controllers, PHY (delay lines, DLL)", "1 path, 16 channels"],
        ]),
    ]),
    ("A tile", [
        ("diagram", "tile"),
        "A tile is a via-programmed ROM of 4096 rows by 64 columns of 4-bit signed-digit coefficients, read two rows per "
        "cycle by a cycle counter, feeding 64 columns that accumulate in carry-save form and a shared requantizer that walks "
        "the columns after the 2048-cycle pass. With T tokens a pass, the columns keep T accumulator sets and the "
        "activations of T tokens arrive together.",
        ("table", ["Signal", "Direction", "Width", "Meaning"], [
            ["start", "in", "1", "Begins a pass: loads the accumulators from psum_in (zero for a fresh pass, the previous row block's psum_out for a chained one) and resets the cycle counter."],
            ["psum_in", "in", "T x COLS x ACC", "Chained partial sums, raw accumulators of the tile whose rows precede this one."],
            ["x_valid / x_data / x_ready", "in / in / out", "1 / T x P x AB / 1", "P=2 activations per cycle per token in row order; x_ready is high while a pass is running."],
            ["mult / shift", "in", "COLS x 16 / COLS x 5", "Per-column requantizer constants from the constants SRAM."],
            ["done", "out", "1", "One cycle at the end of the walk."],
            ["psum_out", "out", "T x COLS x ACC", "The resolved accumulators (raw), for the chained next tile or the one-column gate heads."],
            ["q_out / q_valid", "out", "T x COLS x 8 / 1", "The requantized int8 outputs of every column, valid with done."],
        ]),
    ]),
    ("Command bus: sequencer to adapters", [
        "One command per cycle at most. Each engine port reports itself free; the sequencer picks the oldest lane whose "
        "head step's engine is free and whose buffers are clear, drives its operands and asserts cmd_valid for its unit, "
        "and the addressed engine takes the command that cycle. When the engine's last write has landed it pulses "
        "done_valid with the tag; the sequencer drains one completion a cycle and may issue a dependent step two cycles "
        "after it.",
        ("table", ["Signal", "Direction", "Width", "Meaning"], [
            ["cmd_valid[NU]", "seq to units", "NU=10", "One-hot by unit id: tiles 0, norm 1, conv 2, gates 3, state 4, swiglu 5, residual 6, rotary 7, attention 8, memory 9."],
            ["cmd_engine", "seq to units", "4", "Engine index within the unit (norm 0-1, state 0-3, rotary 0-1, attention 0-3, others 0)."],
            ["cmd_len", "seq to units", "16", "Beat count of the stream, or the token count T for a pass."],
            ["cmd_src, cmd_dst, cmd_a2, cmd_a3", "seq to units", "4 x 30", "Byte addresses into the vector buffer (or beat addresses into memory for the memory unit's DMA operands)."],
            ["cmd_arg", "seq to units", "32", "Unit-specific argument (constant set, flags, operation, position); see the operand conventions."],
            ["cmd_tag", "seq to units", "8", "The issue count modulo 256; returned on completion."],
            ["cmd_lane, cmd_layer, cmd_page", "seq to units", "2, 2, 21", "The command's lane (its token in flight), its run's layer (the units' constant bank, which each port keeps for its command) and its run's part of the slot (the memory unit adds it to the token's slot)."],
            ["port_ready[NU x NE]", "units to seq", "40", "Each engine port free for a command this cycle."],
            ["done_valid[NU x NE], done_tag", "units to seq", "40, 40 x 8", "One port per engine; a one-cycle pulse with the tag of the completed command."],
            ["push, push_lane, push_pc, push_steps, push_layer, push_page", "top", "1, 2, 16, 16, 2, 21", "A run for a lane: a program in the lane's store, its layer and its part of the slot; a lane holds four, and fetches them one after another."],
            ["push_room, lane_busy, lane_done / running", "top", "4, 4, 4 / 1", "Per lane: room for a run, runs outstanding, and a pulse when everything it was given has completed."],
        ]),
        ("table", ["Program word bits", "Field", "Meaning"], [
            ["[3:0]", "unit", "Unit id"],
            ["[7:4]", "engine", "Engine index"],
            ["[8]", "last", "The program ends after this step"],
            ["[31:16]", "len", "cmd_len"],
            ["[63:32]", "arg", "cmd_arg"],
            ["[93:64], [123:94], [153:124], [183:154]", "src, dst, a2, a3", "The four 30-bit address operands"],
            ["[231:184]", "consumed ids", "Six 8-bit buffer ids (0xFF none): the step waits for their outstanding writers"],
            ["[247:232]", "produced ids", "Two 8-bit buffer ids: the step waits for their outstanding readers, and writers unless a contribution"],
            ["[249:248]", "contribution", "Per produced id: this step writes a slice of a vector several steps fill together, so writers do not serialize"],
        ]),
        ("table", ["Unit", "len", "src", "dst", "a2", "a3", "arg"], [
            ["tiles (pass)", "T tokens", "activations", "outputs", "in stride per token", "out stride per token", "pass | row block << 8 | T << 16"],
            ["norm", "beats", "x", "y", "gain vector (gated: the gate vector)", "", "[7:0] constant set (0 residual, 1 unit, 2 gated, 3 ffn), [8] int16 input, [9] gated"],
            ["conv", "beats", "x (q,k,v)", "y", "history", "history out", ""],
            ["gates", "beats", "b accumulators (words)", "gates words", "a accumulators (words)", "", ""],
            ["state", "", "q then k (K bytes each)", "y", "v", "slot (header beat + K rows)", "gates word address"],
            ["swiglu", "beats", "gate", "y", "up", "", "constant set"],
            ["residual", "beats", "h", "h out", "y", "", "constant set"],
            ["rotary", "", "head vector (op 1)", "table / rotated head", "table (op 1)", "position (op 0)", "[3:0] op (0 table, 1 head), [7:4] kind (q or k)"],
            ["attention", "N records", "queries", "output", "gate base (stride 2 x HD)", "rows buffer (records as stored)", "[31] rows from the position, [30] a chunk's shared rows, [23:16] its tokens, [15:0] the token's place"],
            ["memory", "beats, or a chunk's tokens", "memory beat address (rd) / buffer (wr, append v)", "buffer (rd) / memory (wr)", "v (append), head (rows)", "context page", "[3:0] op (0 rd, 1 wr, 2 append, 3 scan, 4 rows, 5/6 a chunk's window before and after its appends, 7 its blocks), [31:4] the token's place in its chunk"],
        ]),
    ]),
    ("Vector buffer port", [
        ("table", ["Signal", "Direction", "Width", "Meaning"], [
            ["rd_addr[NR x AW]", "unit to buffer", "24 per port", "Byte address of a 16-byte beat, any alignment; NR ports (norm 4, tiles TMAX, conv 2, gates 2, state 4, swiglu 2, residual 2, rotary 2, attention 4, memory 1)."],
            ["rd_data[NR x 128]", "buffer to unit", "128 per port", "The beat, one cycle after the address."],
            ["wr_en, wr_addr, wr_data, wr_be", "unit to buffer", "1, 24, 128, 16 per port", "19 write ports; a beat with byte enables lands the same cycle (blocking write, reads in the same cycle see the old data)."],
        ]),
        "Adapters read their streams NL=8 lanes (one beat) per cycle and write one beat per cycle; a unit's last write "
        "precedes its done by one cycle, which is what makes the scoreboard's release safe.",
    ]),
    ("Memory port and the memory path", [
        "One protocol from the memory unit's requesters to the PSRAM channels. A requester may keep several reads in flight; "
        "they come back in the order they were taken.",
        ("table", ["Signal", "Direction", "Width", "Meaning"], [
            ["req_valid / req_ready", "requester / memory", "1 / 1", "A request is taken on valid and ready."],
            ["req_write", "requester", "1", "Write (beats follow on wdata) or read (beats return on rdata)."],
            ["req_addr", "requester", "32", "Byte address aligned to a 16-byte beat."],
            ["req_beats", "requester", "12", "Beats in the burst, up to 4095."],
            ["wdata_valid / wdata_ready / wdata", "requester / memory / requester", "1 / 1 / 128", "The write beats in order."],
            ["rdata_valid / rdata", "memory", "1 / 128", "The read beats in order, no ready: the requester must sink them."],
        ]),
        ("table", ["Block", "Upstream", "Downstream", "Contract"], [
            ["fabric_mem_arbiter", "N=4 requester ports (mover, append, scan, rows fetcher)", "One memory port", "Round-robin: the next requester after the last granted with a request pending. A write holds the port until its data has gone; a read lets it go once taken, and its data is steered back from an in-order queue of the reads in flight (16)."],
            ["fabric_mem_bridge", "The core-clock memory port", "The 250 MHz controller-clock memory port", "Three asynchronous FIFOs (requests 4, write beats 16, read beats 16) and a queue of the reads in flight (16); rd_overflow flags a read burst the core did not drain in time."],
            ["fabric_hpi_stripe", "One memory port", "NDEV=16 channels: x_valid[dev]/x_ready, x_write, x_addr[24:0], x_beats[7:0], per-device wdata/rdata, x_done[dev]", "Consecutive 1 KB stripes on consecutive devices: a burst is split into its stripe chunks, which run on their devices, and read beats are reassembled in order; a chunk never crosses a device page. The next request is taken once the last one's chunks are issued, so requests overlap; idle when nothing is in flight."],
            ["fabric_hpi_channel", "One transaction port: xact_valid/ready, xact_write, xact_addr[24:0], xact_beats[7:0], 128-bit wdata/rdata, xact_done", "HPI x16 pins: ce_n, dq[15:0] with dq_oe, dm[1:0], dqs[1:0]", "Power-up and reset timing (tPU, tRST), MR0/MR4/MR8 writes, linear bursts within tCEM, tCPH between transactions, write latency WLC; 128-bit beats become eight 16-bit words. Two page buffers: a read drains as it arrives, and the device takes the next transaction while the last one's data waits its turn."],
            ["fabric_phy", "The channel's DQS", "Delay lines, DLL", "The DLL locks a delay line to the clock period and gives the quarter-period code that centres DQS on DQ."],
        ]),
        ("table", ["Region per context", "Contents", "Placement"], [
            ["State (per recurrent layer)", "v_heads slots, each a header beat (scale g, exponent e, peak, saturated count) and K rows of V int8", "2 KB page aligned"],
            ["History (per recurrent layer)", "conv_dim channels x (kernel - 1) int8", "page aligned"],
            ["Window (global layer)", "local_window positions x KV heads x (key, value) at kv_bits, head-major", "page aligned"],
            ["Block store", "One record per closed block: pooled key and value per KV head", "page aligned"],
            ["Index", "One record per block: index_dim 4-bit codes and a scale", "page aligned"],
            ["Block sums", "The running sums of the open block's keys, values and index vector", "one record"],
        ]),
    ]),
    ("Vector unit streams", [
        "Each vector unit is a streaming datapath of L lanes per beat: in_valid presents a beat, out_valid follows after "
        "the unit's fixed latency with the beat's results, and the per-channel constants arrive with the beat. The "
        "adapters supply the beats from the vector buffer and the constants from the constants SRAM.",
        ("table", ["Unit", "Inputs per beat", "Outputs", "Control"], [
            ["fabric_rmsnorm", "in_x (L x XW), in_gain (L x GW); mult, shift, eps", "out_y (L x OW)", "n_beats at run time; phases fill (sum of squares), rsqrt (the inverse square root unit), drain; one instance serves the residual norm, the L2 normaliser and the gated norm."],
            ["fabric_conv_silu", "in_x (L x 8), in_hist (L x (K-1) x 8), in_w (L x K x 8), input and output mult/shift per lane", "out_y (L x 8), out_hist (L x (K-1) x 8)", "History in with the beat, the shifted history out for the next token."],
            ["fabric_head_gates", "a_acc, b_acc (ACC bits), mult/shift for each, a_coef, dt_bias", "decay, beta (16 bits each)", "One head per beat; softplus and exp through the tables."],
            ["fabric_delta_state8", "q, k (K x 8), v (V x 8), decay, beta; g_in, e_in, peak_in, nsat_in; row_in stream (V x 8 int8)", "row_out stream, y (V x 16); g_out, e_out, peak_out, nsat_out", "start latches the vectors; pass 1 takes the rows and accumulates the prediction while a sequential divider forms 1/g; pass 2 streams the updated rows and accumulates y; y_valid after the last row."],
            ["fabric_swiglu", "in_g, in_u (L x 8); mult_g, sh_g, mult_o, sh_o", "out_y (L x 8)", "silu through the table, product requantized."],
            ["fabric_residual", "in_h (L x 16), in_y (L x 8); mult, shift", "out_h (L x 16)", "Saturating int16 add."],
            ["fabric_rotary_table", "pos (32), inv_freq (R/2 x 32)", "sin_tab, cos_tab (R/2 x 16), done", "start; done after R/2 + 3 cycles; the tables hold until the next start."],
            ["fabric_rotary", "in_x (L x 16), the tables, mult, shift", "out_y (L x 8)", "Buffers the head (HD/L beats), then streams the rotated pairs (i, i + R/2) and the rest, requantized to int8."],
            ["fabric_attention", "in_kind (0 query, 1 gate, 2 key, 3 value), in_data (L x 8), in_ready; mult_s, sh_s, mult_gate, sh_gate, mult_o, sh_o", "out_data (L x 8) stream, done", "start; G query rows then G gate rows; per record HD/L key beats then HD/L value beats, in_ready dropping while the exponential runs; finish starts the output: per head the reciprocal of the sum, then HD/L output beats."],
        ]),
    ]),
    ("Memory units", [
        ("table", ["Unit", "Command side", "Memory side", "Contract"], [
            ["fabric_row_dma (mover)", "rd_start/rd_base, wr_start/wr_base, row streams", "one request port", "Reads or writes ROWS rows of ROW_BITS as one burst; the state slots and histories."],
            ["fabric_kv_append", "start, pos, window/block/index bases, k_rows, v_rows, idx_k, running sums in", "one request port (writes)", "Writes the token's keys and values into the window at pos; adds them to the block sums; at a block end pools the block into a block record and codes its index record (4-bit codes and a scale); sums out for the memory."],
            ["fabric_index_scan", "start, base, n_blocks, NQ queries' q_codes and counts", "one request port (reads RPB records a page, a whole number of stripes, eight pages in flight, two beats a transfer)", "Streams every eligible block's index record, scores it against each coded query (a dot product of 4-bit codes times the record's scale), both beats of a transfer at once, and emits (cand_id, cand_score) for every query whose count takes the block."],
            ["fabric_topk", "clear, cand_valid/cand_id/cand_score, finish", "", "Keeps the K best candidates; after finish streams them out (out_valid, out_id, out_score, out_last) and pulses done."],
        ]),
        "The memory unit's adapter turns the program's memory commands into these: op 0 is a read of beats into the buffer, "
        "the rows fetcher's, reported done on the engine the command came on (the unit has four, so the recurrent layer's "
        "state reads ahead are in flight together), op 1 the mover's posted write of the buffer to memory, op 2 the append of the token in the buffer, op 3 the scan of the "
        "context's index into the top-K ids, op 4 the rows command that turns a selection into the rows fetcher's moves (the "
        "window in page runs, then one record per chosen block), the records into the head's buffer as they are stored, "
        "for the attention adapter to unpack; ops 5 to 7 are a prefill chunk's: its window before and after its appends "
        "and its tokens' blocks, into one rows buffer the chunk's tokens share.",
    ]),
]

FILES = ("fabric_engine.sv (adapters, vector buffer, memory unit, layer engine)", "fabric_sequencer.sv", "fabric_tile.sv",
         "fabric_norm.sv, fabric_recurrent.sv, fabric_ffn.sv, fabric_attention.sv, fabric_vector.sv", "fabric_memory.sv",
         "fabric_hpi.sv, fabric_cdc.sv, fabric_phy.sv", "sequencer.py (programs and operand conventions), engine.py (layouts and images), memory.py (the map)")


def md_cell(s: str) -> str:
    return s.replace("|", "\\|")


def render_markdown() -> str:
    out = ["# Block diagrams and interface contracts", "",
           "The appliance, a layer die and a tile, then the contracts of the interfaces between the blocks, "
           "transcribed from the RTL. `python -m fabric.docs.draw_blocks` redraws the diagrams and "
           "`python -m fabric.docs.blocks_doc` rewrites this file.", ""]
    for title, blocks in SECTIONS:
        out += [f"## {title}", ""]
        for b in blocks:
            if isinstance(b, str):
                out += [b, ""]
            elif b[0] == "diagram":
                out += [f"![{title}]({b[1]}.svg)", ""]
            elif b[0] == "table":
                out += ["| " + " | ".join(b[1]) + " |", "| " + " | ".join("---" for _ in b[1]) + " |"]
                out += ["| " + " | ".join(md_cell(c) for c in row) + " |" for row in b[2]]
                out.append("")
    out += ["## Sources", ""] + [f"* `{f}`" for f in FILES] + [""]
    return "\n".join(out)


PAGE_HEAD = """<title>Fabric Block Map</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--bg:#f3f5f7;--panel:#ffffff;--ink:#17202a;--muted:#4a5866;--line:#d3dae1;--accent:#1f5f7a;--mem:#9a6a12;--ctl:#3d7a2e;--ext:#5a4a8a;
 --unit-fill:#e8f1f6;--mem-fill:#fbf1de;--ctl-fill:#eaf3e6;--ext-fill:#f1eef7;--bus-fill:#dfe6ec;--box-fill:#ffffff;--stroke:#2a3a4a;--dash:#8a96a3;--sub:#3e4c5a;--grp:#5a6773}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#12181f;--panel:#1a222b;--ink:#e6ebf0;--muted:#a9b4bf;--line:#2c3743;--accent:#7cc0dd;--mem:#e0b45c;--ctl:#8fcf7a;--ext:#b9a9e6;
 --unit-fill:#1e3340;--mem-fill:#3b2f16;--ctl-fill:#1f3320;--ext-fill:#2a2540;--bus-fill:#26313c;--box-fill:#1a222b;--stroke:#c7d1db;--dash:#6b7885;--sub:#c2ccd6;--grp:#9aa6b2}}
:root[data-theme="dark"]{--bg:#12181f;--panel:#1a222b;--ink:#e6ebf0;--muted:#a9b4bf;--line:#2c3743;--accent:#7cc0dd;--mem:#e0b45c;--ctl:#8fcf7a;--ext:#b9a9e6;
 --unit-fill:#1e3340;--mem-fill:#3b2f16;--ctl-fill:#1f3320;--ext-fill:#2a2540;--bus-fill:#26313c;--box-fill:#1a222b;--stroke:#c7d1db;--dash:#6b7885;--sub:#c2ccd6;--grp:#9aa6b2}
body{background:var(--bg);color:var(--ink);font-family:'IBM Plex Sans',system-ui,sans-serif;font-size:15px;line-height:1.55;margin:0}
.wrap{max-width:1180px;margin:0 auto;padding-block:32px 64px;padding-inline:20px}
header{border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:28px}
h1{font-size:30px;font-weight:600;margin:0 0 6px;letter-spacing:-.01em;text-wrap:balance}
.eyebrow{font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
h2{font-size:21px;font-weight:600;margin:40px 0 12px;text-wrap:balance}
p{max-width:78ch;margin:0 0 14px}
.lead{font-size:16px;color:var(--muted);max-width:78ch}
.fig{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px;margin:8px 0 18px;overflow-x:auto}
.fig svg{display:block;width:100%;min-width:900px;height:auto}
.tbl{overflow-x:auto;margin:6px 0 18px;border:1px solid var(--line);border-radius:6px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;vertical-align:top;padding:8px 10px;border-bottom:1px solid var(--line)}
th{font-weight:600;background:var(--bus-fill);font-size:12.5px;letter-spacing:.02em;white-space:nowrap}
tr:last-child td{border-bottom:none}
td:first-child{font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:12.5px;white-space:nowrap}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--muted);margin:0 0 10px}
.legend span::before{content:"";display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;vertical-align:-1px;border:1px solid var(--stroke)}
.legend .u::before{background:var(--unit-fill);border-color:var(--accent)} .legend .m::before{background:var(--mem-fill);border-color:var(--mem)} .legend .c::before{background:var(--ctl-fill);border-color:var(--ctl)} .legend .x::before{background:var(--ext-fill);border-color:var(--ext)}
ul.src{font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:12.5px;color:var(--muted);padding-left:18px}
/* the diagrams */
.box{fill:var(--box-fill);stroke:var(--stroke);stroke-width:1.2} .unit{fill:var(--unit-fill);stroke:var(--accent);stroke-width:1.2}
.mem{fill:var(--mem-fill);stroke:var(--mem);stroke-width:1.2} .ctl{fill:var(--ctl-fill);stroke:var(--ctl);stroke-width:1.2}
.ext{fill:var(--ext-fill);stroke:var(--ext);stroke-width:1.2} .bus{fill:var(--bus-fill);stroke:var(--stroke);stroke-width:1.2}
.group{fill:none;stroke:var(--dash);stroke-width:1;stroke-dasharray:4 3}
svg .t{font:12px 'IBM Plex Sans',system-ui,sans-serif;fill:var(--ink)} svg .h{font:600 13px 'IBM Plex Sans',system-ui,sans-serif;fill:var(--ink)}
svg .s{font:10px 'IBM Plex Mono',ui-monospace,monospace;fill:var(--sub)} svg .g{font:600 11px 'IBM Plex Sans',system-ui,sans-serif;fill:var(--grp);letter-spacing:.06em}
svg .lbl{font:10px 'IBM Plex Mono',ui-monospace,monospace;fill:var(--sub)}
.e{stroke:var(--stroke);stroke-width:1.2;fill:none;marker-end:url(#ah)} .e2{stroke:var(--stroke);stroke-width:1.2;fill:none;marker-end:url(#ah);marker-start:url(#at)}
.ed{stroke:var(--mem);stroke-width:1.4;fill:none;marker-end:url(#ahm);marker-start:url(#atm)} .ec{stroke:var(--ctl);stroke-width:1.2;fill:none;marker-end:url(#ahc)}
svg marker path{fill:var(--stroke)} svg #ahm path, svg #atm path{fill:var(--mem)} svg #ahc path{fill:var(--ctl)}
@media (max-width:600px){h1{font-size:24px} body{font-size:14px}}
</style>
"""


def render_html() -> str:
    out = [PAGE_HEAD, '<div class="wrap"><header><div class="eyebrow">rustydc/asic, fabric</div><h1>Fabric Block Map</h1>',
           '<p class="lead">The fixed-weight inference appliance from the board down to a tile, and the input and output '
           'contracts of the blocks between, transcribed from the RTL port lists and the program conventions.</p>',
           '<div class="legend"><span class="c">control</span><span class="u">datapath units</span><span class="m">memory path</span><span class="x">external</span></div></header>']
    seen_defs = False
    for title, blocks in SECTIONS:
        out.append(f"<h2>{html.escape(title)}</h2>")
        for b in blocks:
            if isinstance(b, str):
                out.append(f"<p>{html.escape(b)}</p>")
            elif b[0] == "diagram":
                svg = (HERE / f"{b[1]}.inline.svg").read_text(encoding="utf-8")
                if seen_defs:                     # one set of marker defs per page
                    start, end = svg.find("<defs>"), svg.find("</defs>") + len("</defs>")
                    svg = svg[:start] + svg[end:]
                seen_defs = True
                out.append(f'<div class="fig">{svg}</div>')
            elif b[0] == "table":
                rows = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>" for row in b[2])
                head = "".join(f"<th>{html.escape(h)}</th>" for h in b[1])
                out.append(f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>')
    out.append("<h2>Sources</h2><ul class=\"src\">" + "".join(f"<li>{html.escape(f)}</li>" for f in FILES) + "</ul></div>")
    return "\n".join(out)


def main() -> None:
    (HERE / "BLOCKS.md").write_text(render_markdown(), encoding="utf-8")
    print("wrote", HERE / "BLOCKS.md")
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(render_html(), encoding="utf-8")
        print("wrote", sys.argv[1])


if __name__ == "__main__":
    main()
