// The token sequencer: a microcoded issue engine that runs a layer's
// program (fabric/sequencer.py) over the tiles, the vector units and the
// DMAs, several tokens of different contexts interleaved in one stream.
//
// The program is a list of steps.  Each step is a command to one unit
// (unit id, engine, a length, a 32-bit argument and four 30-bit address
// operands), up to six buffer ids it consumes, up to two it produces
// (each either a write or a contribution to a whole vector), and a last
// flag.  The controller issues in program order.  Per buffer id it keeps
// the number of outstanding writers and readers; the head step issues when
// every consumed buffer has no outstanding writer, every produced buffer
// no outstanding reader and (for a write, not a contribution) no
// outstanding writer, and the unit reports the addressed engine free.  On
// issue the step's ids are remembered against its engine port and counted;
// a unit returns the tag on its engine's done port and the counts are
// released, forwarded into the same cycle's issue check.
//
// The release is bounded.  Taking every completion in a cycle means every
// engine port's ids may decrement the same counter, which is NU*NE*(NC+NP)
// read-modify-writes of the whole counter array chained in one cycle --
// 55,000 cells and seven gigabytes before yosys is killed, and the reason
// this module did not map.  So NREL completions drain a cycle, lowest
// engine port first, and a port whose release has not drained can be given
// no new command: it still holds the buffers it must return.  Nothing else
// is needed to make the bound safe, because the ids live with the port --
// which is also why a port with a command outstanding is given no second
// one.  A unit reports done a cycle after it drops its ready, so without
// that the controller handed a port its next command first and the
// completion returned the newer command's ids for the older one.
//
// Timing, which fabric/sequencer.py reproduces cycle for cycle: a step
// issues at the earliest cycle that is after the previous issue, at or
// after the cycle its last dependency's release drained, and at or after
// the cycle its engine reported free.  A drain with nothing held is the
// cycle the completion arrives, which is one cycle after the unit's last
// working cycle, so an unheld release is what it always was.

`timescale 1ns/1ps
`default_nettype none

module fabric_sequencer #(
    parameter int NU        = 10,               // units
    parameter int NE        = 4,                // engines per unit at most
    parameter int DEPTH     = 1024,             // program steps at most
    parameter int NID       = 256,              // buffer ids
    parameter int CW        = 6,                // counter width: outstanding writers or readers of one buffer
    parameter int NREL      = 1,                // completions drained a cycle (fabric.sequencer.RELEASES)
    parameter     PROG_FILE = "program.hex"
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 start,
    input  wire [15:0]          n_steps,
    output reg                  running,
    output reg                  done,
    // the command bus: one unit addressed per cycle
    output wire [NU-1:0]        cmd_valid,
    output wire [3:0]           cmd_engine,
    output wire [15:0]          cmd_len,
    output wire [29:0]          cmd_src,
    output wire [29:0]          cmd_dst,
    output wire [29:0]          cmd_a2,
    output wire [29:0]          cmd_a3,
    output wire [31:0]          cmd_arg,
    output wire [7:0]           cmd_tag,
    input  wire [NU-1:0]        cmd_ready,      // the addressed engine of that unit is free
    // completions, one port per engine
    input  wire [NU*NE-1:0]     done_valid,
    input  wire [NU*NE*8-1:0]   done_tag
);
    localparam int NC = 6, NP = 2, IDB = 64 + 4 * 30;   // the ids follow the four address operands
    reg [255:0] prog [0:DEPTH-1];
    initial if (PROG_FILE != "") $readmemh(PROG_FILE, prog);

    reg  [15:0]  pc;
    wire [255:0] cur      = prog[pc];
    wire [3:0]   cur_unit = cur[3:0];
    wire         cur_last = cur[8];
    assign cmd_engine = cur[7:4];
    assign cmd_len    = cur[31:16];
    assign cmd_arg    = cur[63:32];
    assign cmd_src    = cur[64 +: 30];
    assign cmd_dst    = cur[94 +: 30];
    assign cmd_a2     = cur[124 +: 30];
    assign cmd_a3     = cur[154 +: 30];
    assign cmd_tag    = pc[7:0];
    wire [7:0]   cur_c [0:NC-1];
    wire [7:0]   cur_p [0:NP-1];
    wire [NP-1:0] cur_contrib = cur[IDB + 8*(NC+NP) +: NP];
    genvar gi;
    generate
        for (gi = 0; gi < NC; gi = gi + 1) begin : g_c assign cur_c[gi] = cur[IDB + 8*gi +: 8]; end
        for (gi = 0; gi < NP; gi = gi + 1) begin : g_p assign cur_p[gi] = cur[IDB + 8*NC + 8*gi +: 8]; end
    endgenerate

    // The ids of the command each engine is running, for the release at
    // completion.  Held per engine port rather than per tag: a table of 256
    // entries read by every done port's tag is forty 256-to-1 muxes of the
    // whole id set, repeated at each of the issue check's call sites, which
    // does not map.  The port is known at issue, so the release reads a
    // register.  The tag still marks the step live, so a program longer than
    // 256 steps cannot have two of the same tag in flight.
    localparam int NPORT = NU * NE;
    reg [NC*8-1:0] slot_c [0:NPORT-1];
    reg [NP*8-1:0] slot_p [0:NPORT-1];
    reg [7:0]      slot_tag [0:NPORT-1];
    reg [255:0]    tab_live;
    wire [$clog2(NPORT)-1:0] cur_port = cur_unit * NE + cmd_engine;

    // Which completions drain this cycle: the NREL lowest ports of what a
    // unit is reporting now and what an earlier cycle could not take.  The
    // ids come out with them, so the rest of the module sees NREL releases
    // rather than NPORT of them, and a held port's slot is not overwritten
    // because a held port is not given a command.
    reg [NPORT-1:0]      pend;
    wire [NPORT-1:0]     want_rel = done_valid | pend;
    reg  [NPORT-1:0]     rel_now;
    reg  [NREL-1:0]      rel_en;
    reg  [NREL*NC*8-1:0] rel_c;                   // consumed and produced ids at the
    reg  [NREL*NC*8-1:0] rel_p;                   // same stride, so one scan serves both
    reg  [NREL*8-1:0]    rel_tag;
    // "The lowest port still to drain" is x & -x, not a scan.  Written as a
    // scan it is NPORT stages of "nothing found yet", each selecting that
    // port's ids, which is a chain of forty gates of fanout sixty -- 15 of the
    // module's 21 nanoseconds.  One-hot, the select is an or of masks, which
    // is associative and comes out of synthesis as a tree.
    reg [NPORT-1:0] remaining, one;
    reg [NC*8-1:0]  sel_c;
    reg [NP*8-1:0]  sel_p;
    reg [7:0]       sel_tag;
    integer q, r;
    always @* begin
        rel_now = 0;
        rel_en  = 0;
        rel_c   = {(NREL*NC){8'hFF}};
        rel_p   = {(NREL*NC){8'hFF}};
        rel_tag = 0;
        remaining = want_rel;
        for (r = 0; r < NREL; r = r + 1) begin
            one = remaining & (~remaining + {{(NPORT-1){1'b0}}, 1'b1});
            sel_c = 0; sel_p = 0; sel_tag = 0;
            for (q = 0; q < NPORT; q = q + 1) begin
                sel_c   = sel_c   | (slot_c[q]   & {(NC*8){one[q]}});
                sel_p   = sel_p   | (slot_p[q]   & {(NP*8){one[q]}});
                sel_tag = sel_tag | (slot_tag[q] & {8{one[q]}});
            end
            if (|one) begin
                rel_en[r] = 1'b1;
                rel_c[r*NC*8 +: NC*8] = sel_c;
                rel_p[r*NC*8 +: NP*8] = sel_p;
                rel_tag[r*8 +: 8]     = sel_tag;
            end
            rel_now   = rel_now | one;
            remaining = remaining & ~one;
        end
    end
    wire [NPORT-1:0] held = want_rel & ~rel_now;
    // A port's record is one command, so it can be given no second one while
    // the first is outstanding: a unit reports done a cycle after it drops
    // its ready, and without this the completion returned the newer
    // command's ids for the older one -- buffers the newer command was still
    // reading.  A port whose completion drains this cycle is free, because
    // the drain reads the register and the issue writes it.
    reg  [NPORT-1:0] busy;
    wire [NPORT-1:0] blocked = busy & ~rel_now;

    // Outstanding writers and readers per buffer.  The issue check sees the
    // head step's ids with this cycle's drains forwarded; the counters
    // themselves are updated by the drains and then the issue.
    reg [CW-1:0] wr_cnt [0:NID-1];
    reg [CW-1:0] rd_cnt [0:NID-1];
    integer p, k, m, id, x;
    reg signed [CW:0] dr, dw;                            // a counter's move this cycle

    // The drained ids travel as arguments, not as a reference to rel_c and
    // rel_p: what a function reads is not in an always @* block's sensitivity,
    // only what it is passed, and a stale deps_ok deadlocks.
    function automatic [CW-1:0] released(input [7:0] buf_id, input integer n,
                                         input [NREL-1:0] en, input [NREL*NC*8-1:0] ids);
        integer y, j;
        begin
            released = 0;
            for (y = 0; y < NREL; y = y + 1)
                if (en[y])
                    for (j = 0; j < NC; j = j + 1)
                        if (j < n && ids[(y*NC + j)*8 +: 8] == buf_id) released = released + 1'b1;
        end
    endfunction

    reg deps_ok;
    reg [7:0] n_done_now;
    always @* begin
        deps_ok = 1'b1;
        for (k = 0; k < NC; k = k + 1)
            if (cur_c[k] != 8'hFF && wr_cnt[cur_c[k]] != released(cur_c[k], NP, rel_en, rel_p)) deps_ok = 1'b0;
        for (k = 0; k < NP; k = k + 1)
            if (cur_p[k] != 8'hFF && (rd_cnt[cur_p[k]] != released(cur_p[k], NC, rel_en, rel_c)
                                      || (!cur_contrib[k] && wr_cnt[cur_p[k]] != released(cur_p[k], NP, rel_en, rel_p)))) deps_ok = 1'b0;
        n_done_now = 0;
        for (p = 0; p < NPORT; p = p + 1) if (done_valid[p]) n_done_now = n_done_now + 1'b1;
    end

    reg  [9:0]  outstanding;
    reg         finishing;
    wire        want  = running && !finishing && deps_ok && !tab_live[pc[7:0]] && !blocked[cur_port];
    assign cmd_valid = want ? (1 << cur_unit) : 0;
    wire        issue = want && cmd_ready[cur_unit];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc <= 0; running <= 1'b0; done <= 1'b0; outstanding <= 0; finishing <= 1'b0; tab_live <= 0; pend <= 0; busy <= 0;
            for (id = 0; id < NID; id = id + 1) begin wr_cnt[id] = 0; rd_cnt[id] = 0; end
        end else begin
            done <= 1'b0;
            if (start && !running) begin
                pc <= 0; running <= 1'b1; outstanding <= 0; finishing <= 1'b0; tab_live <= 0; pend <= 0; busy <= 0;
                for (id = 0; id < NID; id = id + 1) begin wr_cnt[id] = 0; rd_cnt[id] = 0; end
            end else begin
                // Not gated on `running`: when it is low nothing issues and no
                // unit is busy, so none of this moves anyway, and the gate cost
                // more than everything else here -- one flop's enable reaching
                // every counter and every live bit was 835 loads, and the three
                // gates behind it were 42 of the module's 43 nanoseconds.
                // The drains: NREL completions return their buffers; the rest wait.
                pend <= held;
                for (p = 0; p < NPORT; p = p + 1)
                    if (rel_now[p]) busy[p] <= 1'b0;
                for (x = 0; x < NREL; x = x + 1)
                    if (rel_en[x]) tab_live[rel_tag[x*8 +: 8]] <= 1'b0;
                // Every counter moves by what this cycle did to it, once.
                // Written as the drain's decrements and then the issue's
                // increments, each a read-modify-write of a 256-entry array at
                // a computed index, it was sixteen of those chained: 118 gates
                // from a done port to a counter, and the whole of this
                // module's path.  The ids in play are at most NREL*(NC+NP)
                // returning and NC+NP taken, so a counter's delta is a couple
                // of dozen compares against them, and all of them in parallel.
                for (id = 0; id < NID; id = id + 1) begin
                    dr = 0;
                    dw = 0;
                    for (x = 0; x < NREL; x = x + 1)
                        if (rel_en[x]) begin
                            for (k = 0; k < NC; k = k + 1)
                                if (rel_c[x*NC*8 + k*8 +: 8] == id[7:0]) dr = dr - 1;
                            for (k = 0; k < NP; k = k + 1)
                                if (rel_p[x*NC*8 + k*8 +: 8] == id[7:0]) dw = dw - 1;
                        end
                    if (issue) begin
                        for (k = 0; k < NC; k = k + 1)
                            if (cur_c[k] == id[7:0]) dr = dr + 1;
                        for (k = 0; k < NP; k = k + 1)
                            if (cur_p[k] == id[7:0]) dw = dw + 1;
                    end
                    if (dr != 0) rd_cnt[id] = rd_cnt[id] + dr[CW-1:0];
                    if (dw != 0) wr_cnt[id] = wr_cnt[id] + dw[CW-1:0];
                end
                outstanding <= outstanding + {9'd0, issue} - {2'd0, n_done_now};
                if (issue) begin
                    busy[cur_port] <= 1'b1;                  // after the drains: a port reissued in its drain cycle stays busy
                    slot_c[cur_port] <= cur[IDB +: NC*8];
                    slot_p[cur_port] <= cur[IDB + 8*NC +: NP*8];
                    slot_tag[cur_port] <= pc[7:0];
                    tab_live[pc[7:0]] <= 1'b1;
                    pc <= pc + 1'b1;
                    if (cur_last || pc + 1 == n_steps) finishing <= 1'b1;
                end
                if (running && finishing && outstanding == n_done_now && held == 0) begin
                    running <= 1'b0; done <= 1'b1;
                end
            end
        end
    end

`ifndef FABRIC_SYNTH
    // The port's own record of the command and the tag the unit returns are
    // the same command: the release reads the register, not the reply.  This
    // is what `busy` is for, and it was firing before there was one.
    always @(posedge clk)
        if (rst_n && running)
            for (m = 0; m < NPORT; m = m + 1)
                if (done_valid[m] && (!busy[m] || done_tag[m*8 +: 8] !== slot_tag[m]))
                    $display("FAIL: port %0d returned tag %0d, it was given %0d%s", m, done_tag[m*8 +: 8], slot_tag[m],
                             busy[m] ? "" : " and had no command outstanding");
`endif
endmodule

`default_nettype wire
