// The token sequencer: a microcoded issue engine that runs a die's
// programs (fabric/sequencer.py) over the tiles, the vector units and the
// DMAs, for up to LN tokens of different contexts at once, one a lane.
//
// The program is a list of steps.  Each step is a command to one unit
// (unit id, engine, a length, a 32-bit argument and four 30-bit address
// operands), up to six buffer ids it consumes, up to two it produces
// (each either a write or a contribution to a whole vector), and a last
// flag.  A lane issues its steps in program order.  Per buffer id it keeps
// the number of outstanding writers and readers; the head step may issue
// when every consumed buffer has no outstanding writer, every produced
// buffer no outstanding reader and (for a write, not a contribution) no
// outstanding writer, and the unit reports the addressed engine free.  On
// issue the step's ids are remembered against its engine port and counted;
// a unit returns the tag on its engine's done port and the counts are
// released.
//
// The lanes.  A lane is given runs -- a program in its own store, the layer
// it is and the page its part of the context's slot is at -- and fetches
// them one after another with nothing between, so a token's four layers are
// one program to it.  Its buffers are its own, so are its counters, and so
// its steps wait on nothing of another lane's but the engines they share.
// Of the lanes whose head step may issue this cycle the oldest does: a lane
// is as old as its token, the first run pushed after it last reported done.
// When one lane's head waits -- the next layer's first step on the last
// one's residual, a head's update on its state read -- another's goes, which
// a single in-order program could not do: its next step was behind the one
// waiting.  A lane reports done when everything it was given has drained.
// (fabric.sequencer.schedule_lanes; one lane is the in-order controller.)
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
// issues at the earliest cycle that is after the previous issue of any lane,
// at or after the cycle its last dependency's release drained, and at or
// after the cycle its engine reported free, if no older lane's step issues
// then.  A drain with nothing held is the cycle the completion arrives,
// which is one cycle after the unit's last working cycle, so an unheld
// release is what it always was.  A lane's first step may issue three
// cycles after its first run is pushed: the run is taken, fetched, queued.

`timescale 1ns/1ps
`default_nettype none

module fabric_sequencer #(
    parameter int NU        = 10,               // units
    parameter int NE        = 4,                // engines per unit at most
    parameter int LN        = 4,                // lanes, at most four (fabric.sequencer.LANES)
    parameter int DEPTH     = 4096,             // a lane's program store: every program the lane runs
    parameter int NID       = 128,              // buffer ids a lane: a lane's programs use 85 at the 9B geometry (fabric.sequencer.LANE_IDS)
    parameter int CW        = 6,                // counter width: outstanding writers or readers of one buffer
    parameter int NREL      = 1,                // completions drained a cycle (fabric.sequencer.RELEASES)
    parameter int RQ        = 4,                // runs a lane holds: a token's layers
    parameter     PROG_FILE = "program.hex"     // lane 0's store; lane l's is lane<l>_<PROG_FILE>
) (
    input  wire                 clk,
    input  wire                 rst_n,
    // A run for a lane: its first step and length in the lane's store, the
    // layer it is (the units' bank of constants) and the page its part of
    // the slot starts at, from the slot's first.
    input  wire                 push,
    input  wire [1:0]           push_lane,
    input  wire [15:0]          push_pc,
    input  wire [15:0]          push_steps,
    input  wire [1:0]           push_layer,
    input  wire [20:0]          push_page,
    output wire [LN-1:0]        push_room,
    output reg  [LN-1:0]        lane_busy,      // given runs it has not reported done
    output reg  [LN-1:0]        lane_done,      // a cycle: everything the lane was given has drained
    output wire                 running,
    // the command bus: one unit addressed per cycle, valid only on issue
    output wire [NU-1:0]        cmd_valid,
    output wire [3:0]           cmd_engine,
    output wire [15:0]          cmd_len,
    output wire [29:0]          cmd_src,
    output wire [29:0]          cmd_dst,
    output wire [29:0]          cmd_a2,
    output wire [29:0]          cmd_a3,
    output wire [31:0]          cmd_arg,
    output wire [7:0]           cmd_tag,
    output wire [1:0]           cmd_lane,
    output wire [1:0]           cmd_layer,
    output wire [20:0]          cmd_page,
    input  wire [NU*NE-1:0]     port_ready,     // each engine port free for a command
    // completions, one port per engine
    input  wire [NU*NE-1:0]     done_valid,
    input  wire [NU*NE*8-1:0]   done_tag
);
    localparam int NC = 6, NP = 2, IDB = 64 + 4 * 30;   // the ids follow the four address operands
    localparam int NPORT = NU * NE;
    localparam int PW = $clog2(NPORT);

    // ---------------------------------------------------------------------
    // The lanes' heads.
    // ---------------------------------------------------------------------
    wire [LN-1:0]  head_v, lane_idle, win;
    wire [255:0]   head       [0:LN-1];
    wire [1:0]     head_layer [0:LN-1];
    wire [20:0]    head_page  [0:LN-1];
    genvar gl;
    generate
        for (gl = 0; gl < LN; gl = gl + 1) begin : g_lane
            fabric_seq_lane #(.LANE(gl), .DEPTH(DEPTH), .RQ(RQ), .PROG_FILE(PROG_FILE)) u (
                .clk(clk), .rst_n(rst_n),
                .push(push && push_lane == gl), .push_pc(push_pc), .push_steps(push_steps), .push_layer(push_layer),
                .push_page(push_page), .room(push_room[gl]),
                .issue(win[gl]), .head_v(head_v[gl]), .head(head[gl]), .head_layer(head_layer[gl]), .head_page(head_page[gl]),
                .idle(lane_idle[gl]));
        end
    endgenerate
    assign running = |lane_busy;

    // ---------------------------------------------------------------------
    // The ports' records and the drain.
    // ---------------------------------------------------------------------
    // The ids of the command each engine is running, for the release at
    // completion.  Held per engine port rather than per tag: a table of 256
    // entries read by every done port's tag is forty 256-to-1 muxes of the
    // whole id set, repeated at each of the issue check's call sites, which
    // does not map.  The port is known at issue, so the release reads a
    // register.  The tag still marks the step live, so a program longer than
    // 256 steps cannot have two of the same tag in flight.  The port also
    // records the lane, whose counters the release returns to.
    reg [NC*8-1:0] slot_c    [0:NPORT-1];
    reg [NP*8-1:0] slot_p    [0:NPORT-1];
    reg [7:0]      slot_tag  [0:NPORT-1];
    reg [1:0]      slot_lane [0:NPORT-1];
    reg [255:0]    tab_live;
    reg [15:0]     icount;                              // steps issued: the tag is its low byte

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
    reg  [NREL*2-1:0]    rel_lane;
    // "The lowest port still to drain" is one-hot, not a scan.  Written as a
    // scan it is NPORT stages of "nothing found yet", each selecting that
    // port's ids, which is a chain of forty gates of fanout sixty -- 15 of the
    // module's 21 nanoseconds.  One-hot, the select is an or of masks, which
    // is associative and comes out of synthesis as a tree.
    // The one-hot is a prefix or, not ``x & -x``: the incrementer's carry is
    // a ripple forty bits long and synthesis leaves it one, which measured
    // twenty gates and 915 ps of the release's path.  Doubling the span each
    // pass answers "is any lower port set" in ceil(log2 NPORT) levels.
    reg [NPORT-1:0] remaining, one, lower;
    integer sh;
    reg [NC*8-1:0]  sel_c;
    reg [NP*8-1:0]  sel_p;
    reg [7:0]       sel_tag;
    reg [1:0]       sel_lane;
    integer q, r;
    always @* begin
        rel_now  = 0;
        rel_en   = 0;
        rel_c    = {(NREL*NC){8'hFF}};
        rel_p    = {(NREL*NC){8'hFF}};
        rel_tag  = 0;
        rel_lane = 0;
        remaining = want_rel;
        for (r = 0; r < NREL; r = r + 1) begin
            lower = {remaining[NPORT-2:0], 1'b0};              // lower[i] = remaining[i-1]
            for (sh = 1; sh < NPORT; sh = sh * 2) lower = lower | (lower << sh);
            one = remaining & ~lower;
            sel_c = 0; sel_p = 0; sel_tag = 0; sel_lane = 0;
            for (q = 0; q < NPORT; q = q + 1) begin
                sel_c    = sel_c    | (slot_c[q]    & {(NC*8){one[q]}});
                sel_p    = sel_p    | (slot_p[q]    & {(NP*8){one[q]}});
                sel_tag  = sel_tag  | (slot_tag[q]  & {8{one[q]}});
                sel_lane = sel_lane | (slot_lane[q] & {2{one[q]}});
            end
            if (|one) begin
                rel_en[r] = 1'b1;
                rel_c[r*NC*8 +: NC*8] = sel_c;
                rel_p[r*NC*8 +: NP*8] = sel_p;
                rel_tag[r*8 +: 8]     = sel_tag;
                rel_lane[r*2 +: 2]    = sel_lane;
            end
            rel_now   = rel_now | one;
            remaining = remaining & ~one;
        end
    end
    wire [NPORT-1:0] held = want_rel & ~rel_now;
    // The drain, registered.  Picking the port, reading its ids and then
    // moving 256 counters by them was one cycle, and it is the only structure
    // in this module anywhere near the clock: its two endpoints, rd_cnt and
    // wr_cnt, were 2,317 and 2,245 ps with nothing else close.  The pick and
    // the read are one cycle now and the counters the next, so a release
    // lands the cycle after the completion rather than in it -- which is what
    // `release[d] < cycle` says in fabric/sequencer.py, and what lets the
    // issue check read the counters plainly instead of forwarding this
    // cycle's drains into them.
    reg [NREL-1:0]      d_en;
    reg [NREL*NC*8-1:0] d_c, d_p;
    reg [NREL*8-1:0]    d_tag;
    reg [NREL*2-1:0]    d_lane;
    reg [NPORT-1:0]     d_now;
    // `d_en` gates every one of the counter updates, and registering it is
    // exactly what the mapper cannot buffer: one flop at 495 loads and 739
    // fF, 1,857 of this module's 2,833 ps spent before any logic runs.  It is
    // the shape `fetched_v` had and the fix is the same -- a copy per slice
    // of the counters, each taking the same combinational pick, so every copy
    // and `d_en` are one register.
    localparam int NDC = 16;                       // copies, one per slice of a lane's counters
    localparam int IPC = NID / NDC;                // ids to a slice
    wire [NDC*NREL-1:0] d_en_c;
    genvar gdc, gdx;
    generate
        for (gdc = 0; gdc < NDC; gdc = gdc + 1) begin : g_dec
            for (gdx = 0; gdx < NREL; gdx = gdx + 1) begin : g_dex
                fabric_seq_copy u_d (.clk(clk), .rst_n(rst_n), .d(rel_en[gdx]),
                                     .q(d_en_c[gdc*NREL + gdx]));
            end
        end
    endgenerate
    // A port's record is one command, so it can be given no second one while
    // the first is outstanding: a unit reports done a cycle after it drops
    // its ready, and without this the completion returned the newer
    // command's ids for the older one -- buffers the newer command was still
    // reading.  A port whose completion drains this cycle is free, because
    // the drain reads the register and the issue writes it.
    // A port is free in the cycle its drain applies, not the cycle it is
    // picked: the drain reads `d_c` and `d_p`, which were registered from the
    // slot a cycle earlier, so an issue may overwrite the slot underneath it.
    // Forgiving it a cycle earlier than that -- when the pick happens -- would
    // hand the port a command before its ids had been taken.
    reg  [NPORT-1:0] busy;
    wire [NPORT-1:0] blocked = busy & ~d_now;

    // ---------------------------------------------------------------------
    // The issue: each lane's check against its own counters, then the oldest.
    // ---------------------------------------------------------------------
    // Outstanding writers and readers per buffer, a set a lane.  The drains
    // are registered, so the counters already hold every release that has
    // landed and the check is a read.
    reg [CW-1:0] wr_cnt [0:LN*NID-1];
    reg [CW-1:0] rd_cnt [0:LN*NID-1];
    // older[m][l]: lane m's token came before lane l's.
    reg [LN-1:0] older [0:LN-1];
    reg [LN-1:0] deps_ok, cand;
    reg [PW-1:0] port_of [0:LN-1];
    integer l, k, m, p, id, x, w;
    reg [7:0] cid, pid;
    always @* begin
        for (l = 0; l < LN; l = l + 1) begin
            port_of[l] = head[l][3:0] * NE + head[l][7:4];
            deps_ok[l] = 1'b1;
            for (k = 0; k < NC; k = k + 1) begin
                cid = head[l][IDB + 8*k +: 8];
                if (cid != 8'hFF && wr_cnt[l*NID + cid] != 0) deps_ok[l] = 1'b0;
            end
            for (k = 0; k < NP; k = k + 1) begin
                pid = head[l][IDB + 8*NC + 8*k +: 8];
                if (pid != 8'hFF && (rd_cnt[l*NID + pid] != 0
                                     || (!head[l][IDB + 8*(NC+NP) + k] && wr_cnt[l*NID + pid] != 0))) deps_ok[l] = 1'b0;
            end
            cand[l] = head_v[l] && deps_ok[l] && !blocked[port_of[l]] && port_ready[port_of[l]] && !tab_live[icount[7:0]];
        end
    end
    genvar gw;
    generate
        for (gw = 0; gw < LN; gw = gw + 1) begin : g_win
            wire [LN-1:0] elder;
            genvar gm;
            for (gm = 0; gm < LN; gm = gm + 1) begin : g_b
                assign elder[gm] = older[gm][gw];
            end
            assign win[gw] = cand[gw] && !(|(cand & elder));
        end
    endgenerate
    wire       issue = |win;
    reg  [1:0] sel;
    always @* begin
        sel = 0;
        for (m = 0; m < LN; m = m + 1) if (win[m]) sel = 2'(m);
    end
    wire [255:0] cur      = head[sel];
    wire [3:0]   cur_unit = cur[3:0];
    wire [PW-1:0] cur_port = port_of[sel];
    wire [NP-1:0] cur_contrib = cur[IDB + 8*(NC+NP) +: NP];
    assign cmd_valid  = issue ? (1 << cur_unit) : 0;
    assign cmd_engine = cur[7:4];
    assign cmd_len    = cur[31:16];
    assign cmd_arg    = cur[63:32];
    assign cmd_src    = cur[64 +: 30];
    assign cmd_dst    = cur[94 +: 30];
    assign cmd_a2     = cur[124 +: 30];
    assign cmd_a3     = cur[154 +: 30];
    assign cmd_tag    = icount[7:0];
    assign cmd_lane   = sel;
    assign cmd_layer  = head_layer[sel];
    assign cmd_page   = head_page[sel];

    // Commands issued and not yet returned, a count a lane.  Counted against
    // the drains, which are at most NREL a cycle, and not against the
    // completions, which are up to NPORT: a population count of the forty
    // done ports is forty chained increments.  A command that has completed
    // and not drained is still outstanding, so reaching zero is every step
    // the lane issued having returned its buffers.
    reg  [9:0]  outstanding [0:LN-1];
    reg  [9:0]  out_next    [0:LN-1];
    always @* begin
        for (l = 0; l < LN; l = l + 1) begin
            out_next[l] = outstanding[l] + {9'd0, win[l]};
            for (x = 0; x < NREL; x = x + 1)
                if (d_en[x] && d_lane[x*2 +: 2] == 2'(l)) out_next[l] = out_next[l] - 1'b1;
        end
    end

    reg signed [CW:0] dr, dw;                            // a counter's move this cycle
    reg touch;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lane_busy <= 0; lane_done <= 0; icount <= 0; tab_live <= 0; pend <= 0; busy <= 0;
            d_en <= 0; d_now <= 0; d_lane <= 0;
            for (l = 0; l < LN; l = l + 1) begin
                outstanding[l] <= 0;
                for (m = 0; m < LN; m = m + 1) older[l][m] <= (l < m);
            end
            for (id = 0; id < LN*NID; id = id + 1) begin wr_cnt[id] = 0; rd_cnt[id] = 0; end
        end else begin
            lane_done <= 0;
            // The drains: NREL completions return their buffers; the rest wait.
            pend <= held;
            // This cycle's pick, for the next one to apply.
            d_en <= rel_en; d_c <= rel_c; d_p <= rel_p; d_tag <= rel_tag; d_lane <= rel_lane; d_now <= rel_now;
            for (p = 0; p < NPORT; p = p + 1)
                if (d_now[p]) busy[p] <= 1'b0;
            for (x = 0; x < NREL; x = x + 1)
                if (d_en[x]) tab_live[d_tag[x*8 +: 8]] <= 1'b0;
            // Every counter moves by what this cycle did to it, once: the ids
            // in play are at most NREL*(NC+NP) returning and NC+NP taken, so
            // a counter's delta is a couple of dozen compares against them,
            // all of them in parallel.  A lane neither issuing nor draining
            // this cycle has nothing to move.
            for (l = 0; l < LN; l = l + 1) begin
                touch = win[l];
                for (x = 0; x < NREL; x = x + 1) if (d_en[x] && d_lane[x*2 +: 2] == 2'(l)) touch = 1'b1;
                if (touch)
                    for (id = 0; id < NID; id = id + 1) begin
                        dr = 0;
                        dw = 0;
                        for (x = 0; x < NREL; x = x + 1)
                            if (d_en_c[(id / IPC) * NREL + x] && d_lane[x*2 +: 2] == 2'(l)) begin
                                for (k = 0; k < NC; k = k + 1)
                                    if (d_c[x*NC*8 + k*8 +: 8] == id[7:0]) dr = dr - 1;
                                for (k = 0; k < NP; k = k + 1)
                                    if (d_p[x*NC*8 + k*8 +: 8] == id[7:0]) dw = dw - 1;
                            end
                        if (win[l]) begin
                            for (k = 0; k < NC; k = k + 1)
                                if (head[l][IDB + 8*k +: 8] == id[7:0]) dr = dr + 1;
                            for (k = 0; k < NP; k = k + 1)
                                if (head[l][IDB + 8*NC + 8*k +: 8] == id[7:0]) dw = dw + 1;
                        end
                        if (dr != 0) rd_cnt[l*NID + id] = rd_cnt[l*NID + id] + dr[CW-1:0];
                        if (dw != 0) wr_cnt[l*NID + id] = wr_cnt[l*NID + id] + dw[CW-1:0];
                    end
                outstanding[l] <= out_next[l];
                // Done: everything given fetched and issued, and returned.
                if (lane_busy[l] && lane_idle[l] && out_next[l] == 10'd0) begin
                    lane_busy[l] <= 1'b0; lane_done[l] <= 1'b1;
                end
            end
            if (issue) begin
                busy[cur_port] <= 1'b1;                  // after the drains: a port reissued in its drain cycle stays busy
                slot_c[cur_port] <= cur[IDB +: NC*8];
                slot_p[cur_port] <= cur[IDB + 8*NC +: NP*8];
                slot_tag[cur_port] <= icount[7:0];
                slot_lane[cur_port] <= sel;
                tab_live[icount[7:0]] <= 1'b1;
                icount <= icount + 1'b1;
            end
            // A run for an idle lane starts its token: the lane is the youngest.
            if (push) begin
                lane_busy[push_lane] <= 1'b1;
                if (!lane_busy[push_lane])
                    for (m = 0; m < LN; m = m + 1)
                        if (m != push_lane) begin older[m][push_lane] <= 1'b1; older[push_lane][m] <= 1'b0; end
            end
        end
    end

`ifndef FABRIC_SYNTH
    // The port's own record of the command and the tag the unit returns are
    // the same command: the release reads the register, not the reply.  This
    // is what `busy` is for, and it was firing before there was one.
    always @(posedge clk)
        if (rst_n)
            for (p = 0; p < NPORT; p = p + 1)
                if (done_valid[p] && (!busy[p] || done_tag[p*8 +: 8] !== slot_tag[p]))
                    $display("FAIL: port %0d returned tag %0d, it was given %0d%s", p, done_tag[p*8 +: 8], slot_tag[p],
                             busy[p] ? "" : " and had no command outstanding");
    always @(posedge clk)
        if (rst_n && push && !push_room[push_lane]) $display("FAIL: a run pushed into lane %0d, which is full", push_lane);
`endif
endmodule

// A lane: its program store, the runs it has been given, and the fetch that
// keeps two steps in hand.
//
// The fetch runs ahead of the check.  Read where it is used, the store is a
// DEPTH-entry mux of 256-bit words in the same cycle as the issue check --
// which indexes a counter per buffer id the step names -- and the two
// together were 37 levels of logic and the whole of the sequencer's path.
// It is a memory, which wants its address a cycle early anyway, so `fpc`
// runs ahead on its own and the head step and the one after it wait in `q0`
// and `q1`.  Two deep because a step may issue every cycle and a one-deep
// queue would give up every other one.  At the end of a run the next one's
// first step is fetched in the same way, so a lane's runs follow one another
// as one program.
module fabric_seq_lane #(
    parameter int LANE  = 0,
    parameter int DEPTH = 4096,
    parameter int RQ    = 4,
    parameter     PROG_FILE = "program.hex"
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          push,
    input  wire [15:0]   push_pc,
    input  wire [15:0]   push_steps,
    input  wire [1:0]    push_layer,
    input  wire [20:0]   push_page,
    output wire          room,
    input  wire          issue,
    output wire          head_v,
    output wire [255:0]  head,
    output wire [1:0]    head_layer,
    output wire [20:0]   head_page,
    output wire          idle            // nothing left to fetch or issue
);
    localparam int PAW = (DEPTH > 1) ? $clog2(DEPTH) : 1;
    localparam int RW  = (RQ > 1) ? $clog2(RQ) : 1;

    reg [15:0] rq_pc    [0:RQ-1];
    reg [15:0] rq_steps [0:RQ-1];
    reg [22:0] rq_side  [0:RQ-1];            // {layer, page}
    reg [RW-1:0] rq_rd, rq_wr;
    reg [RW:0]   rq_n;
    assign room = rq_n != (RW+1)'(RQ);

    reg  [15:0]  fpc, fend;                  // the step being fetched, and the one after its run
    reg  [22:0]  f_side;
    reg  [255:0] q0, q1;
    reg  [22:0]  s0, s1, fs;                 // the queue's runs' sides, and the side of the word on its way
    reg  [1:0]   qn;                         // steps in hand
    reg          fetched_v;                  // the memory answers this cycle
    wire [255:0] fetched;
    wire         more  = fpc != fend;
    wire         can   = more || (rq_n != 0);
    // Room for what a fetch started now would bring: the memory answers the
    // cycle after its address, so one step may already be on its way.
    wire [2:0]   after = {1'b0, qn} + {2'b0, fetched_v} - {2'b0, issue};
    wire         fetch = can && (after < 3'd2);
    wire [15:0]  faddr = more ? fpc : rq_pc[rq_rd];
    wire [22:0]  fside = more ? f_side : rq_side[rq_rd];
    assign head_v     = qn != 2'd0;
    assign head       = q0;
    assign head_layer = s0[22:21];
    assign head_page  = s0[20:0];
    assign idle       = !can && qn == 2'd0 && !fetched_v;

    // `fetched_v` selects what each of the queue's 512 bits takes, so one flop
    // held 490 loads and 798 fF: two nanoseconds of clock-to-output.  The
    // mapper buffers what it drives and cannot help the flop itself, so the
    // flop is replicated, a copy per slice of the queue -- the same trick as
    // fabric_const_copy for a lane's shift amount and fabric_strobe_copy for
    // the tile's strobe.  Every copy takes `fetch`, so all of them and
    // `fetched_v` (which keeps the few loads that are left) always agree.
    localparam int NQC = 16;                 // copies, one per queue slice
    localparam int QSL = 256 / NQC;
    wire [NQC-1:0] fetched_vc;
    genvar gq;
    generate
        for (gq = 0; gq < NQC; gq = gq + 1) begin : g_fv
            fabric_seq_copy u_fv (.clk(clk), .rst_n(rst_n), .d(fetch), .q(fetched_vc[gq]));
        end
    endgenerate
    fabric_sram #(.W(256), .D(DEPTH), .NRD(1), .NWR(1), .MB(256)) u_prog (
        .clk(clk), .rd_en(fetch), .rd_addr(faddr[PAW-1:0]), .rd_data(fetched),
        .wr_en(1'b0), .wr_addr({PAW{1'b0}}), .wr_data(256'd0), .wr_mask(1'b0));
`ifndef FABRIC_SYNTH
    initial if (PROG_FILE != "") begin
        if (LANE == 0) $readmemh(PROG_FILE, u_prog.mem);
        else           $readmemh($sformatf("lane%0d_%s", LANE, PROG_FILE), u_prog.mem);
    end
`endif

    integer w;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rq_rd <= 0; rq_wr <= 0; rq_n <= 0;
            fpc <= 0; fend <= 0; f_side <= 0; qn <= 0; fetched_v <= 1'b0; fs <= 0; s0 <= 0; s1 <= 0;
        end else begin
            fetched_v <= fetch;
            if (fetch) begin
                fs <= fside;
                if (more) fpc <= fpc + 1'b1;
                else begin                          // the next run: its first step now, the rest after
                    fpc <= rq_pc[rq_rd] + 1'b1; fend <= rq_pc[rq_rd] + rq_steps[rq_rd]; f_side <= rq_side[rq_rd];
                    rq_rd <= rq_rd + 1'b1;
                end
            end
            if (push) begin
                rq_pc[rq_wr] <= push_pc; rq_steps[rq_wr] <= push_steps; rq_side[rq_wr] <= {push_layer, push_page};
                rq_wr <= rq_wr + 1'b1;
            end
            rq_n <= rq_n + {{RW{1'b0}}, push} - {{RW{1'b0}}, fetch && !more};
            // Each slice of the queue reads its own copy of `fetched_v`.
            for (w = 0; w < NQC; w = w + 1) begin
                if (fetched_vc[w] && issue)
                    q0[w*QSL +: QSL] <= (qn == 2'd1) ? fetched[w*QSL +: QSL] : q1[w*QSL +: QSL];
                else if (fetched_vc[w])
                    begin if (qn == 2'd0) q0[w*QSL +: QSL] <= fetched[w*QSL +: QSL];
                          else            q1[w*QSL +: QSL] <= fetched[w*QSL +: QSL]; end
                else if (issue)
                    q0[w*QSL +: QSL] <= q1[w*QSL +: QSL];
                if (fetched_vc[w] && issue && qn == 2'd2) q1[w*QSL +: QSL] <= fetched[w*QSL +: QSL];
            end
            if (fetched_v && issue)  s0 <= (qn == 2'd1) ? fs : s1;
            else if (fetched_v)      begin if (qn == 2'd0) s0 <= fs; else s1 <= fs; end
            else if (issue)          s0 <= s1;
            if (fetched_v && issue && qn == 2'd2) s1 <= fs;
            qn <= qn + {1'b0, fetched_v} - {1'b0, issue};
        end
    end
`ifndef FABRIC_SYNTH
    always @(posedge clk) if (rst_n && qn > 2'd2) $display("FAIL: lane %0d's fetch queue overran", LANE);
`endif
endmodule

// One registered copy of a strobe the sequencer spreads over a wide register.
// Its own hierarchy, so synthesis cannot merge the copies back into one flop:
// `keep` on the register would hold the net and merge the flop anyway.  The
// vector units and the tile have the same thing for the same reason
// (fabric_const_copy, fabric_strobe_copy); this one carries a reset, because
// what it drives must be known from the first cycle.
(* keep_hierarchy *)
module fabric_seq_copy (
    input  wire clk,
    input  wire rst_n,
    input  wire d,
    output reg  q
);
    always @(posedge clk or negedge rst_n) if (!rst_n) q <= 1'b0; else q <= d;
endmodule

`default_nettype wire
