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
    localparam int PAW = (DEPTH > 1) ? $clog2(DEPTH) : 1;

    // The program is fetched ahead of the step being checked.  Read where it
    // is used, it is a DEPTH-entry mux of 256-bit words in the same cycle as
    // the issue check -- which indexes a counter per buffer id the step names
    // -- and the two together were 37 levels of logic and the whole of this
    // module's path.  They are a memory now, which wants its address a cycle
    // early anyway, so `fpc` runs ahead on its own and the head step and the
    // one after it wait in `q0` and `q1`.  Two deep because a step may issue
    // every cycle and a one-deep queue would give up every other one.
    reg  [15:0]  pc;                                    // the step being checked
    reg  [PAW-1:0] fpc;                                 // the step being fetched
    reg  [255:0] q0, q1;
    reg  [1:0]   qn;                                    // steps in hand
    reg          fetched_v;                             // the memory answers this cycle
    wire [255:0] fetched;
    wire         issue;
    // Room for what a fetch started now would bring: the memory answers the
    // cycle after its address, so one step may already be on its way.
    wire [2:0]   after = {1'b0, qn} + {2'b0, fetched_v} - {2'b0, issue};
    wire         fetch = running && (after < 3'd2) && ({16'd0, fpc} != n_steps);
    // `fetched_v` selects what each of the queue's 512 bits takes, so one flop
    // held 490 loads and 798 fF: two nanoseconds of clock-to-output, and the
    // whole of this module's path once the counting above came off it.  The
    // mapper buffers what it drives and cannot help the flop itself, so the
    // flop is replicated, a copy per slice of the queue -- the same trick as
    // fabric_const_copy for a lane's shift amount and fabric_strobe_copy for
    // the tile's strobe.  Every copy takes `fetch`, so all of them and
    // `fetched_v` (which keeps the few loads that are left) always agree.
    localparam int NQC = 16;                            // copies, one per queue slice
    localparam int QSL = 256 / NQC;
    wire [NQC-1:0] fetched_vc;
    genvar gq;
    generate
        for (gq = 0; gq < NQC; gq = gq + 1) begin : g_fv
            fabric_seq_copy u_fv (.clk(clk), .rst_n(rst_n), .d(fetch), .q(fetched_vc[gq]));
        end
    endgenerate
    fabric_sram #(.W(256), .D(DEPTH), .NRD(1), .NWR(1), .MB(256)) u_prog (
        .clk(clk), .rd_en(fetch), .rd_addr(fpc), .rd_data(fetched),
        .wr_en(1'b0), .wr_addr({PAW{1'b0}}), .wr_data(256'd0), .wr_mask(1'b0));
`ifndef FABRIC_SYNTH
    initial if (PROG_FILE != "") $readmemh(PROG_FILE, u_prog.mem);
`endif

    wire [255:0] cur      = q0;
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
    integer q, r;
    always @* begin
        rel_now = 0;
        rel_en  = 0;
        rel_c   = {(NREL*NC){8'hFF}};
        rel_p   = {(NREL*NC){8'hFF}};
        rel_tag = 0;
        remaining = want_rel;
        for (r = 0; r < NREL; r = r + 1) begin
            lower = {remaining[NPORT-2:0], 1'b0};              // lower[i] = remaining[i-1]
            for (sh = 1; sh < NPORT; sh = sh * 2) lower = lower | (lower << sh);
            one = remaining & ~lower;
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
    reg [NPORT-1:0]     d_now;
    // A port's record is one command, so it can be given no second one while
    // the first is outstanding: a unit reports done a cycle after it drops
    // its ready, and without this the completion returned the newer
    // command's ids for the older one -- buffers the newer command was still
    // reading.  A port whose completion drains this cycle is free, because
    // the drain reads the register and the issue writes it.
    // A port holds its buffers until the drain has actually moved the
    // counters, which is now the cycle after the pick, so `busy` is what it
    // says and nothing is forgiven early: freeing the port in the pick cycle
    // would let an issue overwrite the slot the drain has yet to apply.
    reg  [NPORT-1:0] busy;
    wire [NPORT-1:0] blocked = busy;

    // Outstanding writers and readers per buffer.  The issue check sees the
    // head step's ids with this cycle's drains forwarded; the counters
    // themselves are updated by the drains and then the issue.
    reg [CW-1:0] wr_cnt [0:NID-1];
    reg [CW-1:0] rd_cnt [0:NID-1];
    integer p, k, m, id, x, w;
    reg signed [CW:0] dr, dw;                            // a counter's move this cycle

    // The drained ids travel as arguments, not as a reference to rel_c and
    // rel_p: what a function reads is not in an always @* block's sensitivity,
    // only what it is passed, and a stale deps_ok deadlocks.
    // With the drain registered the counters already hold every release that
    // has landed, so the check is a read.  It used to forward this cycle's
    // drains into itself through `released`, which counted matches as one
    // `+ 1` per id in play -- six of them, and yosys leaves that a chain of
    // six carry-propagate adds -- at eight call sites, all of them behind a
    // 256-to-1 mux of the counter array.
    reg deps_ok;
    always @* begin
        deps_ok = 1'b1;
        for (k = 0; k < NC; k = k + 1)
            if (cur_c[k] != 8'hFF && wr_cnt[cur_c[k]] != 0) deps_ok = 1'b0;
        for (k = 0; k < NP; k = k + 1)
            if (cur_p[k] != 8'hFF && (rd_cnt[cur_p[k]] != 0
                                      || (!cur_contrib[k] && wr_cnt[cur_p[k]] != 0))) deps_ok = 1'b0;
    end

    // Commands issued and not yet returned.  Counted against the drains,
    // which are at most NREL a cycle, and not against the completions, which
    // are up to NPORT: a population count of the forty done ports is forty
    // chained increments, and synthesis leaves it a chain -- 38 levels and
    // 1.75 of this module's 2.44 nanoseconds, for a number read nowhere but
    // the test below.  A command that has completed and not drained is still
    // outstanding, so reaching zero is what it always was: every step issued
    // has returned its buffers, which is also what `held == 0` said.
    localparam int RW = (NREL > 1) ? $clog2(NREL + 1) : 1;
    reg  [RW-1:0] n_rel;
    integer z;
    always @* begin
        n_rel = 0;
        for (z = 0; z < NREL; z = z + 1) if (d_en[z]) n_rel = n_rel + 1'b1;
    end
    reg  [9:0]  outstanding;
    wire [9:0]  out_next = outstanding + {9'd0, issue} - {{(10-RW){1'b0}}, n_rel};
    reg         finishing;
    wire        want  = running && !finishing && qn != 2'd0 && deps_ok && !tab_live[pc[7:0]]
                        && !blocked[cur_port];
    assign cmd_valid = want ? (1 << cur_unit) : 0;
    assign      issue = want && cmd_ready[cur_unit];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc <= 0; running <= 1'b0; done <= 1'b0; outstanding <= 0; finishing <= 1'b0; tab_live <= 0; pend <= 0; busy <= 0;
                d_en <= 0; d_now <= 0;
            fpc <= 0; qn <= 0; fetched_v <= 1'b0;
            for (id = 0; id < NID; id = id + 1) begin wr_cnt[id] = 0; rd_cnt[id] = 0; end
        end else begin
            done <= 1'b0;
            if (start && !running) begin
                pc <= 0; running <= 1'b1; outstanding <= 0; finishing <= 1'b0; tab_live <= 0; pend <= 0; busy <= 0;
                d_en <= 0; d_now <= 0;
                fpc <= 0; qn <= 0; fetched_v <= 1'b0;
                for (id = 0; id < NID; id = id + 1) begin wr_cnt[id] = 0; rd_cnt[id] = 0; end
            end else begin
                // Not gated on `running`: when it is low nothing issues and no
                // unit is busy, so none of this moves anyway, and the gate cost
                // more than everything else here -- one flop's enable reaching
                // every counter and every live bit was 835 loads, and the three
                // gates behind it were 42 of the module's 43 nanoseconds.
                // The fetch runs ahead of the check: the memory answers the cycle
                // after its address, so `fetched_v` says a step arrives now and
                // the queue takes it while the head one issues.
                fetched_v <= fetch;
                if (fetch) fpc <= fpc + 1'b1;
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
                qn <= qn + {1'b0, fetched_v} - {1'b0, issue};
                if (issue) pc <= pc + 1'b1;
                // The drains: NREL completions return their buffers; the rest wait.
                pend <= held;
                // This cycle's pick, for the next one to apply.
                d_en <= rel_en; d_c <= rel_c; d_p <= rel_p; d_tag <= rel_tag; d_now <= rel_now;
                for (p = 0; p < NPORT; p = p + 1)
                    if (d_now[p]) busy[p] <= 1'b0;
                for (x = 0; x < NREL; x = x + 1)
                    if (d_en[x]) tab_live[d_tag[x*8 +: 8]] <= 1'b0;
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
                        if (d_en[x]) begin
                            for (k = 0; k < NC; k = k + 1)
                                if (d_c[x*NC*8 + k*8 +: 8] == id[7:0]) dr = dr - 1;
                            for (k = 0; k < NP; k = k + 1)
                                if (d_p[x*NC*8 + k*8 +: 8] == id[7:0]) dw = dw - 1;
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
                outstanding <= out_next;
                if (issue) begin
                    busy[cur_port] <= 1'b1;                  // after the drains: a port reissued in its drain cycle stays busy
                    slot_c[cur_port] <= cur[IDB +: NC*8];
                    slot_p[cur_port] <= cur[IDB + 8*NC +: NP*8];
                    slot_tag[cur_port] <= pc[7:0];
                    tab_live[pc[7:0]] <= 1'b1;
                    if (cur_last || pc + 1 == n_steps) finishing <= 1'b1;
                end
                if (running && finishing && out_next == 10'd0) begin
                    running <= 1'b0; done <= 1'b1;
                end
            end
        end
    end

`ifndef FABRIC_SYNTH
    always @(posedge clk) if (rst_n && qn > 2'd2) $display("FAIL: the fetch queue overran");
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
