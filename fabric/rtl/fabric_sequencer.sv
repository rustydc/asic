// The token sequencer: a microcoded issue engine that runs a layer's
// program (fabric/sequencer.py) over the tiles, the vector units and the
// DMAs, several tokens of different contexts interleaved in one stream.
//
// The program is a list of steps.  Each step is a command to one unit
// (unit id, engine, a length, a source and destination buffer and an
// argument), up to seven buffer ids it consumes, up to two it produces
// (each either a write or a contribution to a whole vector), and a last
// flag.  The controller issues in program order.  Per buffer id it keeps
// the number of outstanding writers and readers; the head step issues when
// every consumed buffer has no outstanding writer, every produced buffer
// no outstanding reader and (for a write, not a contribution) no
// outstanding writer, and the unit reports the addressed engine free.  On
// issue the step's ids are counted and remembered under its tag; a unit
// returns the tag on its engine's done port and the counts are released,
// forwarded into the same cycle's issue check.
//
// Timing, which fabric/sequencer.py reproduces: a step issues at the
// earliest cycle that is after the previous issue, one cycle after the
// last of its dependencies completed, and one cycle after the engine last
// completed.

`timescale 1ns/1ps
`default_nettype none

module fabric_sequencer #(
    parameter int NU        = 10,               // units
    parameter int NE        = 4,                // engines per unit at most
    parameter int DEPTH     = 1024,             // program steps at most
    parameter int NID       = 256,              // buffer ids
    parameter int CW        = 6,                // counter width: outstanding writers or readers of one buffer
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
    output wire [15:0]          cmd_src,
    output wire [15:0]          cmd_dst,
    output wire [31:0]          cmd_arg,
    output wire [7:0]           cmd_tag,
    input  wire [NU-1:0]        cmd_ready,      // the addressed engine of that unit is free
    // completions, one port per engine
    input  wire [NU*NE-1:0]     done_valid,
    input  wire [NU*NE*8-1:0]   done_tag
);
    localparam int NC = 7, NP = 2;
    reg [255:0] prog [0:DEPTH-1];
    initial if (PROG_FILE != "") $readmemh(PROG_FILE, prog);

    reg  [15:0]  pc;
    wire [255:0] cur      = prog[pc];
    wire [3:0]   cur_unit = cur[3:0];
    wire         cur_last = cur[8];
    assign cmd_engine = cur[7:4];
    assign cmd_len    = cur[31:16];
    assign cmd_arg    = cur[63:32];
    assign cmd_src    = cur[79:64];
    assign cmd_dst    = cur[95:80];
    assign cmd_tag    = pc[7:0];
    wire [7:0]   cur_c [0:NC-1];
    wire [7:0]   cur_p [0:NP-1];
    wire [NP-1:0] cur_contrib = cur[169:168];
    genvar gi;
    generate
        for (gi = 0; gi < NC; gi = gi + 1) begin : g_c assign cur_c[gi] = cur[96 + 8*gi +: 8]; end
        for (gi = 0; gi < NP; gi = gi + 1) begin : g_p assign cur_p[gi] = cur[152 + 8*gi +: 8]; end
    endgenerate

    // The ids of every issued step, by tag, for the release at completion.
    reg [NC*8-1:0] tab_c [0:255];
    reg [NP*8-1:0] tab_p [0:255];
    reg [255:0]    tab_live;

    // Outstanding writers and readers per buffer.  The issue check sees the
    // head step's ids with this cycle's releases forwarded; the counters
    // themselves are updated by the releases and then the issue.
    reg [CW-1:0] wr_cnt [0:NID-1];
    reg [CW-1:0] rd_cnt [0:NID-1];
    integer p, k, m, id;

    function automatic [CW-1:0] released_wr(input [7:0] buf_id);      // completions this cycle that produced buf_id
        integer q, j;
        begin
            released_wr = 0;
            for (q = 0; q < NU * NE; q = q + 1)
                if (done_valid[q])
                    for (j = 0; j < NP; j = j + 1)
                        if (tab_p[done_tag[q*8 +: 8]][j*8 +: 8] == buf_id) released_wr = released_wr + 1'b1;
        end
    endfunction
    function automatic [CW-1:0] released_rd(input [7:0] buf_id);      // completions this cycle that consumed buf_id
        integer q, j;
        begin
            released_rd = 0;
            for (q = 0; q < NU * NE; q = q + 1)
                if (done_valid[q])
                    for (j = 0; j < NC; j = j + 1)
                        if (tab_c[done_tag[q*8 +: 8]][j*8 +: 8] == buf_id) released_rd = released_rd + 1'b1;
        end
    endfunction

    reg deps_ok;
    reg [7:0] n_done_now;
    always @* begin
        deps_ok = 1'b1;
        for (k = 0; k < NC; k = k + 1)
            if (cur_c[k] != 8'hFF && wr_cnt[cur_c[k]] != released_wr(cur_c[k])) deps_ok = 1'b0;
        for (k = 0; k < NP; k = k + 1)
            if (cur_p[k] != 8'hFF && (rd_cnt[cur_p[k]] != released_rd(cur_p[k])
                                      || (!cur_contrib[k] && wr_cnt[cur_p[k]] != released_wr(cur_p[k])))) deps_ok = 1'b0;
        n_done_now = 0;
        for (p = 0; p < NU * NE; p = p + 1) if (done_valid[p]) n_done_now = n_done_now + 1'b1;
    end

    reg  [9:0]  outstanding;
    reg         finishing;
    wire        want  = running && !finishing && deps_ok && !tab_live[pc[7:0]];
    assign cmd_valid = want ? (1 << cur_unit) : 0;
    wire        issue = want && cmd_ready[cur_unit];

    reg [7:0] tg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc <= 0; running <= 1'b0; done <= 1'b0; outstanding <= 0; finishing <= 1'b0; tab_live <= 0;
            for (id = 0; id < NID; id = id + 1) begin wr_cnt[id] = 0; rd_cnt[id] = 0; end
        end else begin
            done <= 1'b0;
            if (start && !running) begin
                pc <= 0; running <= 1'b1; outstanding <= 0; finishing <= 1'b0; tab_live <= 0;
                for (id = 0; id < NID; id = id + 1) begin wr_cnt[id] = 0; rd_cnt[id] = 0; end
            end else if (running) begin
                // Releases: each completion returns its buffers.
                for (p = 0; p < NU * NE; p = p + 1) begin
                    if (done_valid[p]) begin
                        tg = done_tag[p*8 +: 8];
                        for (k = 0; k < NC; k = k + 1)
                            if (tab_c[tg][k*8 +: 8] != 8'hFF) rd_cnt[tab_c[tg][k*8 +: 8]] = rd_cnt[tab_c[tg][k*8 +: 8]] - 1'b1;
                        for (k = 0; k < NP; k = k + 1)
                            if (tab_p[tg][k*8 +: 8] != 8'hFF) wr_cnt[tab_p[tg][k*8 +: 8]] = wr_cnt[tab_p[tg][k*8 +: 8]] - 1'b1;
                        tab_live[tg] <= 1'b0;
                    end
                end
                outstanding <= outstanding + {9'd0, issue} - {2'd0, n_done_now};
                if (issue) begin
                    for (k = 0; k < NC; k = k + 1)
                        if (cur_c[k] != 8'hFF) rd_cnt[cur_c[k]] = rd_cnt[cur_c[k]] + 1'b1;
                    for (k = 0; k < NP; k = k + 1)
                        if (cur_p[k] != 8'hFF) wr_cnt[cur_p[k]] = wr_cnt[cur_p[k]] + 1'b1;
                    tab_c[pc[7:0]] <= cur[96 +: NC*8];
                    tab_p[pc[7:0]] <= cur[152 +: NP*8];
                    tab_live[pc[7:0]] <= 1'b1;
                    pc <= pc + 1'b1;
                    if (cur_last || pc + 1 == n_steps) finishing <= 1'b1;
                end
                if (finishing && outstanding == n_done_now) begin
                    running <= 1'b0; done <= 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
