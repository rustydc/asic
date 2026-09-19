// The token sequencer: a microcoded issue engine that runs a layer's
// program (fabric/sequencer.py) over the tiles, the vector units and the
// DMAs.
//
// The program is a list of steps.  Each step is a command to one unit
// (unit id, engine, a length, a source and destination buffer and an
// argument), a dependency mask over the previous WINDOW steps, and two
// flags: barrier (wait for every earlier step) and last.  The controller
// issues in program order: the step at the head issues when every step in
// its mask has completed, the barrier (if set) is satisfied, and the unit
// reports the addressed engine free; then it moves to the next step
// without waiting for this one.  A unit returns the step's tag when it
// completes, on its own engine's done port.  Completed steps are kept in a
// ring of WINDOW bits relative to the head, so a dependency is one compare.
//
// Timing, which fabric/sequencer.py reproduces: a step issues at the
// earliest cycle that is after the previous issue, one cycle after the
// last of its dependencies completed (done is forwarded into the ring in
// the cycle it arrives), and one cycle after the engine last completed.

`timescale 1ns/1ps
`default_nettype none

module fabric_sequencer #(
    parameter int NU        = 10,               // units
    parameter int NE        = 4,                // engines per unit at most
    parameter int DEPTH     = 512,              // program steps at most
    parameter int WINDOW    = 32,
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
    reg [127:0] prog [0:DEPTH-1];
    initial if (PROG_FILE != "") $readmemh(PROG_FILE, prog);

    reg  [15:0]  pc;
    wire [127:0] cur      = prog[pc];
    wire [3:0]   cur_unit = cur[3:0];
    wire         cur_bar  = cur[8];
    wire         cur_last = cur[9];
    wire [31:0]  cur_mask = cur[63:32];
    assign cmd_engine = cur[7:4];
    assign cmd_len    = cur[31:16];
    assign cmd_src    = cur[79:64];
    assign cmd_dst    = cur[95:80];
    assign cmd_arg    = cur[127:96];
    assign cmd_tag    = pc[7:0];

    // Completions this cycle, mapped to ring positions relative to the head.
    reg  [WINDOW-1:0] ring;                     // bit j: step pc-1-j has completed
    reg  [WINDOW-1:0] done_now;
    reg  [7:0]        n_done_now;
    integer p;
    reg  [7:0] rel;
    always @* begin
        done_now = 0; n_done_now = 0;
        for (p = 0; p < NU * NE; p = p + 1) begin
            if (done_valid[p]) begin
                n_done_now = n_done_now + 1'b1;
                rel = pc[7:0] - 8'd1 - done_tag[p*8 +: 8];
                if (rel < WINDOW) done_now[rel] = 1'b1;
            end
        end
    end
    wire [WINDOW-1:0] ring_fwd = ring | done_now;

    reg  [9:0]  outstanding;
    reg         finishing;
    wire        deps_ok    = ((cur_mask[WINDOW-1:0] & ~ring_fwd) == 0);
    wire        barrier_ok = !cur_bar || (outstanding == n_done_now);
    wire        want       = running && !finishing && deps_ok && barrier_ok;
    assign cmd_valid = want ? (1 << cur_unit) : 0;
    wire        issue      = want && cmd_ready[cur_unit];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc <= 0; running <= 1'b0; done <= 1'b0; ring <= 0; outstanding <= 0; finishing <= 1'b0;
        end else begin
            done <= 1'b0;
            if (start && !running) begin
                pc <= 0; running <= 1'b1; ring <= 0; outstanding <= 0; finishing <= 1'b0;
            end else if (running) begin
                // The ring shifts by one on an issue (the new head-1 is the step just issued, not done).
                ring <= issue ? {ring_fwd[WINDOW-2:0], 1'b0} : ring_fwd;
                outstanding <= outstanding + {9'd0, issue} - {2'd0, n_done_now};
                if (issue) begin
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
