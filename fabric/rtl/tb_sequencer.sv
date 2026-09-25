// Self-checking testbench for fabric_sequencer: the controller runs a
// program written by fabric.sequencer.emit_program over stub units, each
// with a number of engines that are busy for the commanded length.  The
// stubs write a trace (tag, unit, engine, issue cycle, done cycle) that
// the Python test checks against the program's dependencies, and the
// cycle count must equal the Python schedule's.

`timescale 1ns/1ps
`default_nettype none

module fabric_unit_stub #(
    parameter int UID = 0,
    parameter int E   = 1,
    parameter int NE  = 4
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          cmd_valid,
    input  wire [3:0]    cmd_engine,
    input  wire [15:0]   cmd_len,
    input  wire [31:0]   cmd_arg,
    input  wire [7:0]    cmd_tag,
    output wire          cmd_ready,
    output reg  [NE-1:0] done_valid,
    output reg  [NE*8-1:0] done_tag
);
    reg [31:0] remaining [0:NE-1];
    reg [7:0]  tag_r     [0:NE-1];
    reg [NE-1:0] busy;
    assign cmd_ready = (cmd_engine < E) && !busy[cmd_engine];
    integer e;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 0; done_valid <= 0; done_tag <= 0;
            for (e = 0; e < NE; e = e + 1) begin remaining[e] <= 0; tag_r[e] <= 0; end
        end else begin
            done_valid <= 0;
            for (e = 0; e < NE; e = e + 1) begin
                if (busy[e]) begin
                    if (remaining[e] <= 1) begin
                        busy[e] <= 1'b0; done_valid[e] <= 1'b1; done_tag[e*8 +: 8] <= tag_r[e];
                        // done_valid is assigned here and seen the cycle after, so
                        // that is the cycle the completion arrives.
                        $fdisplay(tb_sequencer.trace, "%0d %0d %0d %0d %0d", tag_r[e], UID, e, tb_sequencer.issued_at[tag_r[e]], tb_sequencer.cycle + 1);
                        tb_sequencer.last_done = tb_sequencer.cycle + 1;
                    end else remaining[e] <= remaining[e] - 1'b1;
                end
            end
            if (cmd_valid && cmd_ready) begin
                // arg is the command's whole span, issue to completion, as a real
                // adapter's is; the first of those cycles is this one.  It is not
                // the length, which is sixteen bits and holds a real command's beats.
                busy[cmd_engine] <= 1'b1; remaining[cmd_engine] <= (cmd_arg > 1) ? cmd_arg - 1'b1 : 32'd1;
                tag_r[cmd_engine] <= cmd_tag;
            end
        end
    end
endmodule

module tb_sequencer #(
    parameter int N = 4,
    parameter int PC0 = 0,                             // the program's first step in the store
    parameter int EXPECTED_CYCLES = 0,
    parameter int E0 = 1, E1 = 2, E2 = 1, E3 = 1, E4 = 4, E5 = 1, E6 = 1, E7 = 2, E8 = 4, E9 = 1
);
    localparam int NU = 10, NE = 4;
    reg clk = 0, rst_n = 0;
    always #0.625 clk = ~clk;
    integer cycle = 0, trace;
    always @(posedge clk) cycle <= cycle + 1;
    integer issued_at [0:255];                            // by tag (the step index modulo 256)

    reg                start = 0;
    wire               running, done;
    wire [NU-1:0]      cmd_valid, cmd_ready;
    wire [3:0]         cmd_engine;
    wire [15:0]        cmd_len;
    wire [29:0]        cmd_src, cmd_dst, cmd_a2, cmd_a3;
    wire [31:0]        cmd_arg;
    wire [7:0]         cmd_tag;
    wire [NU*NE-1:0]   done_valid;
    wire [NU*NE*8-1:0] done_tag;
    fabric_sequencer #(.NU(NU), .NE(NE), .PROG_FILE("program.hex")) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .pc_start(PC0[15:0]), .n_steps(N[15:0]), .running(running), .done(done),
        .cmd_valid(cmd_valid), .cmd_engine(cmd_engine), .cmd_len(cmd_len), .cmd_src(cmd_src), .cmd_dst(cmd_dst),
        .cmd_a2(cmd_a2), .cmd_a3(cmd_a3), .cmd_arg(cmd_arg), .cmd_tag(cmd_tag), .cmd_ready(cmd_ready), .done_valid(done_valid), .done_tag(done_tag));

    function automatic integer engines(input integer unit);
        case (unit)
            0: engines = E0; 1: engines = E1; 2: engines = E2; 3: engines = E3; 4: engines = E4;
            5: engines = E5; 6: engines = E6; 7: engines = E7; 8: engines = E8; default: engines = E9;
        endcase
    endfunction
    genvar u;
    generate
        for (u = 0; u < NU; u = u + 1) begin : g_unit
            fabric_unit_stub #(.UID(u), .E(engines(u)), .NE(NE)) stub (
                .clk(clk), .rst_n(rst_n), .cmd_valid(cmd_valid[u]), .cmd_engine(cmd_engine), .cmd_len(cmd_len), .cmd_arg(cmd_arg),
                .cmd_tag(cmd_tag), .cmd_ready(cmd_ready[u]), .done_valid(done_valid[u*NE +: NE]), .done_tag(done_tag[u*NE*8 +: NE*8]));
        end
    endgenerate

    // Record issues by tag, and the cycle of the last completion.
    integer issues = 0, t0, guard, last_done = 0;
    always @(posedge clk) if (|(cmd_valid & cmd_ready)) begin
        if (issues == 0) t0 = cycle;                       // cycle 0 of the schedule is the first issue
        issued_at[cmd_tag] = cycle; issues = issues + 1;
    end

    initial begin
        trace = $fopen("trace.txt", "w");
        repeat (2) @(posedge clk);
        rst_n = 1;
        @(posedge clk); #0.1;
        start = 1; @(posedge clk); #0.1; start = 0;
        guard = 0;
        while (!done && guard < 4000000) begin @(posedge clk); guard = guard + 1; end
        $fclose(trace);
        if (!done) $display("FAIL: never finished, %0d of %0d issued", issues, N);
        else if (issues != N) $display("FAIL: %0d issued of %0d", issues, N);
        else if (last_done - t0 != EXPECTED_CYCLES) $display("FAIL: last completion at %0d cycles, expected %0d", last_done - t0, EXPECTED_CYCLES);
        else $display("PASS: %0d steps, last completion at %0d cycles, done %0d cycles later", N, last_done - t0, cycle - last_done);
        $finish;
    end
endmodule

`default_nettype wire
