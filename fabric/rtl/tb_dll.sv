// Self-checking testbench for fabric_dll: at a given tap length it must lock,
// report the period in taps within one tap, and a slave line driven by its
// quarter must delay an edge by a quarter period within a tap and a half.
// With a line too short for the period (EXPECT_RANGE_ERR) it must say so.

`timescale 1ns/1ps
`default_nettype none

module tb_dll #(
    parameter int  TAPS   = 256,
    parameter real TAP_PS = 60.0,
    parameter real T      = 4.0,
    parameter int  EXPECT_RANGE_ERR = 0
);
    localparam int CW = $clog2(TAPS);
    reg clk = 0, rst_n = 0;
    always #(T / 2) clk = ~clk;
    wire          locked, range_err;
    wire [CW-1:0] period_code, quarter;
    fabric_dll #(.TAPS(TAPS), .TAP_PS(TAP_PS)) dut (.clk(clk), .rst_n(rst_n), .update_ok(1'b1), .locked(locked), .range_err(range_err),
                                                    .period_code(period_code), .quarter(quarter));
    // A slave line on the clock, to measure the delay it produces.
    wire slave;
    fabric_delay_line #(.TAPS(TAPS), .TAP_PS(TAP_PS)) line (.in(clk), .code(quarter), .out(slave));
    real t_in, t_out, measured;
    integer guard, errors;
    initial begin
        errors = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        guard = 0;
        while (!locked && !range_err && guard < 20000) begin @(posedge clk); guard = guard + 1; end
        if (EXPECT_RANGE_ERR) begin
            if (range_err && !locked) $display("PASS: tap %0.0f ps, %0d taps cannot span %0.1f ns, reported after %0d clocks", TAP_PS, TAPS, T, guard);
            else $display("FAIL: no range error (locked %0d)", locked);
            $finish;
        end
        if (range_err) begin $display("FAIL: range error at tap %0.0f ps", TAP_PS); $finish; end
        if (!locked) begin $display("FAIL: no lock"); $finish; end
        repeat (40) @(posedge clk);
        // The period in taps, within a tap of the truth.
        if (period_code * TAP_PS / 1000.0 < T - TAP_PS / 1000.0 || period_code * TAP_PS / 1000.0 > T + TAP_PS / 1000.0) begin
            errors = errors + 1;
            $display("period code %0d = %0.3f ns for a %0.1f ns clock", period_code, period_code * TAP_PS / 1000.0, T);
        end
        // The slave delay, measured edge to edge.
        @(posedge clk); t_in = $realtime;
        @(posedge slave); t_out = $realtime;
        measured = t_out - t_in;
        if (measured < T / 4 - 1.5 * TAP_PS / 1000.0 || measured > T / 4 + 1.5 * TAP_PS / 1000.0) begin
            errors = errors + 1;
            $display("quarter %0d taps = %0.3f ns, wanted %0.3f", quarter, measured, T / 4);
        end
        if (errors == 0) $display("PASS: tap %0.0f ps, period %0d taps, quarter %0d taps = %0.3f ns, locked after %0d clocks", TAP_PS, period_code, quarter, measured, guard);
        else $display("FAIL: %0d errors", errors);
        $finish;
    end
endmodule

`default_nettype wire
