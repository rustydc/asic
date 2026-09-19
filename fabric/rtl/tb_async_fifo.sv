// Self-checking testbench for fabric_async_fifo: unrelated clocks, random
// pushes and pops with bursts of each, a scoreboard of the sequence, and a
// check that full and empty are never optimistic.

`timescale 1ns/1ps
`default_nettype none

module tb_async_fifo #(
    parameter int  AW     = 3,
    parameter int  N      = 4000,
    parameter real WCLK   = 1.3,
    parameter real RCLK   = 4.0,
    parameter int  SEED   = 1
);
    localparam int W = 32;
    reg wclk = 0, rclk = 0, rst_n = 0;
    always #(WCLK / 2) wclk = ~wclk;
    always #(RCLK / 2) rclk = ~rclk;

    reg          wr_en = 0, rd_en = 0;
    reg  [W-1:0] wdata = 0;
    wire         wfull, rempty;
    wire [W-1:0] rdata;
    fabric_async_fifo #(.W(W), .AW(AW)) dut (
        .wclk(wclk), .wrst_n(rst_n), .wr_en(wr_en), .wdata(wdata), .wfull(wfull),
        .rclk(rclk), .rrst_n(rst_n), .rd_en(rd_en), .rdata(rdata), .rempty(rempty));

    integer pushed = 0, popped = 0, errors = 0, seed = SEED, occupancy = 0, max_occ = 0;
    integer burst_w = 0, burst_r = 0;
    // The writer presents the next sequence number whenever it wants to push;
    // a word not taken (full) stays presented or is withdrawn, either is legal.
    always @(posedge wclk) begin
        if (rst_n) begin
            if (wr_en && !wfull) begin
                pushed = pushed + 1; occupancy = occupancy + 1;
                if (occupancy > max_occ) max_occ = occupancy;
                if (occupancy > (1 << AW)) begin errors = errors + 1; $display("overflow at %t", $time); end
            end
            if (burst_w == 0) burst_w = $urandom(seed) % 12;
            if (pushed < N && burst_w > 0 && ($urandom(seed) % 4 != 0)) begin
                wr_en <= 1; wdata <= pushed;
                burst_w = burst_w - 1;
            end else wr_en <= 0;
        end
    end
    // The reader: pops at its own pace.
    reg [W-1:0] expect_v = 0;
    always @(posedge rclk) begin
        if (rst_n) begin
            if (rd_en && !rempty) begin
                if (rdata !== expect_v) begin errors = errors + 1; if (errors <= 5) $display("pop %0d: got %0d expected %0d", popped, rdata, expect_v); end
                expect_v <= expect_v + 1;
                popped = popped + 1; occupancy = occupancy - 1;
                if (occupancy < 0) begin errors = errors + 1; $display("underflow at %t", $time); end
            end
            if (burst_r == 0) burst_r = $urandom(seed) % 20;
            rd_en <= (burst_r > 0) && ($urandom(seed) % 3 != 0);
            if (burst_r > 0) burst_r = burst_r - 1;
        end
    end

    initial begin
        repeat (4) @(posedge wclk);
        rst_n = 1;
        wait (popped >= N);
        repeat (10) @(posedge rclk);
        if (errors == 0) $display("PASS: %0d words through a %0d-deep fifo, peak occupancy %0d", popped, 1 << AW, max_occ);
        else $display("FAIL: %0d errors", errors);
        $finish;
    end
    initial begin
        #(N * 40 * RCLK);
        $display("FAIL: timeout with %0d pushed, %0d popped", pushed, popped);
        $finish;
    end
endmodule

`default_nettype wire
