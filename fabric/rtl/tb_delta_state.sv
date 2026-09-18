// Self-checking testbench for fabric_delta_state against fabric.layer.emit_delta_vectors.

`timescale 1ns/1ps
`default_nettype none

module tb_delta_state #(
    parameter int K     = 16,
    parameter int V     = 16,
    parameter int DECAY = 60000,
    parameter int BETA  = 30000,
    parameter int YSH   = 9
);
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg [V*16-1:0] smem [0:K-1], esmem [0:K-1];
    reg [7:0]      km [0:K-1], qm [0:K-1], vm [0:V-1];
    reg [15:0]     ey [0:V-1];

    reg            start = 0;
    reg [K*8-1:0]  q = 0, k = 0;
    reg [V*8-1:0]  v = 0;
    reg            row_in_valid = 0;
    reg [V*16-1:0] row_in = 0;
    wire           row_out_valid, y_valid;
    wire [V*16-1:0] row_out, y;
    fabric_delta_state #(.K(K), .V(V), .YSH(YSH)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .q(q), .k(k), .v(v), .decay(DECAY[15:0]), .beta(BETA[15:0]),
        .row_in_valid(row_in_valid), .row_in(row_in), .row_out_valid(row_out_valid), .row_out(row_out),
        .y_valid(y_valid), .y(y));

    integer i, j, errors, rows, seen_y;
    always @(posedge clk) begin
        if (row_out_valid) begin
            if (row_out !== esmem[rows]) begin
                errors = errors + 1;
                if (errors <= 5) $display("row %0d: got %h expected %h", rows, row_out, esmem[rows]);
            end
            rows = rows + 1;
        end
        if (y_valid) begin
            seen_y = seen_y + 1;
            for (j = 0; j < V; j = j + 1)
                if (y[j*16 +: 16] !== ey[j]) begin
                    errors = errors + 1;
                    if (errors <= 5) $display("y[%0d]: got %h expected %h", j, y[j*16 +: 16], ey[j]);
                end
        end
    end

    initial begin
        $readmemh("s.hex", smem);
        $readmemh("expected_s.hex", esmem);
        $readmemh("k.hex", km);
        $readmemh("q.hex", qm);
        $readmemh("v.hex", vm);
        $readmemh("expected_y.hex", ey);
        for (i = 0; i < K; i = i + 1) begin q[i*8 +: 8] = qm[i]; k[i*8 +: 8] = km[i]; end
        for (i = 0; i < V; i = i + 1) v[i*8 +: 8] = vm[i];
        errors = 0; rows = 0; seen_y = 0;
        repeat (2) @(posedge clk);
        rst_n = 1;
        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;
        for (i = 0; i < K; i = i + 1) begin
            row_in_valid = 1; row_in = smem[i];
            @(negedge clk);
            if (i % 4 == 1) begin row_in_valid = 0; @(negedge clk); end
        end
        row_in_valid = 0;
        repeat (K + 12) @(posedge clk);
        if (rows != K || seen_y != 1) $display("FAIL: %0d rows and %0d y outputs", rows, seen_y);
        else if (errors == 0) $display("PASS: %0dx%0d state", K, V);
        else $display("FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule

`default_nettype wire
