// Self-checking testbench for fabric_conv_silu against fabric.layer.emit_conv_vectors.

`timescale 1ns/1ps
`default_nettype none

module tb_conv_silu #(
    parameter int C = 64,
    parameter int K = 4,
    parameter int L = 8
);
    localparam int HW = (K - 1) * 8;
    localparam int BEATS = C / L;
    reg clk = 0;
    always #5 clk = ~clk;

    reg [HW-1:0]  hmem [0:C-1];
    reg [7:0]     xmem [0:C-1];
    reg [K*8-1:0] wmem [0:C-1];
    reg [15:0]    mim [0:C-1], mom [0:C-1];
    reg [5:0]     sim [0:C-1], som [0:C-1];
    reg [7:0]     ey [0:C-1];
    reg [HW-1:0]  eh [0:C-1];

    reg               in_valid = 0;
    reg [L*8-1:0]     in_x = 0;
    reg [L*HW-1:0]    in_hist = 0;
    reg [L*K*8-1:0]   in_w = 0;
    reg [L*16-1:0]    mult_in = 0, mult_out = 0;
    reg [L*6-1:0]     sh_in = 0, sh_out = 0;
    wire              out_valid;
    wire [L*8-1:0]    out_y;
    wire [L*HW-1:0]   out_hist;
    fabric_conv_silu #(.K(K), .L(L)) dut (
        .clk(clk), .in_valid(in_valid), .in_x(in_x), .in_hist(in_hist), .in_w(in_w), .mult_in(mult_in), .sh_in(sh_in),
        .mult_out(mult_out), .sh_out(sh_out), .out_valid(out_valid), .out_y(out_y), .out_hist(out_hist));

    integer b, k, errors, got;
    always @(posedge clk) begin
        if (out_valid) begin
            for (k = 0; k < L; k = k + 1) begin
                if (out_y[k*8 +: 8] !== ey[got*L + k] || out_hist[k*HW +: HW] !== eh[got*L + k]) begin
                    errors = errors + 1;
                    if (errors <= 5) $display("channel %0d: got %h/%h expected %h/%h", got*L + k, out_y[k*8 +: 8], out_hist[k*HW +: HW], ey[got*L + k], eh[got*L + k]);
                end
            end
            got = got + 1;
        end
    end

    initial begin
        $readmemh("hist.hex", hmem);
        $readmemh("x.hex", xmem);
        $readmemh("w.hex", wmem);
        $readmemh("mult_in.hex", mim);
        $readmemh("sh_in.hex", sim);
        $readmemh("mult_out.hex", mom);
        $readmemh("sh_out.hex", som);
        $readmemh("expected_y.hex", ey);
        $readmemh("expected_hist.hex", eh);
        errors = 0; got = 0;
        repeat (2) @(posedge clk);
        for (b = 0; b < BEATS; b = b + 1) begin
            @(negedge clk);
            in_valid = 1;
            for (k = 0; k < L; k = k + 1) begin
                in_x[k*8 +: 8]        = xmem[b*L + k];
                in_hist[k*HW +: HW]   = hmem[b*L + k];
                in_w[k*K*8 +: K*8]    = wmem[b*L + k];
                mult_in[k*16 +: 16]   = mim[b*L + k];
                sh_in[k*6 +: 6]       = sim[b*L + k];
                mult_out[k*16 +: 16]  = mom[b*L + k];
                sh_out[k*6 +: 6]      = som[b*L + k];
            end
        end
        @(negedge clk);
        in_valid = 0;
        repeat (14) @(posedge clk);          // latency 11 and a margin
        if (got != BEATS) $display("FAIL: %0d output beats, expected %0d", got, BEATS);
        else if (errors == 0) $display("PASS: %0d channels", C);
        else $display("FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule

`default_nettype wire
