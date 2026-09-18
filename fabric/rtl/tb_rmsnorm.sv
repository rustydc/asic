// Self-checking testbench for fabric_rmsnorm against fabric.layer.emit_rmsnorm_vectors.
// The vector is sent twice to show the unit re-arms.

`timescale 1ns/1ps
`default_nettype none

module tb_rmsnorm #(
    parameter int D     = 128,
    parameter int XW    = 16,
    parameter int OW    = 8,
    parameter int L     = 4,
    parameter int SW    = 38,
    parameter int MULT  = 1,
    parameter int SHIFT = 7,
    parameter int EPS   = 0
);
    localparam int BEATS = D / L;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg [XW-1:0] xmem [0:D-1];
    reg [15:0]   gmem [0:D-1];
    reg [OW-1:0] emem [0:D-1];

    reg             in_valid = 0;
    reg [L*XW-1:0]  in_x = 0;
    reg [L*16-1:0]  in_gain = 0;
    wire            out_valid;
    wire [L*OW-1:0] out_y;
    fabric_rmsnorm #(.D(D), .XW(XW), .OW(OW), .L(L), .SW(SW)) dut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_x(in_x), .in_gain(in_gain),
        .mult(MULT[15:0]), .shift(SHIFT[5:0]), .eps(EPS), .out_valid(out_valid), .out_y(out_y));

    integer i, b, k, errors, got;
    always @(posedge clk) begin
        if (out_valid) begin
            for (k = 0; k < L; k = k + 1) begin
                if (out_y[k*OW +: OW] !== emem[(got % BEATS) * L + k]) begin
                    errors = errors + 1;
                    if (errors <= 5) $display("y[%0d]: got %h expected %h", (got % BEATS) * L + k, out_y[k*OW +: OW], emem[(got % BEATS) * L + k]);
                end
            end
            got = got + 1;
        end
    end

    task send_vector;
        begin
            for (b = 0; b < BEATS; b = b + 1) begin
                @(negedge clk);
                in_valid = 1;
                for (k = 0; k < L; k = k + 1) begin
                    in_x[k*XW +: XW]  = xmem[b*L + k];
                    in_gain[k*16 +: 16] = gmem[b*L + k];
                end
                if (b % 5 == 2) begin @(negedge clk); in_valid = 0; end   // a bubble now and then
            end
            @(negedge clk);
            in_valid = 0;
        end
    endtask

    initial begin
        $readmemh("x.hex", xmem);
        $readmemh("gain.hex", gmem);
        $readmemh("expected_y.hex", emem);
        errors = 0; got = 0;
        repeat (2) @(posedge clk);
        rst_n = 1;
        send_vector;
        repeat (BEATS + 20) @(posedge clk);
        send_vector;
        repeat (BEATS + 20) @(posedge clk);
        if (got != 2 * BEATS) $display("FAIL: %0d output beats, expected %0d", got, 2 * BEATS);
        else if (errors == 0) $display("PASS: %0d elements twice", D);
        else $display("FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule

`default_nettype wire
