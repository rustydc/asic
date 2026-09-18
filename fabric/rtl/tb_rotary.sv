// Self-checking testbench for fabric_rotary_table and fabric_rotary against
// fabric.layer.emit_rotary_vectors.

`timescale 1ns/1ps
`default_nettype none

module tb_rotary #(
    parameter int HD    = 64,
    parameter int R     = 16,
    parameter int L     = 8,
    parameter int POS   = 12345,
    parameter int MULT  = 1,
    parameter int SHIFT = 22
);
    localparam int H = R / 2;
    localparam int BEATS = HD / L;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg [31:0] ifm [0:H-1];
    reg [15:0] xm [0:HD-1], es [0:H-1], ec [0:H-1];
    reg [7:0]  ey [0:HD-1];

    reg              start = 0;
    reg [H*32-1:0]   inv_freq = 0;
    wire             tdone;
    wire [H*16-1:0]  sin_tab, cos_tab;
    fabric_rotary_table #(.R(R)) u_tab (.clk(clk), .rst_n(rst_n), .start(start), .pos(POS[31:0]), .inv_freq(inv_freq),
                                        .done(tdone), .sin_tab(sin_tab), .cos_tab(cos_tab));
    reg             in_valid = 0;
    reg [L*16-1:0]  in_x = 0;
    wire            out_valid;
    wire [L*8-1:0]  out_y;
    fabric_rotary #(.HD(HD), .R(R), .L(L)) u_rot (.clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_x(in_x), .sin_tab(sin_tab),
                                                  .cos_tab(cos_tab), .mult(MULT[15:0]), .shift(SHIFT[5:0]), .out_valid(out_valid), .out_y(out_y));

    integer i, b, k, errors, got, guard;
    reg seen_done = 0;
    always @(posedge clk) begin
        if (tdone) seen_done <= 1;
        if (out_valid) begin
            for (k = 0; k < L; k = k + 1)
                if (out_y[k*8 +: 8] !== ey[got*L + k]) begin
                    errors = errors + 1;
                    if (errors <= 5) $display("y[%0d]: got %h expected %h", got*L + k, out_y[k*8 +: 8], ey[got*L + k]);
                end
            got = got + 1;
        end
    end

    initial begin
        $readmemh("inv_freq.hex", ifm);
        $readmemh("x.hex", xm);
        $readmemh("expected_sin.hex", es);
        $readmemh("expected_cos.hex", ec);
        $readmemh("expected_y.hex", ey);
        for (i = 0; i < H; i = i + 1) inv_freq[i*32 +: 32] = ifm[i];
        errors = 0; got = 0;
        repeat (2) @(posedge clk);
        rst_n = 1;
        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;
        guard = 0;
        while (!seen_done && guard < H + 10) begin @(posedge clk); #1; guard = guard + 1; end
        if (!seen_done) begin $display("FAIL: table never done"); $finish; end
        for (i = 0; i < H; i = i + 1)
            if (sin_tab[i*16 +: 16] !== es[i] || cos_tab[i*16 +: 16] !== ec[i]) begin
                errors = errors + 1;
                if (errors <= 5) $display("table %0d: got %h/%h expected %h/%h", i, sin_tab[i*16 +: 16], cos_tab[i*16 +: 16], es[i], ec[i]);
            end
        for (b = 0; b < BEATS; b = b + 1) begin
            @(negedge clk);
            in_valid = 1;
            for (k = 0; k < L; k = k + 1) in_x[k*16 +: 16] = xm[b*L + k];
        end
        @(negedge clk);
        in_valid = 0;
        repeat (BEATS + 8) @(posedge clk);
        if (got != BEATS) $display("FAIL: %0d output beats of %0d", got, BEATS);
        else if (errors == 0) $display("PASS: %0d elements", HD);
        else $display("FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule

`default_nettype wire
