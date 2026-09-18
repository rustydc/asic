// Self-checking testbench for fabric_swiglu feeding fabric_residual, against
// fabric.layer.emit_ffn_vectors.

`timescale 1ns/1ps
`default_nettype none

module tb_ffn #(
    parameter int N         = 256,
    parameter int L         = 8,
    parameter int MULT_G    = 1,
    parameter int SH_G      = 12,
    parameter int MULT_O    = 1,
    parameter int SH_O      = 20,
    parameter int RES_MULT  = 1,
    parameter int RES_SHIFT = 9
);
    localparam int BEATS = N / L;
    reg clk = 0;
    always #5 clk = ~clk;

    reg [7:0]  gm [0:N-1], um [0:N-1], ea [0:N-1];
    reg [15:0] hm [0:N-1], eh [0:N-1];

    reg            in_valid = 0;
    reg [L*8-1:0]  in_g = 0, in_u = 0;
    wire           act_valid, res_valid;
    wire [L*8-1:0] act;
    wire [L*16-1:0] h_out;
    reg  [L*16-1:0] h_in;
    fabric_swiglu #(.L(L)) u_act (.clk(clk), .in_valid(in_valid), .in_g(in_g), .in_u(in_u), .mult_g(MULT_G[15:0]), .sh_g(SH_G[5:0]),
                                  .mult_o(MULT_O[15:0]), .sh_o(SH_O[5:0]), .out_valid(act_valid), .out_y(act));
    // The residual reads its h lane from the beat counter of the act stream.
    integer got_act = 0;
    integer k;
    always @* for (k = 0; k < L; k = k + 1) h_in[k*16 +: 16] = hm[got_act*L + k];
    fabric_residual #(.L(L)) u_res (.clk(clk), .in_valid(act_valid), .in_h(h_in), .in_y(act), .mult(RES_MULT[15:0]),
                                    .shift(RES_SHIFT[5:0]), .out_valid(res_valid), .out_h(h_out));

    integer b, errors, got_h;
    always @(posedge clk) begin
        if (act_valid) begin
            for (k = 0; k < L; k = k + 1)
                if (act[k*8 +: 8] !== ea[got_act*L + k]) begin
                    errors = errors + 1;
                    if (errors <= 5) $display("act[%0d]: got %h expected %h", got_act*L + k, act[k*8 +: 8], ea[got_act*L + k]);
                end
            got_act = got_act + 1;
        end
        if (res_valid) begin
            for (k = 0; k < L; k = k + 1)
                if (h_out[k*16 +: 16] !== eh[got_h*L + k]) begin
                    errors = errors + 1;
                    if (errors <= 5) $display("h[%0d]: got %h expected %h", got_h*L + k, h_out[k*16 +: 16], eh[got_h*L + k]);
                end
            got_h = got_h + 1;
        end
    end

    initial begin
        $readmemh("g.hex", gm);
        $readmemh("u.hex", um);
        $readmemh("h.hex", hm);
        $readmemh("expected_act.hex", ea);
        $readmemh("expected_h.hex", eh);
        errors = 0; got_h = 0;
        repeat (2) @(posedge clk);
        for (b = 0; b < BEATS; b = b + 1) begin
            @(negedge clk);
            in_valid = 1;
            for (k = 0; k < L; k = k + 1) begin
                in_g[k*8 +: 8] = gm[b*L + k];
                in_u[k*8 +: 8] = um[b*L + k];
            end
        end
        @(negedge clk);
        in_valid = 0;
        repeat (12) @(posedge clk);
        if (got_act != BEATS || got_h != BEATS) $display("FAIL: %0d act and %0d h beats of %0d", got_act, got_h, BEATS);
        else if (errors == 0) $display("PASS: %0d elements", N);
        else $display("FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule

`default_nettype wire
