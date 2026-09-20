// SwiGLU and the residual add, L elements per beat.
// Golden model: fabric/layer.py (swiglu_int, residual_int).

`default_nettype none
`include "fabric_fx.svh"

// ---------------------------------------------------------------------------
// act = sat8((silu(g * mult_g >> sh_g) * u) * mult_o >> sh_o).  Latency 9,
// one multiply or one round to a stage.
// ---------------------------------------------------------------------------
module fabric_swiglu #(
    parameter int L = 8,
    parameter     LUT_DIR = "./"
) (
    input  wire           clk,
    input  wire           in_valid,
    input  wire [L*8-1:0] in_g,
    input  wire [L*8-1:0] in_u,
    input  wire [15:0]    mult_g,
    input  wire [5:0]     sh_g,
    input  wire [15:0]    mult_o,
    input  wire [5:0]     sh_o,
    output reg            out_valid,
    output reg  [L*8-1:0] out_y
);
    // One multiply, or one round with its saturate, to a stage: the gate's
    // requantize is T1 and T2, the product with the up vector Q1, its scale
    // Q2 and the output's requantize Q3.  The up vector waits alongside.
    reg            v1, v2;
    reg [L*24-1:0] tm1;
    reg [L*16-1:0] tg2;
    integer c;
    always @(posedge clk) begin
        v1 <= in_valid;
        for (c = 0; c < L; c = c + 1)
            tm1[c*24 +: 24] <= $signed(in_g[c*8 +: 8]) * $signed({8'b0, mult_g});
        v2 <= v1;
        for (c = 0; c < L; c = c + 1)
            tg2[c*16 +: 16] <= fx_sat(fx_rnd_shr($signed(tm1[c*24 +: 24]), sh_g), 16);
    end
    wire [L-1:0]    sv;
    wire [L*16-1:0] s6;
    genvar g;
    generate
        for (g = 0; g < L; g = g + 1) begin : g_silu
            fabric_silu #(.LUT_DIR(LUT_DIR)) u_silu (.clk(clk), .valid_in(v2), .t(tg2[g*16 +: 16]), .valid_out(sv[g]), .y(s6[g*16 +: 16]));
        end
    endgenerate
    localparam int UD = 6;                       // T1, T2 and the SiLU's four
    reg [L*8-1:0] ud [0:UD-1];
    integer d;
    always @(posedge clk) begin
        ud[0] <= in_u;
        for (d = 1; d < UD; d = d + 1) ud[d] <= ud[d-1];
    end
    reg            q1v, q2v;
    reg [L*24-1:0] p1;
    reg [L*40-1:0] p2;
    always @(posedge clk) begin
        q1v <= sv[0];
        for (c = 0; c < L; c = c + 1)
            p1[c*24 +: 24] <= $signed(s6[c*16 +: 16]) * $signed(ud[UD-1][c*8 +: 8]);
        q2v <= q1v;
        for (c = 0; c < L; c = c + 1)
            p2[c*40 +: 40] <= $signed(p1[c*24 +: 24]) * $signed({8'b0, mult_o});
        out_valid <= q2v;
        for (c = 0; c < L; c = c + 1)
            out_y[c*8 +: 8] <= fx_sat(fx_rnd_shr($signed(p2[c*40 +: 40]), sh_o), 8);
    end
endmodule

// ---------------------------------------------------------------------------
// h' = sat16(h + (y * mult + 2^(shift-1)) >> shift).  Latency 2.
// ---------------------------------------------------------------------------
module fabric_residual #(
    parameter int L = 8
) (
    input  wire            clk,
    input  wire            in_valid,
    input  wire [L*16-1:0] in_h,
    input  wire [L*8-1:0]  in_y,
    input  wire [15:0]     mult,
    input  wire [5:0]      shift,
    output reg             out_valid,
    output reg  [L*16-1:0] out_h
);
    // The scale's multiply, then the round and the saturating add.
    reg            v1;
    reg [L*24-1:0] p1;
    reg [L*16-1:0] h1;
    integer c;
    always @(posedge clk) begin
        v1 <= in_valid;
        h1 <= in_h;
        for (c = 0; c < L; c = c + 1)
            p1[c*24 +: 24] <= $signed(in_y[c*8 +: 8]) * $signed({8'b0, mult});
        out_valid <= v1;
        for (c = 0; c < L; c = c + 1)
            out_h[c*16 +: 16] <= fx_sat($signed(h1[c*16 +: 16])
                                        + fx_rnd_shr($signed(p1[c*24 +: 24]), shift), 16);
    end
endmodule

`default_nettype wire
