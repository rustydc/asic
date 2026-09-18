// SwiGLU and the residual add, L elements per beat.
// Golden model: fabric/layer.py (swiglu_int, residual_int).

`default_nettype none
`include "fabric_fx.svh"

// ---------------------------------------------------------------------------
// act = sat8((silu(g * mult_g >> sh_g) * u) * mult_o >> sh_o).  Latency 6.
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
    reg            v1;
    reg [L*16-1:0] tg1;
    integer c;
    always @(posedge clk) begin
        v1 <= in_valid;
        for (c = 0; c < L; c = c + 1)
            tg1[c*16 +: 16] <= fx_requant($signed(in_g[c*8 +: 8]), mult_g, sh_g, 16);
    end
    wire [L-1:0]    sv;
    wire [L*16-1:0] s5;
    genvar g;
    generate
        for (g = 0; g < L; g = g + 1) begin : g_silu
            fabric_silu #(.LUT_DIR(LUT_DIR)) u_silu (.clk(clk), .valid_in(v1), .t(tg1[g*16 +: 16]), .valid_out(sv[g]), .y(s5[g*16 +: 16]));
        end
    endgenerate
    reg [L*8-1:0] u1, u2, u3, u4, u5;
    always @(posedge clk) begin
        u1 <= in_u; u2 <= u1; u3 <= u2; u4 <= u3; u5 <= u4;
    end
    always @(posedge clk) begin
        out_valid <= sv[0];
        for (c = 0; c < L; c = c + 1)
            out_y[c*8 +: 8] <= fx_requant($signed(s5[c*16 +: 16]) * $signed(u5[c*8 +: 8]), mult_o, sh_o, 8);
    end
endmodule

// ---------------------------------------------------------------------------
// h' = sat16(h + (y * mult + 2^(shift-1)) >> shift).  Latency 1.
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
    integer c;
    always @(posedge clk) begin
        out_valid <= in_valid;
        for (c = 0; c < L; c = c + 1)
            out_h[c*16 +: 16] <= fx_sat($signed(in_h[c*16 +: 16])
                                        + fx_rnd_shr($signed(in_y[c*8 +: 8]) * $signed({48'b0, mult}), shift), 16);
    end
endmodule

`default_nettype wire
