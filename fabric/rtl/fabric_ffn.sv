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
        tg2 <= tg2_n;
    end
    // Both requantizes at the width their value has -- 24 bits for the gate
    // and 40 for the output -- rather than the 64 the helpers in
    // fabric_fx.svh evaluate at.  At 64 the barrel shifter, the rounding
    // incrementer and the saturating compare are each more than twice as
    // wide as the number going through them, and the output's was this
    // unit's path at 2,104 ps with the gate's behind it at 1,736.
    wire [L*16-1:0] tg2_n;
    wire [L*8-1:0]  outy_n;
    genvar gq;
    generate
        for (gq = 0; gq < L; gq = gq + 1) begin : g_rq
            fabric_rnd_sat #(.W(24), .SW(6), .N(16)) u_tg (
                .v(tm1[gq*24 +: 24]), .sh(sh_g), .y(tg2_n[gq*16 +: 16]));
            fabric_rnd_sat #(.W(40), .SW(6), .N(8)) u_oy (
                .v(p2[gq*40 +: 40]), .sh(sh_o), .y(outy_n[gq*8 +: 8]));
        end
    endgenerate
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
        out_y     <= outy_n;
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
    // The scale's multiply, then the round, then the saturating add: one
    // operation a stage.  The round and the add together were a variable
    // shift, a 24-bit add and a saturate between two flops, and the shift
    // came straight off the command's input pin, so each lane takes its own
    // registered copy of it as the norm and the state engine do.
    reg            v1, v2;
    reg [L*24-1:0] p1, p2;
    reg [L*16-1:0] h1, h2;
    // The round and the saturating add at the width of the numbers.  Both
    // went through the fx_ helpers, whose argument is a signed [63:0], so a
    // 24-bit round-shift and a 25-bit add were done at 64 -- and the round's
    // mux over the shift amount picked one of 64 bits where the value only
    // has 24.
    reg signed [23:0] pshr;
    reg               prnd;
    reg signed [24:0] hsum;
    wire [5:0]     sh_l [0:L-1];
    genvar gl;
    generate
        for (gl = 0; gl < L; gl = gl + 1) begin : g_sh
            fabric_const_copy #(.W(6)) u_sh (.clk(clk), .d(shift), .q(sh_l[gl]));
        end
    endgenerate
    integer c, b;
    always @(posedge clk) begin
        v1 <= in_valid;
        h1 <= in_h;
        for (c = 0; c < L; c = c + 1)
            p1[c*24 +: 24] <= $signed(in_y[c*8 +: 8]) * $signed({8'b0, mult});
        v2 <= v1;
        h2 <= h1;
        for (c = 0; c < L; c = c + 1)
            begin
                pshr = $signed(p1[c*24 +: 24]) >>> sh_l[c];
                // The round bit is bit sh-1, a mux over the shift amount.
                // Past the width every bit is the sign, as it was when the
                // value was sign-extended to 64 first.
                prnd = 1'b0;
                for (b = 1; b < 24; b = b + 1)
                    if (sh_l[c] == b[5:0]) prnd = p1[c*24 + b - 1];
                if (sh_l[c] >= 6'd24) prnd = p1[c*24 + 23];
                p2[c*24 +: 24] <= pshr + {23'b0, prnd};
            end
        out_valid <= v2;
        for (c = 0; c < L; c = c + 1)
            begin
                hsum = $signed(h2[c*16 +: 16]) + $signed(p2[c*24 +: 24]);
                out_h[c*16 +: 16] <= ((&hsum[24:15]) | (~|hsum[24:15]))
                                     ? hsum[15:0] : (hsum[24] ? 16'sh8000 : 16'sh7FFF);
            end
    end
endmodule

`default_nettype wire
