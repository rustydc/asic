// Scalar fixed-point units shared by the vector datapath: interpolated
// tables, sigmoid, SiLU, exp(-t), softplus, inverse square root and
// reciprocal.  Formats: F16 is int16 with 10 fraction bits, U16 is unsigned
// Q0.16 with 1.0 clipped to 65535.  Every unit is a fixed-latency pipeline
// with a valid that travels alongside; the latencies are stated per module.
//
// Golden model: fabric/layer.py (sigmoid_fixed, silu_fixed, exp_neg_fixed,
// softplus_fixed, rsqrt_fixed, recip_fixed).

`default_nettype none
`include "fabric_fx.svh"

// ---------------------------------------------------------------------------
// Table with linear interpolation: 2^IB + 1 entries of W bits, an index of
// IB bits and FB fraction bits.  y = T[i] + ((T[i+1] - T[i]) * f + 2^(FB-1)) >> FB.
// Latency 2.
// ---------------------------------------------------------------------------
module fabric_lut #(
    parameter int IB = 8,
    parameter int FB = 6,
    parameter int W  = 16,
    parameter     FILE = ""
) (
    input  wire            clk,
    input  wire [IB+FB-1:0] u,
    output reg  [W-1:0]    y
);
    reg [W-1:0] t [0:(1<<IB)];
    initial begin
        if (FILE != "") $readmemh(FILE, t);
    end
    reg [W-1:0]  t0, t1;
    reg [FB-1:0] frac;
    always @(posedge clk) begin
        t0   <= t[u[IB+FB-1:FB]];
        t1   <= t[u[IB+FB-1:FB] + 1];
        frac <= u[FB-1:0];
    end
    wire signed [63:0] d    = $signed({{(64-W){1'b0}}, t1}) - $signed({{(64-W){1'b0}}, t0});
    wire signed [63:0] step = fx_rnd_shr(d * $signed({{(64-FB){1'b0}}, frac}), FB);
    wire signed [63:0] sum  = $signed({{(64-W){1'b0}}, t0}) + step;
    always @(posedge clk) y <= sum[W-1:0];
endmodule

// ---------------------------------------------------------------------------
// Sigmoid: F16 -> U16 over [-8, 8), the ends held.  Latency 3.
// ---------------------------------------------------------------------------
module fabric_sigmoid #(parameter LUT_DIR = "./") (
    input  wire               clk,
    input  wire               valid_in,
    input  wire signed [15:0] t,
    output reg                valid_out,
    output wire [15:0]        y
);
    reg  [13:0] u;
    wire signed [16:0] off = $signed({t[15], t}) + 17'sd8192;
    always @(posedge clk)
        u <= (off < 0) ? 14'd0 : (off > 17'sd16383) ? 14'd16383 : off[13:0];
    fabric_lut #(.IB(8), .FB(6), .W(16), .FILE({LUT_DIR, "lut_sigmoid.hex"})) lut (.clk(clk), .u(u), .y(y));
    reg v1, v2;
    always @(posedge clk) begin
        v1 <= valid_in;
        v2 <= v1;
        valid_out <= v2;
    end
endmodule

// ---------------------------------------------------------------------------
// SiLU: F16 -> F16, t * sigmoid(t) rounded from Q5.26.  Latency 4.
// ---------------------------------------------------------------------------
module fabric_silu #(parameter LUT_DIR = "./") (
    input  wire               clk,
    input  wire               valid_in,
    input  wire signed [15:0] t,
    output reg                valid_out,
    output reg  signed [15:0] y
);
    wire        sv;
    wire [15:0] sig;
    fabric_sigmoid #(.LUT_DIR(LUT_DIR)) sigm (.clk(clk), .valid_in(valid_in), .t(t), .valid_out(sv), .y(sig));
    reg signed [15:0] t1, t2, t3;
    always @(posedge clk) begin
        t1 <= t;
        t2 <= t1;
        t3 <= t2;
    end
    wire signed [63:0] p = fx_rnd_shr($signed({{48{t3[15]}}, t3}) * $signed({48'b0, sig}), 16);
    always @(posedge clk) begin
        y         <= p[15:0];
        valid_out <= sv;
    end
endmodule

// ---------------------------------------------------------------------------
// exp(-t) for an unsigned t in F16 units (22 bits): U16, zero from 32 up.
// Latency 3.
// ---------------------------------------------------------------------------
module fabric_exp_neg #(parameter LUT_DIR = "./") (
    input  wire        clk,
    input  wire        valid_in,
    input  wire [21:0] t,
    output reg         valid_out,
    output wire [15:0] y
);
    reg  [14:0] u;
    reg         in1, in2, in3;
    wire [15:0] raw;
    always @(posedge clk) begin
        u   <= (t >= 22'd32768) ? 15'd32767 : t[14:0];
        in1 <= (t < 22'd32768);
        in2 <= in1;
        in3 <= in2;
    end
    fabric_lut #(.IB(10), .FB(5), .W(16), .FILE({LUT_DIR, "lut_exp.hex"})) lut (.clk(clk), .u(u), .y(raw));
    assign y = in3 ? raw : 16'd0;
    reg v1, v2;
    always @(posedge clk) begin
        v1 <= valid_in;
        v2 <= v1;
        valid_out <= v2;
    end
endmodule

// ---------------------------------------------------------------------------
// softplus: F16 -> unsigned F16 over [-16, 16), the identity from 16 up.
// Latency 3.
// ---------------------------------------------------------------------------
module fabric_softplus #(parameter LUT_DIR = "./") (
    input  wire               clk,
    input  wire               valid_in,
    input  wire signed [15:0] t,
    output reg                valid_out,
    output wire [15:0]        y
);
    reg  [14:0] u;
    reg         big1, big2, big3;
    reg  [15:0] t1, t2, t3;
    wire signed [16:0] off = $signed({t[15], t}) + 17'sd16384;
    always @(posedge clk) begin
        u    <= (off < 0) ? 15'd0 : (off > 17'sd32767) ? 15'd32767 : off[14:0];
        big1 <= (t >= 16'sd16384);
        big2 <= big1;
        big3 <= big2;
        t1 <= t; t2 <= t1; t3 <= t2;
    end
    wire [15:0] raw;
    fabric_lut #(.IB(11), .FB(4), .W(16), .FILE({LUT_DIR, "lut_softplus.hex"})) lut (.clk(clk), .u(u), .y(raw));
    assign y = big3 ? t3 : raw;
    reg v1, v2;
    always @(posedge clk) begin
        v1 <= valid_in;
        v2 <= v1;
        valid_out <= v2;
    end
endmodule

// ---------------------------------------------------------------------------
// Inverse square root of an SW-bit unsigned (SW even, ss >= 1):
//   1/sqrt(ss) = r * 2^(a/2 - 15 - SW/2), r in Q1.15 (17 bits), a even.
// A table seed over the normalised operand and one Newton step.  Sequential:
// start, then done 5 cycles later.
// ---------------------------------------------------------------------------
module fabric_rsqrt #(
    parameter int SW = 44,
    parameter     LUT_DIR = "./"
) (
    input  wire          clk,
    input  wire          start,
    input  wire [SW-1:0] ss,
    output reg           done,
    output reg  [16:0]   r,
    output reg  [6:0]    a
);
    reg [16:0] seed [0:767];
    initial begin
        $readmemh({LUT_DIR, "lut_rsqrt.hex"}, seed);
    end
    // S1: leading zeros, even shift, 16-bit mantissa in [2^14, 2^16).
    // Leading zeros: the last set bit seen from the bottom is the highest.
    integer i;
    reg [6:0] lz;
    always @* begin
        lz = SW - 1;
        for (i = 0; i < SW; i = i + 1)
            if (ss[i]) lz = SW - 1 - i;
    end
    wire [6:0]    a_w = {lz[6:1], 1'b0};
    wire [SW-1:0] sh  = ss << a_w;
    reg  [15:0]   m1;
    reg  [6:0]    a1;
    reg           v1, v2, v3, v4;
    reg  [15:0]   m2, m3, m4;
    reg  [6:0]    a2, a3, a4;
    reg  [16:0]   r0_2, r0_3, r0_4;
    reg  [33:0]   sq3;
    reg  signed [63:0] u4;
    always @(posedge clk) begin
        v1 <= start;
        m1 <= sh[SW-1:SW-16];
        a1 <= a_w;
        // S2: seed.
        v2 <= v1; m2 <= m1; a2 <= a1;
        r0_2 <= seed[(m1 >> 6) - 256];
        // S3: r0^2.
        v3 <= v2; m3 <= m2; a3 <= a2; r0_3 <= r0_2;
        sq3 <= r0_2 * r0_2;
        // S4: u = 3 - M r0^2 in Q2.30.
        v4 <= v3; a4 <= a3; r0_4 <= r0_3;
        u4 <= (64'sd3 <<< 30) - (($signed({48'b0, m3}) * $signed({30'b0, sq3})) >>> 16);
        // S5: r1 = r0 (3 - M r0^2) / 2.
        done <= v4;
        r    <= ($signed({47'b0, r0_4}) * u4) >>> 31;
        a    <= a4;
    end
endmodule

// ---------------------------------------------------------------------------
// Reciprocal of an LW-bit unsigned (l >= 1):
//   1/l = r * 2^(lz - 15 - LW), r in Q1.15 (17 bits).
// Seed and one Newton step; start, then done 4 cycles later.
// ---------------------------------------------------------------------------
module fabric_recip #(
    parameter int LW = 28,
    parameter     LUT_DIR = "./"
) (
    input  wire          clk,
    input  wire          start,
    input  wire [LW-1:0] l,
    output reg           done,
    output reg  [16:0]   r,
    output reg  [5:0]    lz_out
);
    reg [16:0] seed [0:511];
    initial begin
        $readmemh({LUT_DIR, "lut_recip.hex"}, seed);
    end
    integer i;
    reg [5:0] lz;
    always @* begin
        lz = LW - 1;
        for (i = 0; i < LW; i = i + 1)
            if (l[i]) lz = LW - 1 - i;
    end
    wire [LW-1:0] sh = l << lz;
    wire [15:0]   m_w;
    generate
        if (LW >= 16) begin : g_wide
            assign m_w = sh[LW-1:LW-16];
        end else begin : g_narrow
            assign m_w = {sh, {(16-LW){1'b0}}};
        end
    endgenerate
    reg  [15:0]   m1, m2;
    reg  [5:0]    z1, z2, z3;
    reg           v1, v2, v3;
    reg  [16:0]   r0_2, r0_3;
    reg  signed [63:0] u3;
    always @(posedge clk) begin
        v1 <= start; m1 <= m_w; z1 <= lz;
        v2 <= v1; m2 <= m1; z2 <= z1;
        r0_2 <= seed[(m1 >> 6) - 512];
        v3 <= v2; z3 <= z2; r0_3 <= r0_2;
        u3 <= (64'sd2 <<< 15) - (($signed({48'b0, m2}) * $signed({47'b0, r0_2})) >>> 16);
        done   <= v3;
        r      <= ($signed({47'b0, r0_3}) * u3) >>> 15;
        lz_out <= z3;
    end
endmodule

`default_nettype wire
