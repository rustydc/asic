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
// One registered copy of a value a lane needs.  A shift amount shared by L
// lanes drives every lane's shifter select, hundreds of loads from one flop;
// a copy per lane divides that by L for a handful of flops.  Its own
// hierarchy, so synthesis cannot merge the copies back into one net (the
// tile's fabric_strobe_copy does the same for a one-bit strobe).
// ---------------------------------------------------------------------------
(* keep_hierarchy *)
module fabric_const_copy #(
    parameter int W = 6
) (
    input  wire         clk,
    input  wire [W-1:0] d,
    output reg  [W-1:0] q
);
    always @(posedge clk) q <= d;
endmodule

// ---------------------------------------------------------------------------
// Carry-save reduction of N operands to a (sum, carry) pair: value = s + 2c.
// Each layer turns groups of three operands into two; depth is logarithmic.
// The tile's column datapath is built on this (fabric_tile.sv); it lives here
// because the vector units want it for the same reason.
// ---------------------------------------------------------------------------
// Written as a loop over the layers rather than a module that instantiates
// itself.  Recursion reads better and cost a module of hierarchy per layer:
// the attention core's contribution tree is fifteen layers at the width its
// testbench uses, which is past Icarus's recursion limit of ten and fifteen
// levels of hierarchy for synthesis to flatten.  Same structure, one module.
module fabric_csa_tree #(
    parameter int N = 3,
    parameter int W = 41
) (
    input  wire [N*W-1:0] ops,
    output wire [W-1:0]   s,
    output wire [W-1:0]   c
);
    // Operands left after `lv` layers, and the number of layers to reach two.
    function automatic integer after;
        input integer n;
        input integer lv;
        integer k;
        begin
            after = n;
            for (k = 0; k < lv; k = k + 1)
                if (after > 2) after = 2 * (after / 3) + (after % 3);
        end
    endfunction
    function automatic integer layers;
        input integer n;
        integer m;
        begin
            layers = 0;
            m = n;
            while (m > 2) begin
                m = 2 * (m / 3) + (m % 3);
                layers = layers + 1;
            end
        end
    endfunction
    localparam int LV = layers(N);
    localparam int NF = after(N, LV);        // one or two
    // Every layer is narrower than the one before it, so one N-operand slot
    // per layer holds all of them.
    wire [(LV+1)*N*W-1:0] lvl;
    assign lvl[0 +: N*W] = ops;
    genvar v, i;
    generate
        for (v = 0; v < LV; v = v + 1) begin : g_lv
            localparam int NI = after(N, v);
            localparam int NG = NI / 3;
            localparam int NR = NI % 3;
            for (i = 0; i < NG; i = i + 1) begin : g_csa
                wire [W-1:0] a = lvl[(v*N + 3*i)*W     +: W];
                wire [W-1:0] b = lvl[(v*N + 3*i + 1)*W +: W];
                wire [W-1:0] d = lvl[(v*N + 3*i + 2)*W +: W];
                wire [W-1:0] cy = (a & b) | (a & d) | (b & d);
                assign lvl[((v+1)*N + 2*i)*W     +: W] = a ^ b ^ d;
                assign lvl[((v+1)*N + 2*i + 1)*W +: W] = {cy[W-2:0], 1'b0};   // carry at weight 2, as a plain operand
            end
            for (i = 0; i < NR; i = i + 1) begin : g_pass
                assign lvl[((v+1)*N + 2*NG + i)*W +: W] = lvl[(v*N + 3*NG + i)*W +: W];
            end
        end
    endgenerate
    wire [W-1:0] f0 = lvl[(LV*N)*W +: W];
    wire [W-1:0] f1 = (NF == 2) ? lvl[(LV*N + 1)*W +: W] : {W{1'b0}};
    assign s = (NF == 2) ? (f0 ^ f1) : f0;
    assign c = (NF == 2) ? (f0 & f1) : {W{1'b0}};
endmodule

// ---------------------------------------------------------------------------
// One requantize: round-shift a W-bit signed value right by sh, then saturate
// to N bits.  The same number as fx_sat(fx_rnd_shr(v, sh), N), at the width
// the value actually has.
//
// The helpers in fabric_fx.svh evaluate at 64 bits so that a chain of two or
// three multiplies never truncates.  That is right for the arithmetic and
// expensive here: a 33-bit product sign-extended to 64 buys a barrel shifter
// of twice the width, and each bit of the shift amount selects every mux in
// its own level, so it doubles that register's fanout too -- and a register's
// own output is the one net the mapper cannot buffer.  The convolution spent
// 421 of its 2,071 ps on the clk-to-Q of one shift-amount bit driving 87
// loads.  Saturating with a compare costs a second carry chain the width of
// the shifter; a value fits in N bits exactly when its bits above N-1 are all
// copies of bit N-1, which is a pair of reduction trees instead.
// ---------------------------------------------------------------------------
// The shift half: the value shifted, and the round bit beside it.  Kept
// separate so a pipeline can put a register here -- the shift and the
// saturate are each about half of the requantize, and in one stage they were
// the convolution's two worst paths.
module fabric_rnd_sat_shift #(
    parameter int W  = 32,                 // width of the value
    parameter int SW = 6                   // width of the shift amount
) (
    input  wire signed [W-1:0] v,
    input  wire [SW-1:0]       sh,
    output wire signed [W:0]   sv,
    output wire                rb
);
    localparam int SMAX = 1 << SW;
    // The round bit is bit sh-1 of v; a shift of zero rounds nothing.  Above
    // W-1 every bit of v is the sign bit, so those arms of the mux are the
    // same net and fold away.
    localparam int VW = (W > SMAX) ? W : SMAX;
    wire signed [VW-1:0] vx = $signed(v);
    reg r;
    integer i;
    always @* begin
        r = 1'b0;
        for (i = 1; i < SMAX; i = i + 1)
            if (sh == i[SW-1:0]) r = vx[i-1];
    end
    assign rb = r;
    // One bit above the sign, so the top two bits of `sv` always agree; that
    // is what lets the carry out of the low half be resolved in the round
    // half without a second add.
    assign sv = $signed({v[W-1], v}) >>> sh;
endmodule

// The round and saturate half.
module fabric_rnd_sat_round #(
    parameter int W = 32,
    parameter int N = 8                    // width of the result
) (
    input  wire signed [W:0]   sv,
    input  wire                rb,
    output wire signed [N-1:0] y
);
    wire [N-1:0]      svl = sv[N-1:0];
    wire [W-N:0]      hi  = sv[W:N];
    // Only the low N bits of the rounded value survive the saturate, so only
    // they are added.  Rounding the whole width is a carry chain as long as
    // the value: 34 gates of it, and the convolution's path once the shifter
    // came off it.  The carry out of the low bits is the round bit and those
    // bits all ones, and the high part's only job is to say whether the
    // result still fits -- which is two reductions, not a compare.
    wire [N-1:0]   lo  = svl + {{(N-1){1'b0}}, rb};
    wire           cy  = rb & (&svl);
    wire [W-N+1:0] top = {hi, lo[N-1]};
    // With a carry the low bits are zero, so the result fits exactly when the
    // high part was all ones and the carry cleared it.
    wire fits = cy ? (&hi) : ((&top) | (~|top));
    assign y = fits ? lo : (sv[W] ? {1'b1, {(N-1){1'b0}}} : {1'b0, {(N-1){1'b1}}});
endmodule

// Both halves with nothing between them, for the stages that can afford it.
module fabric_rnd_sat #(
    parameter int W  = 32,
    parameter int SW = 6,
    parameter int N  = 8
) (
    input  wire signed [W-1:0] v,
    input  wire [SW-1:0]       sh,
    output wire signed [N-1:0] y
);
    wire signed [W:0] sv;
    wire              rb;
    fabric_rnd_sat_shift #(.W(W), .SW(SW)) u_s (.v(v), .sh(sh), .sv(sv), .rb(rb));
    fabric_rnd_sat_round #(.W(W), .N(N))   u_r (.sv(sv), .rb(rb), .y(y));
endmodule

// ---------------------------------------------------------------------------
// The round-shift left in two pieces: the value is sv + rb, and nothing here
// propagates a carry.  A stage that consumes it usually has an add of its own
// -- a difference against the running maximum, an accumulate -- and the round
// bit rides into that tree for one more operand, where resolving it here
// costs an incrementer as wide as the value.
module fabric_rnd_cs #(
    parameter int W  = 48,
    parameter int SW = 6
) (
    input  wire signed [W-1:0] v,
    input  wire [SW-1:0]       sh,
    output wire signed [W-1:0] sv,
    output wire                rb
);
    localparam int SMAX = 1 << SW;
    localparam int VW = (W > SMAX) ? W : SMAX;
    wire signed [VW-1:0] vx = $signed(v);
    reg r;
    integer i;
    always @* begin
        r = 1'b0;
        for (i = 1; i < SMAX; i = i + 1)
            if (sh == i[SW-1:0]) r = vx[i-1];
    end
    assign rb = r;
    assign sv = v >>> sh;
endmodule

// ---------------------------------------------------------------------------
// A signed multiplicand by an unsigned multiplier, left in carry-save form:
// value = s + 2c, and nothing along the way propagates a carry.  Whatever the
// stage was going to add to the product -- a rounding constant, a bias --
// goes in as ADD more operands of the same tree for the price of a layer.
//
// A multiply written as `a * b` is a partial-product tree *and* a final
// carry-propagate add of PW bits, and that add is most of its delay: it is
// the floor the vector units sit on.  Split here, the tree is one stage and
// the resolve belongs to the next, where it shares a stage with whatever
// followed it -- which is the tile's own arrangement (fabric_tile.sv, stages
// M and A1..A3).  The caller resolves with a plain `s + {c, 1'b0}`.
// ---------------------------------------------------------------------------
module fabric_mul_cs #(
    parameter int AW  = 24,          // multiplicand bits, signed
    parameter int BW  = 16,          // multiplier bits, unsigned
    parameter int PW  = 41,          // product width: AW + BW + 1 is exact
    parameter int ADD = 1            // extra operands added into the tree
) (
    input  wire signed [AW-1:0] a,
    input  wire [BW-1:0]        b,
    input  wire [ADD*PW-1:0]    addend,
    output wire [PW-1:0]        s,
    output wire [PW-1:0]        c
);
    wire [PW-1:0] ax = {{(PW-AW){a[AW-1]}}, a};
    wire [(BW+ADD)*PW-1:0] ops;
    genvar i;
    generate
        for (i = 0; i < BW; i = i + 1) begin : g_pp
            assign ops[i*PW +: PW] = b[i] ? (ax <<< i) : {PW{1'b0}};
        end
        for (i = 0; i < ADD; i = i + 1) begin : g_add
            assign ops[(BW + i)*PW +: PW] = addend[i*PW +: PW];
        end
    endgenerate
    fabric_csa_tree #(.N(BW + ADD), .W(PW)) u_tree (.ops(ops), .s(s), .c(c));
endmodule

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
    // The table stays an array of constants rather than a memory macro, which
    // is the one place in this design where that is the right answer: a sine
    // or a sigmoid is smooth, so synthesis folds 2^IB entries into a fraction
    // of the logic their bits would suggest, where an SRAM of the same bits is
    // paid for in full.  Made macros, the tables cost the head gates 23,190
    // NAND2-eq -> 153,257 and the rotary table 15,493 -> 51,422, and both got
    // slower by the macro's access time.
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
    // The interpolation at the width it has.  Two W-bit entries differ by
    // W + 1 bits, the fraction is FB, and W bits of the sum are kept -- but
    // through the helpers in fabric_fx.svh it was a 64-bit subtract, a 64-bit
    // multiply, a 64-bit round and a 64-bit add for sixteen bits of answer.
    // This table is under every scalar unit in the design: the sigmoid, the
    // SiLU, exp, softplus, the inverse square root, the reciprocal and the
    // rotary table.
    localparam int PW = W + 1 + FB;
    wire signed [W:0]    d    = $signed({1'b0, t1}) - $signed({1'b0, t0});
    wire signed [PW-1:0] prod = d * $signed({1'b0, frac});
    // The shift is by a constant, so the round is an increment on the bits
    // that are kept: bit FB-1 of the product is the round bit.
    wire signed [W:0]    step = $signed(prod[PW-1:FB]) + {{W{1'b0}}, prod[FB-1]};
    wire signed [W+1:0]  sum  = $signed({2'b0, t0}) + $signed({step[W], step});
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
    // Sixteen by sixteen is 32 bits, and sixteen of them are kept: the round
    // by a constant sixteen is an increment on the top half, not a 64-bit
    // shift of a 64-bit product.
    wire signed [32:0] pr = $signed(t3) * $signed({1'b0, sig});
    always @(posedge clk) begin
        y         <= pr[31:16] + {15'b0, pr[15]};
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
// start, then done 6 cycles later (one multiply to a stage).
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
    reg           v1, v2, v3, v4, v5;
    reg  [15:0]   m2, m3;
    reg  [6:0]    a2, a3, a4, a5;
    reg  [16:0]   r0_2, r0_3, r0_4, r0_5;
    reg  [33:0]   sq3;
    reg  [49:0]   p4;
    // 3 - M r0^2 spans [3*2^30 - 2^34, 3*2^30], which is 35 bits.  At 64 the
    // Newton step below is a 17 by 64 multiply for a 17-bit result.
    reg  signed [34:0] u5;
    // The product needs its own name: assigned straight into `r` the context
    // is seventeen bits and the multiply is evaluated there, which is the
    // same width-inference trap the 64-bit extensions were hiding.
    wire signed [52:0] rp5;
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
        // S4: M r0^2, the multiply on its own.
        v4 <= v3; a4 <= a3; r0_4 <= r0_3;
        p4 <= m3 * sq3;
        // S5: u = 3 - M r0^2 in Q2.30.
        v5 <= v4; a5 <= a4; r0_5 <= r0_4;
        u5 <= $signed(35'sd3 <<< 30) - $signed({1'b0, p4[49:16]});
        // S6: r1 = r0 (3 - M r0^2) / 2.
        done <= v5;
        r    <= rp5 >>> 31;
        a    <= a5;
    end
    assign rp5 = $signed({1'b0, r0_5}) * u5;
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
    // 2 - m r0 spans [-2^16, 2^16], which is 18 bits, and the same 17 by 64
    // multiply followed it.
    reg  signed [17:0] u3;
    wire [32:0]   mr = m2 * r0_2;
    wire signed [35:0] rp3;
    always @(posedge clk) begin
        v1 <= start; m1 <= m_w; z1 <= lz;
        v2 <= v1; m2 <= m1; z2 <= z1;
        r0_2 <= seed[(m1 >> 6) - 512];
        v3 <= v2; z3 <= z2; r0_3 <= r0_2;
        u3 <= $signed(18'sd2 <<< 15) - $signed({1'b0, mr[32:16]});
        done   <= v3;
        r      <= rp3 >>> 15;
        lz_out <= z3;
    end
    assign rp3 = $signed({1'b0, r0_3}) * u3;
endmodule

`default_nettype wire
