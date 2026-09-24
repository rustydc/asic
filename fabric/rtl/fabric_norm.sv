// RMS norm over a vector of D elements, L per beat.
//
// The vector streams in and is buffered while its sum of squares
// accumulates; the inverse square root of (ss + eps) is then formed once
// and the vector streams back out as
//   n = sat16((x * R) >> (1 + SW/2 - a/2))          (x / sqrt(ss) in Q.14)
//   y = sat_OW((n * gain * mult + 2^(shift-1)) >> shift)
// The per-element gain is the norm weight when it cannot be folded into the
// following matrix, or the gate of the gated norm, and travels with the
// input beat.  sqrt(D) is in mult; with mult = 1 and shift = 7 the unit is
// an L2 normaliser emitting an int8 unit vector.
//
// Latency: D/L input beats, 7 cycles, then D/L output beats, the first
// 6 cycles after the drain starts (the drain is pipelined a multiply to a
// stage; out_valid carries the beats, so the length is the adapter's only
// contract).  A vector
// shorter than D is normalised over its first n_beats beats (D/L at most):
// the result does not depend on SW beyond it being wide enough, so one
// unit serves the residual norm, the unit norms and the gated norm.
// Golden model: fabric/layer.py rmsnorm_int.

`default_nettype none
`include "fabric_fx.svh"

module fabric_rmsnorm #(
    parameter int D  = 4096,
    parameter int XW = 16,          // input element width
    parameter int OW = 8,           // output element width
    parameter int L  = 2,           // elements per beat
    parameter int SW = 44,          // width of the sum of squares (even)
    parameter int GW = 16,          // gain width
    parameter     LUT_DIR = "./"
) (
    input  wire            clk,
    input  wire            rst_n,
    input  wire            in_valid,
    input  wire [$clog2(D/L):0] n_beats,      // beats of this vector, D/L at most
    input  wire [L*XW-1:0] in_x,
    input  wire [L*GW-1:0] in_gain,
    input  wire [15:0]     mult,
    input  wire [5:0]      shift,
    input  wire [SW-1:0]   eps,
    output reg             out_valid,
    output reg  [L*OW-1:0] out_y
);
    localparam int BEATS = D / L;
    localparam int BW    = $clog2(BEATS) + 1;

    // The vector waits here between the sum pass and the drain.  Macros, not
    // register arrays: at the 9B width these are 512 beats of 128 bits each,
    // which as registers is 131,072 flops -- 1.55M NAND2 equivalents, more
    // than any other unit in the engine -- and the read is a 512-to-1 mux
    // whose address carried 203 loads and 1.72 of this unit's 2.31 ns.  The
    // rotation and the attention core already hold their beats this way.
    reg [BW-1:0]   wr;
    reg [SW-1:0]   ss;

    // Fill pipeline: F1 squares the beat's elements, F2 adds each lane's
    // square into that lane's own running sum.  One multiply and one add to a
    // stage, and a sum per lane so no adder tree stands between them; the L
    // sums are added together once per vector in the reduce phase, which
    // costs L + 2 cycles against the vector's D/L beats.  Wrapping at SW bits
    // is associative, so the total is the one the model accumulates.
    localparam int PW = 2 * XW;
    reg [SW-1:0]   ssk [0:L-1];
    reg [L*PW-1:0] sq1;
    reg            f1;
    integer i;

    // Phases: 0 fill, 1 reduce, 2 rsqrt in flight, 3 drain.
    reg [1:0]  phase;
    reg        rs_start;
    wire       rs_done;
    wire [16:0] r_w;
    wire [6:0]  a_w;
    reg  [16:0] r;
    // The normalising shift and its rounding constant, decoded once per
    // vector when the inverse square root lands: the exponent would otherwise
    // drive every lane's shifter select through a subtract and a decoder,
    // which is the drain's longest path.
    reg  [5:0]  sh_u;
    reg  signed [XW+17:0] rnd_r;
    // The output requantizer's shift is a command constant, stable for the
    // whole vector; its rounding constant is registered for the same reason.
    reg  signed [47:0] ornd_r;
    always @(posedge clk) ornd_r <= (shift == 0) ? 48'sd0 : (48'sd1 <<< (shift - 1));
    // A copy of each shift amount per lane: one flop driving every lane's
    // shifter select is hundreds of loads, and the rounding constants' bits
    // fan out only L ways.
    wire [5:0]  sh_l [0:L-1];
    wire [5:0]  osh_l [0:L-1];
    wire [16:0] r_l [0:L-1];
    genvar gl;
    generate
        for (gl = 0; gl < L; gl = gl + 1) begin : g_sh
            fabric_const_copy #(.W(6))  u_sh  (.clk(clk), .d(sh_u),  .q(sh_l[gl]));
            fabric_const_copy #(.W(6))  u_osh (.clk(clk), .d(shift), .q(osh_l[gl]));
            fabric_const_copy #(.W(17)) u_r   (.clk(clk), .d(r),     .q(r_l[gl]));
        end
    endgenerate
    wire [SW-1:0] ss_eps = ss + eps;
    wire [SW-1:0] ss_in  = (ss_eps == 0) ? {{(SW-1){1'b0}}, 1'b1} : ss_eps;
    fabric_rsqrt #(.SW(SW), .LUT_DIR(LUT_DIR)) rsq (.clk(clk), .start(rs_start), .ss(ss_in), .done(rs_done), .r(r_w), .a(a_w));

    reg [BW-1:0] rd, rd_addr;
    reg          rd_valid;
    reg [$clog2(L+2):0] red;
    integer      sh_next;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase <= 2'd0; wr <= 0; ss <= 0; rs_start <= 1'b0; rd <= 0; rd_valid <= 1'b0; sh_u <= 0; rnd_r <= 0;
            f1 <= 1'b0; red <= 0;
            for (i = 0; i < L; i = i + 1) ssk[i] <= 0;
        end else begin
            rs_start <= 1'b0;
            rd_valid <= 1'b0;
            // F1 then F2, running whatever the phase: the last beat's square
            // lands two cycles after it arrives, which the reduce waits out.
            f1 <= (phase == 2'd0) && in_valid;
            for (i = 0; i < L; i = i + 1)
                sq1[i*PW +: PW] <= $signed(in_x[i*XW +: XW]) * $signed(in_x[i*XW +: XW]);
            if (f1)
                for (i = 0; i < L; i = i + 1)
                    ssk[i] <= ssk[i] + {{(SW-PW){1'b0}}, sq1[i*PW +: PW]};
            case (phase)
                2'd0: if (in_valid) begin
                    wr <= wr + 1'b1;
                    if (wr == n_beats - 1) begin
                        phase <= 2'd1;
                        red <= 0;
                        ss <= 0;
                    end
                end
                2'd1: begin
                    // Two cycles for the squares to land, then one lane's sum
                    // a cycle; each is cleared as it is taken.
                    red <= red + 1'b1;
                    if (red >= 2) begin
                        ss <= ss + ssk[red - 2];
                        ssk[red - 2] <= 0;
                        if (red == L + 1) begin
                            phase <= 2'd2;
                            rs_start <= 1'b1;
                        end
                    end
                end
                2'd2: begin
                    if (rs_done) begin
                        r <= r_w;
                        sh_next = 1 + SW / 2 - a_w / 2;
                        sh_u  <= (sh_next > 0) ? sh_next[5:0] : 6'd0;
                        rnd_r <= (sh_next > 0) ? (1 <<< (sh_next - 1)) : 0;
                        phase <= 2'd3;
                        rd <= 0;
                    end
                end
                default: begin
                    rd_valid <= 1'b1;
                    rd_addr <= rd;
                    rd <= rd + 1'b1;
                    if (rd == n_beats - 1) begin
                        phase <= 2'd0;
                        wr <= 0;
                    end
                end
            endcase
        end
    end
    // The start pulse is registered with the last lane sum, so the square
    // root unit samples the complete ss.

    // Drain pipeline: P0 read, P1 x * r, P2 normalise, P3 n * gain,
    // P4 * mult, P5 round, P6 shift and saturate.  One multiply is the clock's floor (1.4 ns
    // on NanGate 45 post-synthesis), and a round with its shift and saturate
    // costs the same again, so no stage holds two of them: the arithmetic is
    // the same expressions as the model's, cut between the multiplies.
    localparam int MW = XW + 18;                 // x * r
    localparam int NG = 16 + GW;                 // n * gain
    localparam int QW = NG + 16;                 // that * mult
    // The macros answer the cycle after their address, which is the cycle
    // `x0 <= xmem[rd_addr]` used to land in, so `rd_addr` is the address to
    // present and the beat arrives where the drain pipeline expects it.
    localparam int RA = (BEATS > 1) ? $clog2(BEATS) : 1;
    wire [L*XW-1:0] x0;
    wire [L*GW-1:0] g0;
    fabric_sram #(.W(L*XW), .D(BEATS), .NRD(1), .NWR(1), .MB(L*XW)) u_x (
        .clk(clk), .rd_en(1'b1), .rd_addr(rd_addr[RA-1:0]), .rd_data(x0),
        .wr_en(in_valid && phase == 2'd0), .wr_addr(wr[RA-1:0]), .wr_data(in_x), .wr_mask(1'b1));
    fabric_sram #(.W(L*GW), .D(BEATS), .NRD(1), .NWR(1), .MB(L*GW)) u_g (
        .clk(clk), .rd_en(1'b1), .rd_addr(rd_addr[RA-1:0]), .rd_data(g0),
        .wr_en(in_valid && phase == 2'd0), .wr_addr(wr[RA-1:0]), .wr_data(in_gain), .wr_mask(1'b1));
    reg [L*GW-1:0] g1, g2;
    reg            v0, v1, v2, v3, v4, v5;
    reg signed [MW-1:0] m1 [0:L-1];
    reg [L*16-1:0] n2;
    reg signed [NG-1:0] p3 [0:L-1];
    reg signed [QW-1:0] q4 [0:L-1];
    reg signed [QW-1:0] s5 [0:L-1];
    // Both requantizers ran at 64 bits.  fx_sat takes a signed [63:0], and a
    // function argument is an assignment, so the add and the variable shift
    // in front of it were widened to 64 as well -- on a product that is
    // MW bits and a result that is sixteen.  They run at their own width
    // now, and the saturate is the check the rounding primitives use: the
    // bits above the ones kept must all match their sign, which is two
    // reductions rather than a pair of compares against the limits.
    reg signed [MW:0]   nsm;
    reg signed [QW-1:0] osm;
    integer k;
    // The macro's read is a stage of its own.  Feeding its output straight
    // into the multiply put the access time and a sixteen-bit multiply in
    // one cycle -- at the 9B width, 1.16 of the unit's 2.66 ns was the macro
    // getting its data out -- so the stage is now the longer of the two
    // rather than their sum, for one cycle of latency.
    reg [L*XW-1:0] x0q;
    reg [L*GW-1:0] g0q;
    reg            v0q;
    always @(posedge clk) begin
        v0 <= rd_valid;
        v0q <= v0;
        x0q <= x0;
        g0q <= g0;
        v1 <= v0q;
        g1 <= g0q;
        for (k = 0; k < L; k = k + 1)
            m1[k] <= $signed({{(MW-XW){x0q[k*XW+XW-1]}}, x0q[k*XW +: XW]}) * $signed({{(MW-17){1'b0}}, r_l[k]});
        v2 <= v1;
        g2 <= g1;
        for (k = 0; k < L; k = k + 1)
            begin
                nsm = (m1[k] + rnd_r) >>> sh_l[k];
                n2[k*16 +: 16] <= ((&nsm[MW:15]) | (~|nsm[MW:15]))
                                  ? nsm[15:0] : (nsm[MW] ? 16'sh8000 : 16'sh7FFF);
            end
        v3 <= v2;
        for (k = 0; k < L; k = k + 1)
            p3[k] <= $signed({{(NG-16){n2[k*16+15]}}, n2[k*16 +: 16]}) * $signed({{(NG-GW){g2[k*GW+GW-1]}}, g2[k*GW +: GW]});
        v4 <= v3;
        for (k = 0; k < L; k = k + 1)
            q4[k] <= $signed({{(QW-NG){p3[k][NG-1]}}, p3[k]}) * $signed({{(QW-16){1'b0}}, mult});
        v5 <= v4;
        for (k = 0; k < L; k = k + 1)
            s5[k] <= q4[k] + ornd_r;
        out_valid <= v5;
        for (k = 0; k < L; k = k + 1)
            begin
                osm = s5[k] >>> osh_l[k];
                out_y[k*OW +: OW] <= ((&osm[QW-1:OW-1]) | (~|osm[QW-1:OW-1]))
                                     ? osm[OW-1:0]
                                     : (osm[QW-1] ? {1'b1, {(OW-1){1'b0}}} : {1'b0, {(OW-1){1'b1}}});
            end
    end
endmodule

`default_nettype wire
