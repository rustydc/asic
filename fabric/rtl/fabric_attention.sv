// The global layer's vector units: the rotary table and rotation, and the
// online-softmax attention core.
// Golden model: fabric/layer.py (rotary_table_int, rotary_int, attention_int).

`default_nettype none
`include "fabric_fx.svh"

// ---------------------------------------------------------------------------
// sin and cos of pos * inv_freq[j] for the R/2 rotary frequencies of one
// position: the product in Q0.32 turns keeps its fraction, whose top 16
// bits index the sine table (cos is sin a quarter turn on).  start, then
// done after R/2 + 4 cycles; the tables stay valid until the next start.
// ---------------------------------------------------------------------------
module fabric_rotary_table #(
    parameter int R = 64,
    parameter     LUT_DIR = "./"
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               start,
    input  wire [31:0]        pos,
    input  wire [(R/2)*32-1:0] inv_freq,
    output reg                done,
    output reg  [(R/2)*16-1:0] sin_tab,
    output reg  [(R/2)*16-1:0] cos_tab
);
    localparam int H  = R / 2;
    localparam int JW = $clog2(H) + 1;
    reg          busy;
    reg [JW-1:0] j;
    // The frequency this cycle, as a one-hot select rather than
    // `inv_freq[j*32 +: 32]`.  A variable part-select of the whole vector is
    // a barrel shifter over all H*32 bits, and `j` addresses twice as many
    // positions as there are frequencies, so one bit of `j` reached 132
    // loads: 651 ps of clock-to-output, 31 percent of this unit's path.  An
    // or of masks is the structure the index actually has, and it costs `j`
    // H comparators.
    wire [H-1:0] jsel;
    genvar gjs;
    generate
        for (gjs = 0; gjs < H; gjs = gjs + 1) begin : g_jsel
            assign jsel[gjs] = (j == gjs[JW-1:0]);
        end
    endgenerate
    reg [31:0] inv_sel;
    integer qj;
    always @* begin
        inv_sel = 0;
        for (qj = 0; qj < H; qj = qj + 1) inv_sel = inv_sel | (inv_freq[qj*32 +: 32] & {32{jsel[qj]}});
    end
    // The turn is the fraction of a revolution, so only bits 31:16 of the
    // product are wanted and the upper half of the multiplier is not built.
    // It is a stage of its own: the multiply and then the table's own index,
    // read and interpolation in one cycle were two multiplies and a table.
    wire [31:0]  prod = pos * inv_sel;
    reg  [15:0]  turn;
    reg          v0, v1, v2;
    reg [JW-1:0] j0, j1, j2;
    wire [15:0]  s_w, c_w;
    fabric_lut #(.IB(10), .FB(6), .W(16), .FILE({LUT_DIR, "lut_sin.hex"})) u_sin (.clk(clk), .u(turn), .y(s_w));
    fabric_lut #(.IB(10), .FB(6), .W(16), .FILE({LUT_DIR, "lut_sin.hex"})) u_cos (.clk(clk), .u(turn + 16'h4000), .y(c_w));
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0; j <= 0; v0 <= 1'b0; v1 <= 1'b0; v2 <= 1'b0; done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (start) begin busy <= 1'b1; j <= 0; end
            else if (busy) begin
                j <= j + 1'b1;
                if (j == H - 1) busy <= 1'b0;
            end
            turn <= prod[31:16];
            v0 <= busy; j0 <= j;
            v1 <= v0;   j1 <= j0;
            v2 <= v1;   j2 <= j1;
            if (v2) begin
                sin_tab[j2*16 +: 16] <= s_w;
                cos_tab[j2*16 +: 16] <= c_w;
            end
            if (v2 && j2 == H - 1) done <= 1'b1;
        end
    end
endmodule

// ---------------------------------------------------------------------------
// Rotate the first R elements of an int16 head vector of HD elements as
// pairs (i, i + R/2), then requantize every element to int8.  The head
// streams in L per beat, is buffered, and streams out L per beat five
// cycles after the last input beat (one multiply to a stage).
//   y1 = sat16((x1 cos - x2 sin + 2^14) >> 15)
//   y2 = sat16((x2 cos + x1 sin + 2^14) >> 15)
//   out = sat8((y * mult + 2^(shift-1)) >> shift)
// ---------------------------------------------------------------------------
module fabric_rotary #(
    parameter int HD = 256,
    parameter int R  = 64,
    parameter int L  = 8
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                in_valid,
    input  wire [L*16-1:0]     in_x,
    input  wire [(R/2)*16-1:0] sin_tab,
    input  wire [(R/2)*16-1:0] cos_tab,
    input  wire [15:0]         mult,
    input  wire [5:0]          shift,
    output reg                 out_valid,
    output reg  [L*8-1:0]      out_y
);
    localparam int BEATS = HD / L;
    localparam int BW    = $clog2(BEATS) + 1;
    localparam int H     = R / 2;
    // The head is a memory of beats, not a register vector.  Read as a vector
    // it is one mux over HD by 16 bits per lane, and the drain address on one
    // flop at 593 loads was two of this unit's two and a half nanoseconds --
    // the shape the record reader and the attention core's accumulators had.
    // Only the first R elements are rotated, so only those need the partner
    // read that is not beat-aligned, and they stay in logic: R is a fraction
    // of HD and the mux that is left is that fraction of the one that was.
    localparam int RA = (BEATS > 1) ? $clog2(BEATS) : 1;
    reg [R*16-1:0]  rot_buf;                  // the rotated elements, for the partner
    reg [BW-1:0]    wr, rd, rd_addr;
    reg             draining, rd_valid;
    integer         lc;
    // A copy of the drain address per lane.  The address selects three
    // sixteen-bit windows of the buffer for every lane and an entry of each
    // table: 593 loads on one flop at two lanes, and 1,001 at the sixteen
    // the 9B head actually has, where it was 4.07 of this unit's 4.60 ns.
    // These were `(* keep *)` on a register array, which keeps the net and
    // lets the mapper merge the flops back into one -- so the copies were
    // never there.  fabric_const_copy keeps them, because each is its own
    // module.  Every copy takes the address's next value, so all of them
    // and `rd_addr` are the same register.
    wire [BW-1:0] rd_l [0:L-1];
    wire [BW-1:0] rd_l_next = draining ? rd : rd_addr;
    genvar grl;
    generate
        for (grl = 0; grl < L; grl = grl + 1) begin : g_rdl
            fabric_const_copy #(.W(BW)) u_rd (.clk(clk), .d(rd_l_next), .q(rd_l[grl]));
        end
    endgenerate
    // The address leads the data by a cycle, which is what `rd` already is:
    // the drain registers it into rd_addr, so the beat the memory is asked
    // for now is the beat rd_addr will name next cycle.
    wire [L*16-1:0] beat_q;
    fabric_sram #(.W(L*16), .D(BEATS), .NRD(1), .NWR(1), .MB(L*16)) u_buf (
        .clk(clk), .rd_en(1'b1), .rd_addr(rd[RA-1:0]), .rd_data(beat_q),
        .wr_en(in_valid), .wr_addr(wr[RA-1:0]), .wr_data(in_x), .wr_mask(1'b1));
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr <= 0; rd <= 0; draining <= 1'b0; rd_valid <= 1'b0;
        end else begin
            rd_valid <= 1'b0;
            if (in_valid) begin
                for (lc = 0; lc < L; lc = lc + 1)
                    if (wr * L + lc < R) rot_buf[(wr * L + lc)*16 +: 16] <= in_x[lc*16 +: 16];
                wr <= wr + 1'b1;
                if (wr == BEATS - 1) begin draining <= 1'b1; rd <= 0; wr <= 0; end
            end
            if (draining) begin
                rd_valid <= 1'b1;
                rd_addr <= rd;
                rd <= rd + 1'b1;
                if (rd == BEATS - 1) draining <= 1'b0;
            end
        end
    end
    // P1 picks the beat's pair and its sine and cosine, P2 multiplies, P3
    // combines and rounds, P4 and P5 requantize: one multiply, or one round
    // with its saturate, to a stage.  Both halves of a rotated pair read the
    // same two elements and the same table entry, so the half only chooses
    // which is multiplied by the cosine and whether the terms add.
    reg            v1, v2, v3, v4;
    reg [L*16-1:0] sa1, cs1, sn1, pt1;
    reg [L-1:0]    rot1, add1;
    reg [L-1:0]    rot2, add2, rot3;
    reg [L*16-1:0] pt2, pt3;
    reg [L*16-1:0] cs2, sn2;
    reg signed [31:0] pa2 [0:L-1];
    reg signed [31:0] pb2 [0:L-1];
    reg [L*16-1:0] y3;
    reg signed [31:0] pm4 [0:L-1];
    integer l, e, pp, pa;
    // The output requantizer's scale and shift are command constants on input
    // pins, and they drive every lane's multiplier and shifter: a copy each
    // per lane, as the norm, the state engine and the residual take.
    wire [15:0] mult_l [0:L-1];
    wire [5:0]  shift_l [0:L-1];
    genvar gq;
    generate
        for (gq = 0; gq < L; gq = gq + 1) begin : g_q
            fabric_const_copy #(.W(16)) u_m (.clk(clk), .d(mult),  .q(mult_l[gq]));
            fabric_const_copy #(.W(6))  u_s (.clk(clk), .d(shift), .q(shift_l[gq]));
        end
    endgenerate
    always @(posedge clk) begin
        v1 <= rd_valid;
        // The element the cosine multiplies is the lane's own element either
        // way -- for e < H it is buffer[pp] and for e >= H buffer[pp + H],
        // and both are buffer[e] -- so it is the beat itself, one window for
        // every lane rather than a window each, and the same value the
        // pass-through carries.  Only the partner is per lane.
        pt1 <= beat_q;
        for (l = 0; l < L; l = l + 1) begin
            e  = rd_l[l] * L + l;
            pp = (e < H) ? e : (e - H);
            pa = (e < H) ? (e + H) : (e - H);          // the element it is paired with
            rot1[l] <= (e < R);
            add1[l] <= (e >= H);                       // the second half adds its terms
            sa1[l*16 +: 16] <= (pa < R) ? rot_buf[pa*16 +: 16] : 16'd0;
            cs1[l*16 +: 16] <= cos_tab[pp*16 +: 16];
            sn1[l*16 +: 16] <= sin_tab[pp*16 +: 16];
        end
        // P2: the two products.
        v2 <= v1; rot2 <= rot1; add2 <= add1; pt2 <= pt1;
        for (l = 0; l < L; l = l + 1) begin
            pa2[l] <= $signed(pt1[l*16 +: 16]) * $signed(cs1[l*16 +: 16]);
            pb2[l] <= $signed(sa1[l*16 +: 16]) * $signed(sn1[l*16 +: 16]);
        end
        // P3: combine, round and saturate, or pass the element through.
        v3 <= v2; rot3 <= rot2; pt3 <= pt2;
        for (l = 0; l < L; l = l + 1)
            y3[l*16 +: 16] <= rot2[l] ? y3_n[l*16 +: 16] : pt2[l*16 +: 16];
        // P4 and P5: the output requantizer.
        v4 <= v3;
        for (l = 0; l < L; l = l + 1)
            pm4[l] <= $signed(y3[l*16 +: 16]) * $signed({16'b0, mult_l[l]});
        out_valid <= v4;
        out_y <= outy_n;
    end
    // Both rounds go through fabric_rnd_sat at the width the value has.  The
    // combination is 33 bits and the requantized product 32; taken through the
    // 64-bit helpers each one is a shifter, a carry-propagate add and a
    // compare of twice that width, and the output shift amount -- a register,
    // so the mapper cannot buffer it -- fans out to every mux of its level.
    // That was 327 of this unit's 2,017 ps before the arithmetic even started.
    wire signed [32:0] comb_n [0:L-1];
    wire [L*16-1:0]    y3_n;
    wire [L*8-1:0]     outy_n;
    genvar gr;
    generate
        for (gr = 0; gr < L; gr = gr + 1) begin : g_rs
            assign comb_n[gr] = add2[gr]
                ? ($signed({pa2[gr][31], pa2[gr]}) + $signed({pb2[gr][31], pb2[gr]}))
                : ($signed({pa2[gr][31], pa2[gr]}) - $signed({pb2[gr][31], pb2[gr]}));
            fabric_rnd_sat #(.W(33), .SW(6), .N(16)) u_y3 (
                .v(comb_n[gr]), .sh(6'd15), .y(y3_n[gr*16 +: 16]));
            fabric_rnd_sat #(.W(32), .SW(6), .N(8)) u_oy (
                .v(pm4[gr]), .sh(shift_l[gr]), .y(outy_n[gr*8 +: 8]));
        end
    endgenerate
endmodule

// ---------------------------------------------------------------------------
// Online-softmax attention of G queries over one stream of key and value
// rows of HD int8 elements, L per beat.
//
// Beats carry a kind: 0 query, 1 gate (G rows of HD/L beats each, head
// after head), 2 key, 3 value.  A key row's HD/L beats are followed by the
// value row's; the score is complete at the last key beat, its exponential
// takes a few cycles (in_ready drops), and the value beats then update the
// accumulators.  `finish` after the last value row starts the output: for
// each head the reciprocal of the softmax sum, then HD/L beats of
//   out = sat8((sat16((o * r) >> (7 + LW - lz)) * sigmoid(gate) * mult_o) >> sh_o)
//
// Per row and head:  s = (q . k) * mult_s >> sh_s (F16 units, unsaturated)
//   first row or s > m:  f = exp(-(s - m)) (1.0 on the first row), m = s, p = 1.0
//   otherwise:           f = 1.0, p = exp(-(m - s))
//   l = (l * f >> 16) + p,   o = (o * f >> 16) + p * v
// Golden model: fabric/layer.py attention_int.
// ---------------------------------------------------------------------------
module fabric_attention #(
    parameter int HD = 256,
    parameter int G  = 4,
    parameter int L  = 64,
    parameter int LW = 28,
    parameter     LUT_DIR = "./"
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           start,
    input  wire           in_valid,
    output wire           in_ready,
    input  wire [1:0]     in_kind,
    input  wire [L*8-1:0] in_data,
    input  wire           finish,
    input  wire [15:0]    mult_s,
    input  wire [5:0]     sh_s,
    input  wire [15:0]    mult_gate,
    input  wire [5:0]     sh_gate,
    input  wire [15:0]    mult_o,
    input  wire [5:0]     sh_o,
    output reg            out_valid,
    output reg  [L*8-1:0] out_data,
    output reg            done
);
    localparam int BEATS = HD / L;
    localparam int BW    = $clog2(BEATS) + 1;
    localparam int GW    = $clog2(G) + 1;
    localparam int OW    = 36;

    // A head's query, its gate and its running output are memories, one beat
    // to a word: BEATS words of L bytes for the first two and of L
    // accumulators for the third.  Their addresses are presented a cycle
    // ahead -- the beat a memory is asked for is the beat that will arrive,
    // not the one that has -- so the data lands where the arrays put it and
    // nothing downstream moves.
    localparam int QW = L * 8, OWW = L * OW;
    wire [QW-1:0]  q_rd [0:G-1];
    wire [QW-1:0]  gate_rd [0:G-1];
    wire [OWW-1:0] o_rd [0:G-1];
    wire [QW-1:0]  q_wd, gate_wd;
    reg  [G*OWW-1:0] o_wd;
    wire [G-1:0]   q_we, gate_we;
    reg            o_we;
    reg            o_seen;                // the first record has written every beat
    reg signed [31:0] m_r [0:G-1];
    reg               m_valid [0:G-1];
    reg [LW-1:0]      l_r [0:G-1];
    reg [15:0]        f_r [0:G-1];
    reg [15:0]        p_r [0:G-1];

    // Beat counters per kind.
    reg [BW-1:0] beat;
    reg [GW-1:0] head;
    // State.
    localparam [3:0] S_ACCEPT = 4'd0, S_EXP = 4'd1, S_APPLY = 4'd2, S_VALUE = 4'd3,
                     S_RECIP = 4'd4, S_OUT = 4'd5, S_DONE = 4'd6, S_EXP2 = 4'd7;
    reg [3:0] state;
    reg signed [47:0] sm [0:G-1];
    assign in_ready = (state == S_ACCEPT) || (state == S_VALUE);

    // Score contribution of a key beat for every head.  The products of two
    // int8s need 16 bits and their sum over the beat needs 16 + log2(L), so
    // the tree is narrow, and balanced: a chain of L adds is L carry chains
    // deep and ABC cannot restructure them.
    integer g, l;
    // The products are reduced carry-save, not by a tree of adds.  Balanced,
    // the tree is log2(L) adds deep, but each of those is a carry chain that
    // ABC cannot restructure: at sixteen lanes it was four ripples in series
    // and most of this core's path.  A carry-save layer is one gate deep.
    //
    // The accumulator is carry-save as well -- value = s + 2c -- and joins
    // the beat's products as two more operands of the same tree.  Resolved
    // every beat instead, the accumulate is the tree's own resolve and then a
    // 32-bit add, two carry propagations in series behind a macro read that
    // already costs 512 ps of the cycle.  Kept redundant, the accumulate is a
    // layer, and the one real add belongs to S_EXP, whose only other work is
    // the score's multiply.  The tree runs at the accumulator's width because
    // a carry-save pair cannot be sign-extended a word at a time: the sign is
    // a property of the value, and the value is not resolved.
    localparam int SCW = 32;
    reg  [SCW-1:0] score_s [0:G-1], score_c [0:G-1];
    wire [SCW-1:0] score_sn [0:G-1], score_cn [0:G-1];
    wire signed [SCW-1:0] score [0:G-1];
    genvar gc, gl;
    generate
        for (gc = 0; gc < G; gc = gc + 1) begin : g_contrib
            wire [(L+2)*SCW-1:0] cops;
            for (gl = 0; gl < L; gl = gl + 1) begin : g_cp
                wire signed [15:0] pr = $signed(q_rd[gc][gl*8 +: 8]) * $signed(in_data[gl*8 +: 8]);
                assign cops[gl*SCW +: SCW] = {{(SCW-16){pr[15]}}, pr};
            end
            assign cops[L*SCW +: SCW]     = score_s[gc];
            assign cops[(L+1)*SCW +: SCW] = {score_c[gc][SCW-2:0], 1'b0};
            fabric_csa_tree #(.N(L+2), .W(SCW)) u_ct (.ops(cops), .s(score_sn[gc]), .c(score_cn[gc]));
            assign score[gc] = $signed(score_s[gc]) + $signed({score_c[gc][SCW-2:0], 1'b0});
        end
    endgenerate

    // Exponentials: one fabric_exp_neg per head, fed in S_EXP.
    reg  [21:0] d_in [0:G-1];
    reg         newmax [0:G-1];
    reg         exp_go;
    wire [G-1:0] exp_v;
    wire [15:0]  exp_y [0:G-1];
    genvar gg;
    generate
        for (gg = 0; gg < G; gg = gg + 1) begin : g_exp
            fabric_exp_neg #(.LUT_DIR(LUT_DIR)) u_exp (.clk(clk), .valid_in(exp_go), .t(d_in[gg]), .valid_out(exp_v[gg]), .y(exp_y[gg]));
        end
    endgenerate

    // The softmax rescale factors, one per head, each multiplying every one
    // of that head's L lanes.  Shared, `f_r[0]` alone carried 842 loads and
    // 1.38 pF at the 9B geometry -- 3.39 of the core's 5.24 ns -- because a
    // sixteen-bit operand into L multipliers is L times sixteen fanouts.  A
    // copy per head and lane, taking the same next value as the register, so
    // the copy is that register.
    wire fp_take = (state == S_APPLY) && exp_v[0];
    wire [15:0] f_r_c [0:G*L-1];
    wire [15:0] p_r_c [0:G*L-1];
    genvar gfp, lfp;
    generate
        for (gfp = 0; gfp < G; gfp = gfp + 1) begin : g_fp
            wire [15:0] f_nx = fp_take ? (newmax[gfp] ? (m_valid[gfp] ? exp_y[gfp] : 16'hFFFF) : 16'hFFFF) : f_r[gfp];
            wire [15:0] p_nx = fp_take ? (newmax[gfp] ? 16'hFFFF : exp_y[gfp]) : p_r[gfp];
            for (lfp = 0; lfp < L; lfp = lfp + 1) begin : g_fpl
                fabric_const_copy #(.W(16)) u_f (.clk(clk), .d(f_nx), .q(f_r_c[gfp*L + lfp]));
                fabric_const_copy #(.W(16)) u_p (.clk(clk), .d(p_nx), .q(p_r_c[gfp*L + lfp]));
            end
        end
    endgenerate

    // Reciprocal for the output.
    reg          rc_start;
    wire         rc_done;
    wire [16:0]  rc_r;
    wire [5:0]   rc_lz;
    reg [LW-1:0] rc_l;
    fabric_recip #(.LW(LW), .LUT_DIR(LUT_DIR)) u_rc (.clk(clk), .start(rc_start), .l(rc_l), .done(rc_done), .r(rc_r), .lz_out(rc_lz));
    reg [16:0] r_hold;
    reg [5:0]  lz_hold;
    // The output round's shift amount reaches every lane's barrel shifter, so
    // one flop held 122 loads and 207 fF: 575 ps of clock-to-output, a
    // quarter of this core's path.  A copy per lane, which is what
    // fabric_const_copy is for.  Each takes the same next value as `lz_hold`,
    // so they are that register, not a cycle behind it.
    wire [5:0] lz_next = (state == S_RECIP && rc_done) ? rc_lz : lz_hold;
    // `r_hold` is the same story on the output product: one seventeen-bit
    // operand into every lane's multiplier.
    wire [16:0] r_next = (state == S_RECIP && rc_done) ? rc_r : r_hold;
    wire [16:0] r_c [0:L-1];
    genvar grc;
    generate
        for (grc = 0; grc < L; grc = grc + 1) begin : g_rc
            fabric_const_copy #(.W(17)) u_r (.clk(clk), .d(r_next), .q(r_c[grc]));
        end
    endgenerate
    wire [5:0] lz_c [0:L-1];
    genvar glz;
    generate
        for (glz = 0; glz < L; glz = glz + 1) begin : g_lz
            fabric_const_copy #(.W(6)) u_lz (.clk(clk), .d(lz_next), .q(lz_c[glz]));
        end
    endgenerate

    // Output pipeline: O1 the two products, O2 their round and saturate,
    // O3..O5 the sigmoid, then the gate's product, the scale's and the
    // output's round.  One multiply, or one round with its saturate, to a
    // stage.
    reg                ov1, ov2;
    reg signed [OW+17:0] wm1 [0:L-1];
    reg signed [23:0]  gm1 [0:L-1];
    reg [L*16-1:0]     w2;
    reg [L*16-1:0]     tg2;
    wire [L-1:0]       sgv;
    wire [L*16-1:0]    sg5;
    generate
        for (gg = 0; gg < L; gg = gg + 1) begin : g_sig
            fabric_sigmoid #(.LUT_DIR(LUT_DIR)) u_sg (.clk(clk), .valid_in(ov2), .t(tg2[gg*16 +: 16]), .valid_out(sgv[gg]), .y(sg5[gg*16 +: 16]));
        end
    endgenerate
    reg [L*16-1:0] w3, w4, w5;
    always @(posedge clk) begin w3 <= w2; w4 <= w3; w5 <= w4; end

    reg signed [47:0] sc;
    reg signed [49:0] dd;
    reg signed [63:0] ow, tmp;
    // The score's round, out of the always block so it is one module
    // at the value's width rather than the helpers' 64.
    wire signed [47:0] sc_w [0:G-1];
    genvar gsc;
    generate
        for (gsc = 0; gsc < G; gsc = gsc + 1) begin : g_sc
            fabric_rnd #(.W(48), .SW(6)) u_sc (.v(sm[gsc]), .sh(sh_s), .y(sc_w[gsc]));
        end
    endgenerate
    // The value update's two products, applied the cycle after they are formed.
    reg signed [OW+17:0] va [0:G-1][0:L-1];
    reg signed [24:0]    vb [0:G-1][0:L-1];
    reg                  vv;
    reg [BW-1:0]         vbeat;
    reg [GW-1:0] ohead;
    reg [BW-1:0] obeat;
    reg [BW-1:0] o_waddr;
    reg [3:0]    drain;
    reg          out_go;
    // `o_seen` chooses, for every lane of every head, whether the value
    // memory is read or zero, so one flop held G*L multiplexers: 780 loads
    // and 1.46 pF at the 9B geometry, 3.57 of this core's 5.84 ns.  A copy
    // per lane of each head for the value update and per lane for the output
    // product.  `start` clears the copies through their own next value, so
    // they match the register that has the reset.
    wire o_seen_next = (!rst_n || start) ? 1'b0
                     : ((vv && vbeat == BEATS - 1) ? 1'b1 : o_seen);
    wire o_seen_v [0:G*L-1];                 // the value update, per head and lane
    wire o_seen_o [0:L-1];                   // the output product, per lane
    genvar gos, los;
    generate
        for (gos = 0; gos < G; gos = gos + 1) begin : g_osv
            for (los = 0; los < L; los = los + 1) begin : g_osl
                fabric_const_copy #(.W(1)) u_os (.clk(clk), .d(o_seen_next), .q(o_seen_v[gos*L + los]));
            end
        end
        for (los = 0; los < L; los = los + 1) begin : g_oso
            fabric_const_copy #(.W(1)) u_os (.clk(clk), .d(o_seen_next), .q(o_seen_o[los]));
        end
    endgenerate

    // The beat each memory is asked for: the one that will arrive, not the one
    // that has.  A memory answers a cycle later, so an address that tracked
    // `beat` would always be a beat behind.
    wire         beat_last = (beat == BEATS - 1);
    wire [BW-1:0] bnext = (start || (state == S_APPLY && exp_v[0])) ? {BW{1'b0}}
                        : (in_valid && in_ready) ? (beat_last ? {BW{1'b0}} : beat + 1'b1)
                        : beat;
    wire [BW-1:0] onext = (state == S_RECIP) ? {BW{1'b0}}
                        : (obeat == BEATS - 1) ? {BW{1'b0}} : obeat + 1'b1;
    wire          outing = (state == S_OUT) || (state == S_RECIP);
    wire [BW-1:0] oaddr  = outing ? onext : bnext;
    assign q_wd    = in_data;
    assign gate_wd = in_data;
    genvar gm;
    generate
        for (gm = 0; gm < G; gm = gm + 1) begin : g_mem
            assign q_we[gm]    = in_valid && in_ready && (state == S_ACCEPT) && (in_kind == 2'd0) && (head == gm);
            assign gate_we[gm] = in_valid && in_ready && (state == S_ACCEPT) && (in_kind == 2'd1) && (head == gm);
            fabric_sram #(.W(QW), .D(BEATS), .MB(QW)) u_q (
                .clk(clk), .rd_en(1'b1), .rd_addr(bnext[$clog2(BEATS)-1:0]), .rd_data(q_rd[gm]),
                .wr_en(q_we[gm]), .wr_addr(beat[$clog2(BEATS)-1:0]), .wr_data(q_wd), .wr_mask(1'b1));
            fabric_sram #(.W(QW), .D(BEATS), .MB(QW)) u_gate (
                .clk(clk), .rd_en(1'b1), .rd_addr(onext[$clog2(BEATS)-1:0]), .rd_data(gate_rd[gm]),
                .wr_en(gate_we[gm]), .wr_addr(beat[$clog2(BEATS)-1:0]), .wr_data(gate_wd), .wr_mask(1'b1));
            fabric_sram #(.W(OWW), .D(BEATS), .MB(OWW)) u_o (
                .clk(clk), .rd_en(1'b1), .rd_addr(oaddr[$clog2(BEATS)-1:0]), .rd_data(o_rd[gm]),
                .wr_en(o_we), .wr_addr(o_waddr[$clog2(BEATS)-1:0]), .wr_data(o_wd[gm*OWW +: OWW]), .wr_mask(1'b1));
        end
    endgenerate
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_ACCEPT; beat <= 0; head <= 0; exp_go <= 1'b0; rc_start <= 1'b0; done <= 1'b0;
            ov1 <= 1'b0; out_go <= 1'b0; ohead <= 0; obeat <= 0; drain <= 0; vv <= 1'b0; o_we <= 1'b0; o_seen <= 1'b0;
            for (g = 0; g < G; g = g + 1) begin m_valid[g] <= 1'b0; l_r[g] <= 0; score_s[g] <= 0; score_c[g] <= 0; end
        end else begin
            exp_go <= 1'b0;
            rc_start <= 1'b0;
            done <= 1'b0;
            ov1 <= 1'b0;
            // The value update's second half, before the state machine below
            // forms the next beat's products.
            o_we <= 1'b0;
            if (vv) begin
                for (g = 0; g < G; g = g + 1)
                    for (l = 0; l < L; l = l + 1) begin
                        ow = fx_rnd_shr(va[g][l], 16) + $signed({{39{vb[g][l][24]}}, vb[g][l]});
                        o_wd[(g*L + l)*OW +: OW] <= ow[OW-1:0];
                    end
                o_we <= 1'b1;
                o_waddr <= vbeat;
                if (vbeat == BEATS - 1) o_seen <= 1'b1;   // every beat written: the memory may be read
            end
            vv <= 1'b0;
            if (start) begin
                state <= S_ACCEPT; beat <= 0; head <= 0; o_seen <= 1'b0;
                for (g = 0; g < G; g = g + 1) begin
                    m_valid[g] <= 1'b0; l_r[g] <= 0; score_s[g] <= 0; score_c[g] <= 0;
                end
            end
            case (state)
                S_ACCEPT: begin
                    if (finish) begin
                        state <= S_RECIP; ohead <= 0; rc_l <= l_r[0]; rc_start <= 1'b1;
                    end else if (in_valid) begin
                        case (in_kind)
                            2'd0, 2'd1: ;                     // the memories take them; see q_we and gate_we
                            2'd2: for (g = 0; g < G; g = g + 1) begin score_s[g] <= score_sn[g]; score_c[g] <= score_cn[g]; end
                            default: ;
                        endcase
                        if (beat == BEATS - 1) begin
                            beat <= 0;
                            if (in_kind == 2'd0 || in_kind == 2'd1) head <= (head == G - 1) ? 0 : head + 1'b1;
                            if (in_kind == 2'd2) state <= S_EXP;
                        end else begin
                            beat <= beat + 1'b1;
                        end
                    end
                end
                S_EXP: begin
                    // The score is complete: its scale is a stage of its own.
                    for (g = 0; g < G; g = g + 1)
                        sm[g] <= $signed({{16{score[g][31]}}, score[g]}) * $signed({32'b0, mult_s});
                    state <= S_EXP2;
                end
                S_EXP2: begin
                    // Round it, compare with the maximum, launch the exponentials.
                    for (g = 0; g < G; g = g + 1) begin
                        // `sc` comes off fabric_rnd at the 48 bits `sm` has,
                        // so the shift, the compare and the difference are 48
                        // and 50 wide rather than 64: this was the core's
                        // second path, and every one of those is a carry
                        // chain the width of the value.
                        sc = sc_w[g];
                        if (!m_valid[g] || sc > $signed({{18{m_r[g][31]}}, m_r[g]})) begin
                            newmax[g] <= 1'b1;
                            dd = m_valid[g] ? ($signed({{2{sc[47]}}, sc}) - $signed({{20{m_r[g][31]}}, m_r[g]})) : 50'sd0;
                            m_r[g] <= sc[31:0];
                        end else begin
                            newmax[g] <= 1'b0;
                            dd = $signed({{20{m_r[g][31]}}, m_r[g]}) - $signed({{2{sc[47]}}, sc});
                        end
                        d_in[g] <= (dd > 50'sd4194303) ? 22'd4194303 : dd[21:0];
                        score_s[g] <= 0; score_c[g] <= 0;
                    end
                    exp_go <= 1'b1;
                    state <= S_APPLY;
                end
                S_APPLY: begin
                    // Wait for the exponentials, then set f and p and update l.
                    if (exp_v[0]) begin
                        for (g = 0; g < G; g = g + 1) begin
                            if (newmax[g]) begin
                                f_r[g] <= m_valid[g] ? exp_y[g] : 16'hFFFF;
                                p_r[g] <= 16'hFFFF;
                                tmp = fx_rnd_shr($signed({{(64-LW){1'b0}}, l_r[g]}) * $signed({48'b0, (m_valid[g] ? exp_y[g] : 16'hFFFF)}), 16) + 64'sd65535;
                            end else begin
                                f_r[g] <= 16'hFFFF;
                                p_r[g] <= exp_y[g];
                                tmp = fx_rnd_shr($signed({{(64-LW){1'b0}}, l_r[g]}) * 64'sd65535, 16) + $signed({48'b0, exp_y[g]});
                            end
                            l_r[g] <= tmp[LW-1:0];
                            m_valid[g] <= 1'b1;
                        end
                        state <= S_VALUE;
                        beat <= 0;
                    end
                end
                S_VALUE: begin
                    if (in_valid) begin
                        // The two products here, their round and add a cycle
                        // later: consecutive beats touch different elements,
                        // so the read and the write never meet.
                        for (g = 0; g < G; g = g + 1)
                            for (l = 0; l < L; l = l + 1) begin
                                va[g][l] <= $signed(o_seen_v[g*L + l] ? o_rd[g][l*OW +: OW] : {OW{1'b0}}) * $signed({{(OW+1){1'b0}}, f_r_c[g*L + l]});
                                vb[g][l] <= $signed({9'b0, p_r_c[g*L + l]}) * $signed(in_data[l*8 +: 8]);
                            end
                        vv    <= 1'b1;
                        vbeat <= beat;
                        if (beat == BEATS - 1) begin
                            beat <= 0;
                            state <= S_ACCEPT;
                        end else begin
                            beat <= beat + 1'b1;
                        end
                    end
                end
                S_RECIP: begin
                    if (rc_done) begin
                        r_hold <= rc_r; lz_hold <= rc_lz;
                        state <= S_OUT; obeat <= 0;
                    end
                end
                S_OUT: begin
                    // One beat per cycle into the output pipeline.
                    ov1 <= 1'b1;
                    for (l = 0; l < L; l = l + 1) begin
                        wm1[l] <= $signed(o_seen_o[l] ? o_rd[ohead][l*OW +: OW] : {OW{1'b0}}) * $signed({{(OW+1){1'b0}}, r_c[l]});
                        gm1[l] <= $signed(gate_rd[ohead][l*8 +: 8]) * $signed({8'b0, mult_gate});
                    end
                    if (obeat == BEATS - 1) begin
                        if (ohead == G - 1) begin state <= S_DONE; drain <= 0; end
                        else begin
                            ohead <= ohead + 1'b1;
                            rc_l <= l_r[ohead + 1];
                            rc_start <= 1'b1;
                            state <= S_RECIP;
                        end
                    end else begin
                        obeat <= obeat + 1'b1;
                    end
                end
                S_DONE: begin
                    // Let the output pipeline drain before done: O1, O2, the
                    // sigmoid's three and the three after it.  Its own
                    // counter, since obeat only spans a head's beats.
                    drain <= drain + 1'b1;
                    if (drain == 4'd11) begin done <= 1'b1; state <= S_ACCEPT; obeat <= 0; drain <= 0; end
                end
                default: state <= S_ACCEPT;
            endcase
        end
    end
    // O2: the products' round and saturate.  O6 and O7: the gate's product
    // and the output scale's, O8 the output's round.
    reg               o6v, o7v;
    reg signed [33:0] og6 [0:L-1];
    reg signed [55:0] oq7 [0:L-1];
    integer ol;
    always @(posedge clk) begin
        ov2 <= ov1;
        w2  <= w2_n;
        tg2 <= tg2_n;
        o6v <= sgv[0];
        for (ol = 0; ol < L; ol = ol + 1)
            og6[ol] <= $signed({{18{w5[ol*16+15]}}, w5[ol*16 +: 16]}) * $signed({18'b0, sg5[ol*16 +: 16]});
        o7v <= o6v;
        for (ol = 0; ol < L; ol = ol + 1)
            oq7[ol] <= $signed({{22{og6[ol][33]}}, og6[ol]}) * $signed({40'b0, mult_o});
        out_valid <= o7v;
        out_data  <= od_n;
    end
    // The three rounds at the width their value has: 54, 24 and 56 bits, not
    // the 64 the helpers in fabric_fx.svh evaluate at.  Each one saved is a
    // shifter, a carry-propagate add and a compare of the difference, and the
    // weight's shift amount -- which the leading-zero count sets, so it is a
    // register -- fans out to every mux of its level.
    wire [5:0]      sh_w [0:L-1];
    wire [L*16-1:0] w2_n, tg2_n;
    wire [L*8-1:0]  od_n;
    genvar go;
    generate
        for (go = 0; go < L; go = go + 1) begin : g_oq
            assign sh_w[go] = (7 + LW) - lz_c[go];      // in [7, 7+LW]: lz_c counts at most LW
            fabric_rnd_sat #(.W(OW+18), .SW(6), .N(16)) u_w2 (
                .v(wm1[go]), .sh(sh_w[go]), .y(w2_n[go*16 +: 16]));
            fabric_rnd_sat #(.W(24), .SW(6), .N(16)) u_tg (
                .v(gm1[go]), .sh(sh_gate), .y(tg2_n[go*16 +: 16]));
            fabric_rnd_sat #(.W(56), .SW(6), .N(8)) u_od (
                .v(oq7[go]), .sh(sh_o), .y(od_n[go*8 +: 8]));
        end
    endgenerate
endmodule

`default_nettype wire
