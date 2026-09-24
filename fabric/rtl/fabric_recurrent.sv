// The Gated DeltaNet units: the causal convolution with its SiLU, the
// per-head gates, and the delta-rule state update.
// Golden model: fabric/layer.py (conv_silu_int, head_gates_int, delta_state_int).

`default_nettype none
`include "fabric_fx.svh"

// ---------------------------------------------------------------------------
// Depthwise causal convolution of K taps and SiLU, L channels per beat.
// The beat carries the channel's new int8 sample, its K-1 previous samples
// (oldest first), its int8 taps and its constants; it returns the int8
// output and the shifted history.  Latency 9.
// ---------------------------------------------------------------------------
module fabric_conv_silu #(
    parameter int K = 4,
    parameter int L = 8,
    parameter     LUT_DIR = "./"
) (
    input  wire                 clk,
    input  wire                 in_valid,
    input  wire [L*8-1:0]       in_x,
    input  wire [L*(K-1)*8-1:0] in_hist,
    input  wire [L*K*8-1:0]     in_w,
    input  wire [L*16-1:0]      mult_in,
    input  wire [L*6-1:0]       sh_in,
    input  wire [L*16-1:0]      mult_out,
    input  wire [L*6-1:0]       sh_out,
    output reg                  out_valid,
    output reg  [L*8-1:0]       out_y,
    output reg  [L*(K-1)*8-1:0] out_hist
);
    localparam int HW = (K - 1) * 8;
    // The taps are K products of two int8s, which is 16 bits and K of them
    // 16 + log2(K), not the 64 the helpers compute at: written wide they were
    // a chain of 64-bit adds, the mistake the index scan and the append had.
    localparam int AW_  = 16 + ((K > 1) ? $clog2(K) : 1);
    localparam int PW1_ = AW_ + 17;            // the F16 requantize's product: the taps by a 16-bit mult
    localparam int PW2_ = 16 + 17;             // the int8 one's: an int16 by a 16-bit mult
    // S1: accumulate the taps, shift the history.
    reg                 v1;
    reg signed [AW_-1:0] acc1 [0:L-1];
    reg [L*HW-1:0]      hist1;
    reg [L*16-1:0]      mi1, mo1;
    reg [L*6-1:0]       si1, so1;
    integer c, j;
    // The taps are summed carry-save.  Written as `acc = acc + tap` over the
    // window it is one carry-propagate add per tap in series -- the shape the
    // attention core's contribution had and the sequencer's counters -- and
    // it measured 1,569 ps of this unit's 1,877.  A layer is one gate and the
    // pair resolves once at the end.
    wire signed [AW_-1:0] acc_n [0:L-1];
    genvar gt, gj;
    generate
        for (gt = 0; gt < L; gt = gt + 1) begin : g_tap
            wire [K*AW_-1:0] taps;
            for (gj = 0; gj < K - 1; gj = gj + 1) begin : g_h
                assign taps[gj*AW_ +: AW_] =
                    $signed(in_hist[gt*HW + gj*8 +: 8]) * $signed(in_w[gt*K*8 + gj*8 +: 8]);
            end
            assign taps[(K-1)*AW_ +: AW_] =
                $signed(in_x[gt*8 +: 8]) * $signed(in_w[gt*K*8 + (K-1)*8 +: 8]);
            wire [AW_-1:0] ts, tc;
            fabric_csa_tree #(.N(K), .W(AW_)) u_tt (.ops(taps), .s(ts), .c(tc));
            assign acc_n[gt] = $signed(ts) + $signed({tc[AW_-2:0], 1'b0});
        end
    endgenerate
    always @(posedge clk) begin
        v1 <= in_valid;
        for (c = 0; c < L; c = c + 1) begin
            acc1[c] <= acc_n[c];
            if (K > 2) hist1[c*HW +: HW] <= {in_x[c*8 +: 8], in_hist[c*HW + 8 +: HW-8]};
            else       hist1[c*HW +: HW] <= in_x[c*8 +: 8];
        end
        mi1 <= mult_in; si1 <= sh_in; mo1 <= mult_out; so1 <= sh_out;
    end
    // S2: the F16 requantize's multiply, carry-save.  S2B: its resolve and
    // the shift.  S3: the round and saturate.
    //
    // Three stages where there were two.  A multiply written `a * b` is a
    // partial-product tree and a final add the width of the product, and a
    // requantize is a barrel shifter, a round and a saturate; one of each to
    // a stage put a carry propagation either side of the shifter and left
    // this unit's two worst paths at 1,877 and 1,856 ps.  Split, the multiply
    // is a tree, the resolve shares a stage with the shifter it feeds, and
    // the round is what it always was.  The convolution is a feed-forward
    // pipeline, so the two stages are latency and not throughput --
    // `conv_latency` in fabric/sequencer.py, over a command of a thousand
    // beats at 9B.
    reg                  v2, v2b, v3;
    reg [PW1_-1:0]       q2s [0:L-1], q2c [0:L-1];
    reg signed [PW1_:0]  sv2 [0:L-1];
    reg [L-1:0]          rb2;
    reg [L*16-1:0]       t3;
    reg [L*HW-1:0]       hist2, hist2b, hist3;
    reg [L*16-1:0]       mo2b, mo2c, mo3b;
    reg [L*6-1:0]        so2b, so2c, so3b, si2;
    always @(posedge clk) begin
        v2 <= v1;
        for (c = 0; c < L; c = c + 1) begin q2s[c] <= q2s_n[c]; q2c[c] <= q2c_n[c]; end
        si2 <= si1;
        hist2 <= hist1; mo2b <= mo1; so2b <= so1;
        v2b <= v2;
        for (c = 0; c < L; c = c + 1) begin sv2[c] <= sv2_n[c]; rb2[c] <= rb2_n[c]; end
        hist2b <= hist2; mo2c <= mo2b; so2c <= so2b;
        v3 <= v2b;
        t3 <= t3_n;
        hist3 <= hist2b; mo3b <= mo2c; so3b <= so2c;
    end
    // The products are PW1_ and PW2_ wide, not the 64 the helpers in
    // fabric_fx.svh evaluate at: a shifter built at 64 costs twice the delay
    // and twice the fanout on the shift amount, which is a register and so
    // the one net the mapper cannot buffer (see fabric_vector.sv).
    wire [PW1_-1:0]      q2s_n [0:L-1], q2c_n [0:L-1];
    wire signed [PW1_:0] sv2_n [0:L-1];
    wire [L-1:0]         rb2_n;
    wire [PW2_-1:0]      q8s_n [0:L-1], q8c_n [0:L-1];
    wire signed [PW2_:0] sv8_n [0:L-1];
    wire [L-1:0]         rb8_n;
    wire [L*16-1:0]      t3_n;
    wire [L*8-1:0]       y9_n;
    genvar gr;
    generate
        for (gr = 0; gr < L; gr = gr + 1) begin : g_rq
            fabric_mul_cs #(.AW(AW_), .BW(16), .PW(PW1_), .ADD(1)) u_m2 (
                .a(acc1[gr]), .b(mi1[gr*16 +: 16]), .addend({PW1_{1'b0}}),
                .s(q2s_n[gr]), .c(q2c_n[gr]));
            wire signed [PW1_-1:0] q2q =
                $signed(q2s[gr]) + $signed({q2c[gr][PW1_-2:0], 1'b0});
            fabric_rnd_sat_shift #(.W(PW1_), .SW(6)) u_s2 (
                .v(q2q), .sh(si2[gr*6 +: 6]), .sv(sv2_n[gr]), .rb(rb2_n[gr]));
            fabric_rnd_sat_round #(.W(PW1_), .N(16)) u_t3 (
                .sv(sv2[gr]), .rb(rb2[gr]), .y(t3_n[gr*16 +: 16]));
            fabric_mul_cs #(.AW(16), .BW(16), .PW(PW2_), .ADD(1)) u_m8 (
                .a(s6[gr*16 +: 16]), .b(mo7[gr*16 +: 16]), .addend({PW2_{1'b0}}),
                .s(q8s_n[gr]), .c(q8c_n[gr]));
            wire signed [PW2_-1:0] q8q =
                $signed(q8s[gr]) + $signed({q8c[gr][PW2_-2:0], 1'b0});
            fabric_rnd_sat_shift #(.W(PW2_), .SW(6)) u_s8 (
                .v(q8q), .sh(so8[gr*6 +: 6]), .sv(sv8_n[gr]), .rb(rb8_n[gr]));
            fabric_rnd_sat_round #(.W(PW2_), .N(8)) u_y9 (
                .sv(sv8[gr]), .rb(rb8[gr]), .y(y9_n[gr*8 +: 8]));
        end
    endgenerate
    // S4..S7: SiLU per lane; the history and constants wait four cycles.
    wire [L-1:0]    sv;
    wire [L*16-1:0] s6;
    genvar g;
    generate
        for (g = 0; g < L; g = g + 1) begin : g_silu
            fabric_silu #(.LUT_DIR(LUT_DIR)) u_silu (.clk(clk), .valid_in(v3), .t(t3[g*16 +: 16]), .valid_out(sv[g]), .y(s6[g*16 +: 16]));
        end
    endgenerate
    reg [L*HW-1:0] hist4, hist5, hist6, hist7;
    reg [L*16-1:0] mo4, mo5, mo6, mo7;
    reg [L*6-1:0]  so4, so5, so6, so7;
    always @(posedge clk) begin
        hist4 <= hist3; hist5 <= hist4; hist6 <= hist5; hist7 <= hist6;
        mo4 <= mo3b; mo5 <= mo4; mo6 <= mo5; mo7 <= mo6;
        so4 <= so3b; so5 <= so4; so6 <= so5; so7 <= so6;
    end
    // S8: the int8 requantize's multiply, carry-save.  S8B: its resolve and
    // the shift.  S9: the round and saturate.  The same three as above.
    reg                  v8, v8b;
    reg [PW2_-1:0]       q8s [0:L-1], q8c [0:L-1];
    reg signed [PW2_:0]  sv8 [0:L-1];
    reg [L-1:0]          rb8;
    reg [L*HW-1:0]       hist8, hist8b;
    reg [L*6-1:0]        so8;
    always @(posedge clk) begin
        v8 <= sv[0];
        for (c = 0; c < L; c = c + 1) begin q8s[c] <= q8s_n[c]; q8c[c] <= q8c_n[c]; end
        hist8 <= hist7; so8 <= so7;
        v8b <= v8;
        for (c = 0; c < L; c = c + 1) begin sv8[c] <= sv8_n[c]; rb8[c] <= rb8_n[c]; end
        hist8b <= hist8;
        out_valid <= v8b;
        out_hist  <= hist8b;
        out_y     <= y9_n;
    end
endmodule

// ---------------------------------------------------------------------------
// Per-head gates from the fabric's raw accumulators of in_proj_a and in_proj_b:
//   beta  = sigmoid(b * mult_b >> sh_b)
//   decay = exp(-A * softplus(a * mult_a >> sh_a + dt_bias))
// One head per beat, A in Q6.10, dt_bias in F16, both results U16.  Latency 10.
// ---------------------------------------------------------------------------
module fabric_head_gates #(
    parameter int ACC = 24,
    parameter     LUT_DIR = "./"
) (
    input  wire                  clk,
    input  wire                  in_valid,
    input  wire signed [ACC-1:0] a_acc,
    input  wire signed [ACC-1:0] b_acc,
    input  wire [15:0]           mult_a,
    input  wire [5:0]            sh_a,
    input  wire [15:0]           mult_b,
    input  wire [5:0]            sh_b,
    input  wire [15:0]           a_coef,
    input  wire signed [15:0]    dt_bias,
    output wire                  out_valid,
    output wire [15:0]           decay,
    output wire [15:0]           beta
);
    // S1: both products, carry-save, with each one's rounding constant folded
    // into the same tree.  Both accumulators arrive on input pins straight
    // from the fabric, and a 24 by 16 multiply with its round, its shift and
    // its saturate between two flops was the whole of this unit's 2.10 ns:
    // nine XORs of the product's carry chain and the shifter behind them.
    // The resolve belongs to S2 and the saturates to S3.  The shift must be
    // narrower than PW, which every compiled constant is.
    localparam int PW = ACC + 17;                     // ACC by 16 bits, exactly
    wire [PW-1:0] one_pw = {{(PW-1){1'b0}}, 1'b1};
    wire [PW-1:0] rnd_a  = (sh_a == 0) ? {PW{1'b0}} : (one_pw << (sh_a - 1'b1));
    wire [PW-1:0] rnd_b  = (sh_b == 0) ? {PW{1'b0}} : (one_pw << (sh_b - 1'b1));
    wire [PW-1:0] as_w, ac_w, bs_w, bc_w;
    fabric_mul_cs #(.AW(ACC), .BW(16), .PW(PW)) u_ma (.a(a_acc), .b(mult_a), .addend(rnd_a), .s(as_w), .c(ac_w));
    fabric_mul_cs #(.AW(ACC), .BW(16), .PW(PW)) u_mb (.a(b_acc), .b(mult_b), .addend(rnd_b), .s(bs_w), .c(bc_w));

    reg               v1;
    reg [PW-1:0]      as1, acy1, bs1, bcy1;
    reg [5:0]         sha1, shb1;
    reg signed [15:0] dt1;
    reg [15:0]        ac1;
    always @(posedge clk) begin
        v1   <= in_valid;
        as1  <= as_w;  acy1 <= ac_w;
        bs1  <= bs_w;  bcy1 <= bc_w;
        sha1 <= sh_a;  shb1 <= sh_b;
        dt1  <= dt_bias;
        ac1  <= a_coef;
    end
    // S2: resolve each pair and shift.
    reg                 v1b;
    reg signed [PW-1:0] pa1, pb1;
    reg signed [15:0]   dt1b;
    reg [15:0]          ac1b;
    always @(posedge clk) begin
        v1b  <= v1;
        pa1  <= ($signed(as1) + $signed({acy1[PW-2:0], 1'b0})) >>> sha1;
        pb1  <= ($signed(bs1) + $signed({bcy1[PW-2:0], 1'b0})) >>> shb1;
        dt1b <= dt1;
        ac1b <= ac1;
    end
    // S3: saturate, then dt_bias -- the requantize's saturate and the bias's
    // are both kept, so the arithmetic is the model's.
    reg               v2;
    reg signed [15:0] ta2, tb2;
    reg [15:0]        ac2;
    always @(posedge clk) begin
        v2  <= v1b;
        ta2 <= fx_sat(fx_sat($signed({{(64-PW){pa1[PW-1]}}, pa1}), 16)
                      + $signed({{48{dt1b[15]}}, dt1b}), 16);
        tb2 <= fx_sat($signed({{(64-PW){pb1[PW-1]}}, pb1}), 16);
        ac2 <= ac1b;
    end
    // S4..S6: softplus and sigmoid side by side.
    wire        spv, sgv;
    wire [15:0] sp5, sg5;
    fabric_softplus #(.LUT_DIR(LUT_DIR)) u_sp (.clk(clk), .valid_in(v2), .t(ta2), .valid_out(spv), .y(sp5));
    fabric_sigmoid  #(.LUT_DIR(LUT_DIR)) u_sg (.clk(clk), .valid_in(v2), .t(tb2), .valid_out(sgv), .y(sg5));
    reg [15:0] ac3, ac4, ac5;
    always @(posedge clk) begin ac3 <= ac2; ac4 <= ac3; ac5 <= ac4; end
    // S6: t = A * softplus in F16 (22 bits).
    reg        v6;
    reg [21:0] t6;
    // Sixteen by sixteen is 32 bits and 22 are kept; the shift is by a
    // constant, so the round is an increment on the bits that survive it.
    wire [31:0] acsp = ac5 * sp5;
    wire [21:0] t6_n = acsp[31:10] + {21'b0, acsp[9]};
    always @(posedge clk) begin
        v6 <= spv;
        t6 <= t6_n;
    end
    // S7..S9: exp; beta waits four cycles.
    fabric_exp_neg #(.LUT_DIR(LUT_DIR)) u_exp (.clk(clk), .valid_in(v6), .t(t6), .valid_out(out_valid), .y(decay));
    reg [15:0] b6, b7, b8, b9;
    always @(posedge clk) begin b6 <= sg5; b7 <= b6; b8 <= b7; b9 <= b8; end
    assign beta = b9;
endmodule

// ---------------------------------------------------------------------------
// One head of the delta rule for one token.
//
// After `start` latches q, k (int8 unit vectors), v (int8), decay and beta
// (U16), the K rows of the int16 state stream in.  Pass 1 decays each row
// as it arrives, keeps it, and accumulates pred = k . S_d.  Pass 2 walks the
// kept rows, adds beta k_i (v - pred) with saturation, streams the new rows
// out and accumulates y = q . S'.  y is shifted by YSH into int16 after the
// last row.
//
//   S_d   = (S * decay + 2^15) >> 16
//   pred  = (sum_i k_i S_d[i] + 2^14) >> 15,  diff = sat16(v - pred)
//   S'    = sat16(S_d + ((beta * k_i * diff + 2^14) >> 15))
//   y     = sat16((sum_i q_i S'[i] + 2^(YSH-1)) >> YSH)
//
// Row i out follows row i in by K + 4 cycles; y_valid follows the last row out.
// Golden model: fabric/layer.py delta_state_int.
// ---------------------------------------------------------------------------
module fabric_delta_state #(
    parameter int K   = 128,
    parameter int V   = 128,
    parameter int YSH = 11
) (
    input  wire            clk,
    input  wire            rst_n,
    input  wire            start,
    input  wire [K*8-1:0]  q,
    input  wire [K*8-1:0]  k,
    input  wire [V*8-1:0]  v,
    input  wire [15:0]     decay,
    input  wire [15:0]     beta,
    input  wire            row_in_valid,
    input  wire [V*16-1:0] row_in,
    output reg             row_out_valid,
    output reg  [V*16-1:0] row_out,
    output reg             y_valid,
    output reg  [V*16-1:0] y
);
    localparam int KW = $clog2(K) + 1;
    reg [K*8-1:0]  q_r, k_r;
    reg [V*8-1:0]  v_r;
    reg [15:0]     d_r, b_r;
    reg [V*16-1:0] sd_mem [0:K-1];
    reg [KW-1:0]   wr, rd;
    reg [1:0]      phase;            // 0 pass 1, 1 diff, 2 pass 2, 3 y
    reg signed [31:0] pred_acc [0:V-1];
    reg signed [31:0] y_acc [0:V-1];
    reg signed [15:0] diff [0:V-1];

    // Pass 1 stage A: register the row and its index.
    reg            va;
    reg [V*16-1:0] rowa;
    reg [KW-1:0]   ia;
    // Pass 2 stage D: register the kept row and its index.
    reg            vd;
    reg [V*16-1:0] rowd;
    reg [KW-1:0]   id;

    integer j;
    reg signed [63:0] sd, sn, dl;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase <= 2'd0; wr <= 0; rd <= 0; va <= 1'b0; vd <= 1'b0;
            row_out_valid <= 1'b0;
        end else begin
            va <= 1'b0;
            vd <= 1'b0;
            row_out_valid <= 1'b0;
            if (start) begin
                q_r <= q; k_r <= k; v_r <= v; d_r <= decay; b_r <= beta;
                phase <= 2'd0; wr <= 0; rd <= 0;
                for (j = 0; j < V; j = j + 1) begin pred_acc[j] <= 0; y_acc[j] <= 0; end
            end
            // Pass 1 input.
            if (phase == 2'd0 && row_in_valid) begin
                va <= 1'b1; rowa <= row_in; ia <= wr;
                wr <= wr + 1'b1;
            end
            // Pass 1 stage B: decay, keep, accumulate pred.
            if (va) begin
                for (j = 0; j < V; j = j + 1) begin
                    sd = fx_rnd_shr($signed(rowa[j*16 +: 16]) * $signed({48'b0, d_r}), 16);
                    sd_mem[ia][j*16 +: 16] <= sd[15:0];
                    pred_acc[j] <= pred_acc[j] + $signed(k_r[ia*8 +: 8]) * $signed(sd[15:0]);
                end
                if (ia == K - 1) phase <= 2'd1;
            end
            // Diff.
            if (phase == 2'd1) begin
                for (j = 0; j < V; j = j + 1)
                    diff[j] <= fx_sat($signed({{56{v_r[j*8+7]}}, v_r[j*8 +: 8]}) - fx_rnd_shr(pred_acc[j], 15), 16);
                phase <= 2'd2;
                rd <= 0;
            end
            // Pass 2 stage D: read a kept row.
            if (phase == 2'd2) begin
                vd <= 1'b1; rowd <= sd_mem[rd]; id <= rd;
                rd <= rd + 1'b1;
                if (rd == K - 1) phase <= 2'd3;
            end
            // Pass 2 stage E: update, emit, accumulate y.
            if (vd) begin
                for (j = 0; j < V; j = j + 1) begin
                    dl = fx_rnd_shr($signed({48'b0, b_r}) * $signed(k_r[id*8 +: 8]) * $signed(diff[j]), 15);
                    sn = fx_sat($signed(rowd[j*16 +: 16]) + dl, 16);
                    row_out[j*16 +: 16] <= sn[15:0];
                    y_acc[j] <= y_acc[j] + $signed(q_r[id*8 +: 8]) * $signed(sn[15:0]);
                end
                row_out_valid <= 1'b1;
            end
        end
    end
    // y: the accumulators settle at the edge the last row leaves, so y is
    // formed one cycle later.
    reg y_pending;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) y_pending <= 1'b0;
        else        y_pending <= vd && (id == K - 1);
    end
    always @(posedge clk) begin
        if (y_pending)
            for (j = 0; j < V; j = j + 1)
                y[j*16 +: 16] <= fx_sat(fx_rnd_shr(y_acc[j], YSH), 16);
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) y_valid <= 1'b0;
        else        y_valid <= y_pending;
    end
endmodule



// ---------------------------------------------------------------------------
// One head of the delta rule for one token on the int8 state with a scale.
//
// The state is S = g * T * 2^-e: T int8 rows, g a U16 scale and e a small
// exponent per head, with peak the largest |T| written since the last
// rescale (it restarts at a rescale).  After
// `start` latches q, k, v, decay, beta, g, e and peak, the new scale is
// g1 = (g * decay + 2^15) >> 16 (nsat is the number of saturated elements
// written last token).  If g1 < 2^15 or nsat > K*V / 2^SAT_SHIFT the head is
// rescaled this token: as the rows arrive T <- sat8((T * g1 + 2^(15-de)) >> (16-de))
// with de = -1 when saturated (e > E_MIN), +1 when peak <= PEAK_GROW
// (e < E_MAX), else 0; then g1 = 1 and
// e <- e + de.  A sequential divider
// forms r = floor(2^31 / g1).  Pass 1 keeps the rows and accumulates
// pred = k . T; then
//   pred_j = (acc_j * g1 + 2^(22+e)) >> (23 + e),  diff_j = sat16(v_j - pred_j),
//   c_j    = (beta * diff_j * r + 2^(23-e)) >> (24 - e)
// and pass 2 walks the kept rows: T'_ij = sat8(T_ij + ((k_i * c_j + 2^13) >> 14)),
// streams them out, tracks the peak and the saturated count and
// accumulates y = q . T'; after the last row
//   y_j = sat16((y_acc_j * g1 + 2^(YSH+7+e)) >> (YSH + 8 + e)).
// g_out and e_out are valid from two cycles after start, peak_out and
// nsat_out with y.
// Golden model: fabric/layer.py delta_state_int8.
// ---------------------------------------------------------------------------
module fabric_delta_state8 #(
    parameter int K     = 128,
    parameter int V     = 128,
    // Value lanes of arithmetic.  The state rows reach this unit over the
    // buffer's 128-bit port, so a row of V bytes takes V/16 cycles to
    // arrive and a V-wide datapath idles for all but one of them.  VL lanes
    // and V/VL slices do the same work at the same rate for a fraction of
    // the multipliers.  VL = V is the unsliced unit, cycle for cycle what it
    // always was.
    parameter int VL    = V,
    parameter int YSH   = 11,
    parameter int E_MIN = -4,
    parameter int E_MAX = 6,
    parameter int PEAK_GROW = 47,            // fabric.layer.PEAK_GROW
    parameter int SAT_SHIFT = 6              // fabric.layer.SAT_SHIFT
) (
    input  wire            clk,
    input  wire            rst_n,
    input  wire            start,
    input  wire [K*8-1:0]  q,
    input  wire [K*8-1:0]  k,
    input  wire [V*8-1:0]  v,
    input  wire [15:0]     decay,
    input  wire [15:0]     beta,
    input  wire [15:0]     g_in,
    input  wire [7:0]      e_in,
    input  wire [7:0]      peak_in,
    input  wire [15:0]     nsat_in,
    output reg  [15:0]     g_out,
    output reg  [7:0]      e_out,
    output reg  [7:0]      peak_out,
    output reg  [15:0]     nsat_out,
    input  wire            row_in_valid,
    input  wire [V*8-1:0]  row_in,
    output reg             row_out_valid,
    output reg  [V*8-1:0]  row_out,
    output reg             y_valid,
    output reg  [V*16-1:0] y
);
    localparam int KW = $clog2(K) + 1;
    localparam int SL = V / VL;                  // slices of the value dimension
    localparam int SW = (SL > 1) ? $clog2(SL) : 1;
    reg [K*8-1:0]  q_r, k_r;
    reg [V*8-1:0]  v_r;
    reg [15:0]     b_r, g1;
    reg signed [7:0] e1;
    reg            rescale;
    reg signed [3:0] de;
    // The rows pass 1 keeps, for pass 2 to walk.  A macro, not a register
    // array: at the 9B head this is 128 rows of 128 bytes, 131,072 flops,
    // and the elaboration did not map at all -- yosys ran 85 minutes and
    // died in simplemap.  The two passes are disjoint in time, so one read
    // port and one write port are enough, and the read already lands a
    // cycle after its address.
    localparam int TA = (K > 1) ? $clog2(K) : 1;
    reg [KW-1:0]   wr, rd;
    reg [3:0]      phase;            // 0 pass 1, 1 pred, 2 diff, 3 beta*diff, 4 c*r, 5 c, 6 pass 2, 7 y, 8 reduce
    reg signed [31:0] pred_acc [0:V-1];
    reg signed [31:0] y_acc [0:V-1];
    // The scale's two products are left carry-save, each with its own
    // rounding constant as one more operand of its tree.  Resolved where they
    // are formed, the stage is a 32 by 16 multiply and then the add of that
    // constant -- two carry propagations, and 2,010 ps of this unit once the
    // fanout came off it.  The stages that read them shift by a constant, so
    // resolving there costs the one add that was already going to happen.
    localparam int PMW = 48;
    reg [PMW-1:0] pms [0:V-1], pmc [0:V-1];
    reg [PMW-1:0] yms [0:V-1], ymc [0:V-1];   // the y stage's own, so no stage reads another's write
    wire [PMW-1:0] pms_n [0:VL-1], pmc_n [0:VL-1], yms_n [0:VL-1], ymc_n [0:VL-1];
    reg signed [15:0] diff [0:V-1];
    reg signed [31:0] bd [0:V-1];
    // The rescale's product is left carry-save: bd by the 17-bit reciprocal
    // was a 32 by 17 multiply in one phase, and its carry chain was this
    // unit's critical path (the quotient's bits into cm).  The phase that
    // follows already adds and shifts, so it resolves the pair there for the
    // add it was doing anyway -- the rounding constant rides in the tree.
    reg signed [49:0] cms [0:V-1], cmc [0:V-1];
    reg signed [24:0] c [0:V-1];
    reg [7:0]      peak_acc;
    reg [15:0]     nsat_acc;
    // The shifts are the exponent's, fixed for the token: decoded once, with
    // a copy of each amount per lane so one flop does not drive every lane's
    // shifter select.  All of them are positive over the exponent's range.
    reg [5:0]      shp, shc, shy, shr_;
    reg signed [47:0] rndp;
    reg signed [49:0] rndc;
    reg signed [47:0] rndy;
    reg signed [23:0] rndr;
    wire [5:0] shp_l [0:VL-1];
    wire [5:0] shc_l [0:VL-1];
    wire [5:0] shy_l [0:VL-1];
    wire [5:0] shr_l [0:VL-1];
    // The scale multiplies every lane's accumulator, twice over; a copy per
    // lane keeps that off one flop's fanout, as for the shift amounts.
    wire [15:0] g1_l [0:VL-1];
    wire [15:0]    b_r_l [0:VL-1];
    wire [15:0]    gren_l [0:VL-1];
    wire signed [23:0] rndr_l [0:VL-1];
    wire signed [47:0] rndp_l [0:VL-1];
    wire signed [47:0] rndy_l [0:VL-1];
    reg  [7:0]     rab;
    integer        sl;
    wire           resc_l [0:VL-1];
    wire [VL*50-1:0] cms_w, cmc_w;
    genvar gv;
    generate
        for (gv = 0; gv < VL; gv = gv + 1) begin : g_sh
            fabric_const_copy #(.W(6))  u_p (.clk(clk), .d(shp),  .q(shp_l[gv]));
            fabric_const_copy #(.W(6))  u_c (.clk(clk), .d(shc),  .q(shc_l[gv]));
            fabric_const_copy #(.W(6))  u_y (.clk(clk), .d(shy),  .q(shy_l[gv]));
            fabric_const_copy #(.W(6))  u_r (.clk(clk), .d(shr_), .q(shr_l[gv]));
            fabric_const_copy #(.W(16)) u_g (.clk(clk), .d(g1),   .q(g1_l[gv]));
            // Beta multiplies every lane's difference, and one flop at 248
            // loads and 394 fF was 1,021 of this unit's 2,688 ps before any
            // arithmetic started -- the same shape as the shifts and the
            // scale beside it, and the same fix.
            fabric_const_copy #(.W(16)) u_b (.clk(clk), .d(b_r),   .q(b_r_l[gv]));
            // The keep's scale multiplies every lane and its flag selects
            // every lane's mux: 112 and 149 loads on two flops, which is the
            // same shape again.  Both are settled well before pass 1 reaches
            // A3, as `g1` beside them is.
            fabric_const_copy #(.W(16)) u_gr (.clk(clk), .d(gren),    .q(gren_l[gv]));
            // The rounds are added in every lane too: 97 loads on `rndr` and
            // the same shape on `rndp`.
            fabric_const_copy #(.W(24)) u_rr (.clk(clk), .d(rndr),    .q(rndr_l[gv]));
            fabric_const_copy #(.W(48)) u_rp (.clk(clk), .d(rndp),    .q(rndp_l[gv]));
            fabric_const_copy #(.W(48)) u_ry (.clk(clk), .d(rndy),    .q(rndy_l[gv]));
            fabric_mul_cs #(.AW(32), .BW(16), .PW(PMW), .ADD(1)) u_pm (
                .a(pred_acc[ps * VL + gv]), .b(g1_l[gv]), .addend(rndp_l[gv]),
                .s(pms_n[gv]), .c(pmc_n[gv]));
            fabric_mul_cs #(.AW(32), .BW(16), .PW(PMW), .ADD(1)) u_ym (
                .a(y_acc[ps * VL + gv]), .b(g1_l[gv]), .addend(rndy_l[gv]),
                .s(yms_n[gv]), .c(ymc_n[gv]));
            fabric_const_copy #(.W(1))  u_rs (.clk(clk), .d(rescale), .q(resc_l[gv]));
        end
    endgenerate
    wire           sat_in = ({16'd0, nsat_in} > ((K * V) >> SAT_SHIFT));

    // The scale the cycle after start (gren keeps the unclamped value the
    // rescaling multiplies by); then r = floor(2^31 / g1) by restoring
    // division, one quotient bit per cycle.  The diff phase waits for it.
    reg [31:0] gprod;
    reg [15:0] gren;
    reg        g1_ready, dividing, r_ready, saturated;
    reg [32:0] rem, rem_next;
    reg [16:0] quo;
    reg [5:0]  dstep;
    integer j, u;
    // One slice per cycle through each stage.  `ps` walks the phases between
    // the passes; the passes carry a slice with each pipeline stage.
    reg [SW-1:0] ps, sa1, sa2, sa3, sa4, sd1, sd2, sd3, sd4;
    reg signed [7:0] e1f;
    reg signed [3:0] de_w;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            g1 <= 0; gren <= 0; e1 <= 0; rescale <= 1'b0; de <= 0; g1_ready <= 1'b0; g_out <= 0; e_out <= 0;
            dividing <= 1'b0; r_ready <= 1'b0; rem <= 0; quo <= 0; dstep <= 0;
            shp <= 0; shc <= 0; shy <= 0; shr_ <= 0; rndp <= 0; rndc <= 0; rndy <= 0; rndr <= 0;
        end else begin
            g1_ready <= start;
            if (start) begin
                gprod = ({16'd0, g_in} * {16'd0, decay} + 32'd32768) >> 16;
                saturated = sat_in;
                gren <= gprod[15:0];
                de_w = (gprod < 32'd32768 || saturated)
                       ? ((saturated && $signed(e_in) > E_MIN) ? -4'sd1
                          : ((!saturated && ({24'd0, peak_in} <= PEAK_GROW) && $signed(e_in) < E_MAX) ? 4'sd1 : 4'sd0))
                       : 4'sd0;
                shr_ <= 16 - de_w;   rndr <= 24'sd1 <<< (15 - de_w);
                if (gprod < 32'd32768 || saturated) begin
                    rescale <= 1'b1;
                    if (saturated && $signed(e_in) > E_MIN) de <= -4'sd1;
                    else if (!saturated && ({24'd0, peak_in} <= PEAK_GROW) && $signed(e_in) < E_MAX) de <= 4'sd1;
                    else de <= 4'sd0;
                    g1 <= 16'hFFFF;
                end else begin
                    rescale <= 1'b0; de <= 4'sd0; g1 <= gprod[15:0];
                end
                e1 <= $signed(e_in);
            end
            if (g1_ready) begin
                if (rescale) e1 <= e1 + {{4{de[3]}}, de};
                g_out <= g1; e_out <= rescale ? e1 + {{4{de[3]}}, de} : e1;
                e1f = rescale ? e1 + {{4{de[3]}}, de} : e1;
                shp  <= 23 + e1f;        rndp <= 48'sd1  <<< (22 + e1f);
                shc  <= 24 - e1f;        rndc <= 50'sd1  <<< (23 - e1f);
                shy  <= YSH + 8 + e1f;   rndy <= 48'sd1  <<< (YSH + 7 + e1f);
                rem <= 0; quo <= 0; dstep <= 0; dividing <= 1'b1; r_ready <= 1'b0;
            end else if (dividing) begin
                rem_next = {rem[31:0], dstep == 6'd0};                    // the dividend 2^31, top bit first
                if (rem_next >= {17'd0, g1}) begin rem <= rem_next - {17'd0, g1}; quo <= {quo[15:0], 1'b1}; end
                else begin rem <= rem_next; quo <= {quo[15:0], 1'b0}; end
                dstep <= dstep + 1'b1;
                if (dstep == 6'd31) begin dividing <= 1'b0; r_ready <= 1'b1; end
            end
        end
    end
    wire [16:0] r = quo;
    // The reciprocal is the second operand of every lane's carry-save
    // multiply, so the quotient register carried 2,091 loads and 3.37 pF at
    // 32 lanes -- 8.13 of that geometry's 9.68 ns, and it grows with V.  A
    // copy per lane.  The copy is a cycle behind the quotient, which costs
    // nothing: the diff phase waits on `r_ready` and two more phases run
    // before anything multiplies by it.
    wire [16:0] r_l [0:VL-1];
    genvar gr;
    generate
        for (gr = 0; gr < VL; gr = gr + 1) begin : g_rl
            fabric_const_copy #(.W(17)) u_r (.clk(clk), .d(r), .q(r_l[gr]));
            fabric_mul_cs #(.AW(32), .BW(17), .PW(50)) u_c (
                .a(bd[ps * VL + gr]), .b(r_l[gr]), .addend(rndc),
                .s(cms_w[gr*50 +: 50]), .c(cmc_w[gr*50 +: 50]));
        end
    endgenerate

    // Pass 1 is A1 latch, A2 rescale product, A3 rescale and keep, A4 the
    // key's product, then the accumulate; pass 2 is B1 read, B2 the update's
    // product, B3 the saturating add, B4 the query's product, then the
    // accumulate.  One multiply, or one round with its saturate, to a stage.
    // The per-lane arrays are written with blocking assignments so Verilator
    // keeps the lane loops as loops, which makes the order of the stages in
    // this block significant: they are written newest first, so each reads
    // the previous cycle's value of the stage before it.
    localparam int G1N = (V + 7) / 8;
    localparam int G2N = (G1N + 7) / 8;
    reg            va1, va2, va3, va4;
    reg [V*8-1:0]  rowa1, rowa2;
    reg [KW-1:0]   ia1, ia2, ia3, ia4;
    // Pipeline-local: written by one stage and read by the next at the same
    // slice, so they hold a slice's lanes rather than the whole value
    // dimension.  V-wide they would be a mux and a decoder per access.
    reg signed [23:0] rs2 [0:VL-1];
    reg [V*8-1:0]  tr3;
    reg signed [15:0] kp4 [0:VL-1];
    reg            vd1, vd2, vd3, vd4;
    reg [V*8-1:0]  rowd2;
    wire [V*8-1:0] rowd1;               // the macro's output, valid with vd1
    fabric_sram #(.W(V*8), .D(K), .NRD(1), .NWR(1), .MB(V*8)) u_t (
        .clk(clk), .rd_en(1'b1), .rd_addr(rd[TA-1:0]), .rd_data(rowd1),
        .wr_en(va3 && sa3 == SL - 1), .wr_addr(ia3[TA-1:0]), .wr_data(tr3), .wr_mask(1'b1));
    reg [KW-1:0]   id1, id2, id3, id4;
    // `k_r` and `q_r` are K bytes, and indexing them with a row counter is a
    // barrel shifter over all K*8 bits.  At the real 128 rows `ia3` alone
    // carried 1,026 loads and 1.63 pF -- 3.98 of that geometry's 5.26 ns --
    // because the shifter is the whole vector however few bytes come out.
    // An or of masks is the structure the index actually has, and it costs
    // the counter K comparators.
    reg [7:0] k_a3, k_d1, q_d3;
    integer kk;
    always @* begin
        k_a3 = 0; k_d1 = 0; q_d3 = 0;
        for (kk = 0; kk < K; kk = kk + 1) begin
            k_a3 = k_a3 | (k_r[kk*8 +: 8] & {8{ia3 == kk[KW-1:0]}});
            k_d1 = k_d1 | (k_r[kk*8 +: 8] & {8{id1 == kk[KW-1:0]}});
            q_d3 = q_d3 | (q_r[kk*8 +: 8] & {8{id3 == kk[KW-1:0]}});
        end
    end
    reg signed [32:0] dm2 [0:VL-1];
    reg [VL*8-1:0] tn3;
    reg signed [15:0] ym4 [0:VL-1];
    reg [7:0]      pk [0:V-1];
    reg [15:0]     ns [0:V-1];
    reg [7:0]      pk_a [0:G1N-1];
    reg [15:0]     ns_a [0:G1N-1];
    reg [7:0]      pk_b [0:G2N-1];
    reg [15:0]     ns_b [0:G2N-1];
    reg [2:0]      tail;
    reg signed [63:0] tr, tn, dl, pr, mag, nsat;
    integer        gg, jj;
    // The per-element arrays are written with blocking assignments inside the
    // lane loops (each element reads and writes only itself, in one phase), so
    // that Verilator keeps the loops as loops instead of unrolling 128 lanes.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase <= 4'd0; wr <= 0; rd <= 0; row_out_valid <= 1'b0; peak_acc <= 0; y_valid <= 1'b0;
            va1 <= 1'b0; va2 <= 1'b0; va3 <= 1'b0; va4 <= 1'b0;
            vd1 <= 1'b0; vd2 <= 1'b0; vd3 <= 1'b0; vd4 <= 1'b0; tail <= 0;
        end else begin
            va1 <= 1'b0; vd1 <= 1'b0; row_out_valid <= 1'b0; y_valid <= 1'b0;

            // ---- the tail: y and the reduction of the per-lane peak and
            // saturated counts, newest stage first.
            if (phase == 4'd7) begin
                if (tail > 3'd1 || ps == SL - 1) begin tail <= tail + 1'b1; ps <= 0; end
                else ps <= ps + 1'b1;
                if (tail == 3'd4) begin
                    y_valid <= 1'b1;
                    phase <= 4'd0;
                end
                if (tail == 3'd3) begin
                    mag = {56'd0, peak_acc};
                    nsat = 0;
                    for (gg = 0; gg < G2N; gg = gg + 1) begin
                        if ({56'd0, pk_b[gg]} > mag) mag = {56'd0, pk_b[gg]};
                        nsat = nsat + {48'd0, ns_b[gg]};
                    end
                    peak_acc <= mag[7:0];  peak_out <= mag[7:0];
                    nsat_acc <= nsat[15:0]; nsat_out <= nsat[15:0];
                end
                if (tail == 3'd1)
                    for (u = 0; u < VL; u = u + 1) begin
                        j = ps * VL + u;
                        y[j*16 +: 16] <= fx_sat(
                            ($signed(yms[j]) + $signed({ymc[j][PMW-2:0], 1'b0})) >>> shy_l[u], 16);
                    end
                if (tail == 3'd2)
                    for (gg = 0; gg < G2N; gg = gg + 1) begin
                        pk_b[gg] = 0; ns_b[gg] = 0;
                        for (jj = 0; jj < 8; jj = jj + 1)
                            if (gg * 8 + jj < G1N) begin
                                if (pk_a[gg*8 + jj] > pk_b[gg]) pk_b[gg] = pk_a[gg*8 + jj];
                                ns_b[gg] = ns_b[gg] + ns_a[gg*8 + jj];
                            end
                    end
                if (tail == 3'd1)
                    for (gg = 0; gg < G1N; gg = gg + 1) begin
                        pk_a[gg] = 0; ns_a[gg] = 0;
                        for (jj = 0; jj < 8; jj = jj + 1)
                            if (gg * 8 + jj < V) begin
                                if (pk[gg*8 + jj] > pk_a[gg]) pk_a[gg] = pk[gg*8 + jj];
                                ns_a[gg] = ns_a[gg] + ns[gg*8 + jj];
                            end
                    end
                if (tail == 3'd0)
                    for (u = 0; u < VL; u = u + 1) begin
                        j = ps * VL + u;
                        yms[j] = yms_n[u]; ymc[j] = ymc_n[u];
                    end
            end

            // ---- pass 2, newest stage first.
            if (vd4) begin
                for (u = 0; u < VL; u = u + 1) begin
                    j = sd4 * VL + u;
                    y_acc[j] = y_acc[j] + ym4[u];
                end
                if (id4 == K - 1 && sd4 == SL - 1) begin phase <= 4'd7; tail <= 0; ps <= 0; end
            end
            vd4 <= vd3; id4 <= id3; sd4 <= sd3;
            if (vd3) begin
                for (u = 0; u < VL; u = u + 1) begin
                    j = sd3 * VL + u;
                    ym4[u] = $signed(q_d3) * $signed(tn3[u*8 +: 8]);
                    tn = $signed({{56{tn3[u*8+7]}}, tn3[u*8 +: 8]});
                    mag = (tn < 0) ? -tn : tn;
                    if (mag[7:0] > pk[j]) pk[j] = mag[7:0];
                    if (mag >= 127) ns[j] = ns[j] + 1'b1;
                end
            end
            vd3 <= vd2; id3 <= id2; sd3 <= sd2;
            if (vd2) begin
                for (u = 0; u < VL; u = u + 1) begin
                    j = sd2 * VL + u;
                    dl = fx_rnd_shr(dm2[u], 14);
                    tn = fx_sat($signed({{56{rowd2[j*8+7]}}, rowd2[j*8 +: 8]}) + dl, 8);
                    tn3[u*8 +: 8] = tn[7:0];
                    row_out[j*8 +: 8] <= tn[7:0];
                end
                if (sd2 == SL - 1) row_out_valid <= 1'b1;   // the row is whole on the last slice
            end
            vd2 <= vd1; id2 <= id1; sd2 <= sd1;
            // `rd` advances at the issue, so the macro has moved on to the
            // next row by the second slice: take the row once, at the first.
            if (vd1 && sd1 == 0) rowd2 <= rowd1;
            if (vd1)
                for (u = 0; u < VL; u = u + 1) begin
                    j = sd1 * VL + u;
                    dm2[u] = $signed(k_d1) * c[j];
                end
            // As in pass 1: a row is SL cycles of VL lanes, so the read
            // address advances once a row and the slice walks between.
            if (vd1 && sd1 != SL - 1) begin vd1 <= 1'b1; sd1 <= sd1 + 1'b1; end
            else if (phase == 4'd6) begin
                vd1 <= 1'b1; sd1 <= 0; id1 <= rd; rd <= rd + 1'b1;
                if (rd == K - 1) phase <= 4'd0;          // the pipeline carries the rest
            end

            // ---- the phases between the passes, newest first.
            if (phase == 4'd5) begin
                for (u = 0; u < VL; u = u + 1) begin
                    j = ps * VL + u;
                    c[j] = ($signed(cms[j]) + $signed({cmc[j][48:0], 1'b0})) >>> shc_l[u];
                end
                if (ps == SL - 1) begin ps <= 0; phase <= 4'd6; rd <= 0; end
                else ps <= ps + 1'b1;
            end
            if (phase == 4'd4) begin
                for (u = 0; u < VL; u = u + 1) begin
                    j = ps * VL + u;
                    cms[j] = cms_w[u*50 +: 50];
                    cmc[j] = cmc_w[u*50 +: 50];
                end
                if (ps == SL - 1) begin ps <= 0; phase <= 4'd5; end
                else ps <= ps + 1'b1;
            end
            if (phase == 4'd3) begin
                for (u = 0; u < VL; u = u + 1) begin
                    j = ps * VL + u;
                    bd[j] = $signed({16'b0, b_r_l[u]}) * diff[j];
                end
                if (ps == SL - 1) begin ps <= 0; phase <= 4'd4; end
                else ps <= ps + 1'b1;
            end
            if (phase == 4'd2) begin
                for (u = 0; u < VL; u = u + 1) begin
                    j = ps * VL + u;
                    pr = ($signed(pms[j]) + $signed({pmc[j][PMW-2:0], 1'b0})) >>> shp_l[u];
                    diff[j] = fx_sat($signed({{56{v_r[j*8+7]}}, v_r[j*8 +: 8]}) - pr, 16);
                end
                if (ps == SL - 1) begin ps <= 0; phase <= 4'd3; end
                else ps <= ps + 1'b1;
            end
            if (phase == 4'd1 && r_ready) begin
                for (u = 0; u < VL; u = u + 1) begin
                    j = ps * VL + u;
                    pms[j] = pms_n[u]; pmc[j] = pmc_n[u];
                end
                if (ps == SL - 1) begin ps <= 0; phase <= 4'd2; end
                else ps <= ps + 1'b1;
            end

            // ---- pass 1, newest stage first.
            if (va4) begin
                for (u = 0; u < VL; u = u + 1) begin
                    j = sa4 * VL + u;
                    pred_acc[j] = pred_acc[j] + {{16{kp4[u][15]}}, kp4[u]};
                end
                if (ia4 == K - 1 && sa4 == SL - 1) begin phase <= 4'd1; ps <= 0; end
            end
            va4 <= va3; ia4 <= ia3; sa4 <= sa3;
            if (va3)
                for (u = 0; u < VL; u = u + 1) begin
                    j = sa3 * VL + u;
                    kp4[u] = $signed(k_a3) * $signed(tr3[j*8 +: 8]);
                end
            va3 <= va2; ia3 <= ia2; sa3 <= sa2;
            if (va2)
                for (u = 0; u < VL; u = u + 1) begin
                    j = sa2 * VL + u;
                    tr = resc_l[u] ? fx_sat((rs2[u] + rndr_l[u]) >>> shr_l[u], 8)
                                 : $signed({{56{rowa2[j*8+7]}}, rowa2[j*8 +: 8]});
                    tr3[j*8 +: 8] <= tr[7:0];   // a register, not a blocking temp: the macro samples it
                end
            va2 <= va1; ia2 <= ia1; sa2 <= sa1; rowa2 <= rowa1;
            if (va1)
                for (u = 0; u < VL; u = u + 1) begin
                    // The lane's byte as a select over the slices, not a
                    // part-select at `sa1 * VL + u`: a variable part-select is
                    // a barrel shifter over the whole row, and `sa1` carried
                    // 151 loads for it.  Over SL slices it is a small mux, and
                    // at SL of one it folds away entirely.
                    rab = 0;
                    for (sl = 0; sl < SL; sl = sl + 1)
                        if (sa1 == sl[SW-1:0]) rab = rowa1[(sl*VL + u)*8 +: 8];
                    rs2[u] = $signed(rab) * $signed({8'b0, gren_l[u]});
                end
            // A row is VL lanes at a time: the issue holds `va1` for SL cycles
            // and walks the slice, which is the rate the row arrived at.
            if (va1 && sa1 != SL - 1) begin va1 <= 1'b1; sa1 <= sa1 + 1'b1; end
            else if (phase == 4'd0 && row_in_valid) begin
                va1 <= 1'b1; sa1 <= 0; rowa1 <= row_in; ia1 <= wr; wr <= wr + 1'b1;
            end

            if (start) begin
                q_r <= q; k_r <= k; v_r <= v; b_r <= beta;
                phase <= 4'd0; wr <= 0; rd <= 0; tail <= 0;
                va1 <= 1'b0; va2 <= 1'b0; va3 <= 1'b0; va4 <= 1'b0;
                vd1 <= 1'b0; vd2 <= 1'b0; vd3 <= 1'b0; vd4 <= 1'b0;
                // The peak runs since the last rescale; a rescale this token (decided in the same edge) restarts it.
                peak_acc <= (((({16'd0, g_in} * {16'd0, decay} + 32'd32768) >> 16) < 32'd32768) || sat_in) ? 8'd0 : peak_in;
                nsat_acc <= 0;
                for (j = 0; j < V; j = j + 1) begin pred_acc[j] = 0; y_acc[j] = 0; pk[j] = 0; ns[j] = 0; end
            end
        end
    end
`ifndef FABRIC_SYNTH
    // The rows arrive a beat at a time, so a row takes V/16 cycles to reach
    // this unit and the slices take SL.  VL is chosen so those are equal; if
    // a row ever arrives while the last one still has slices to run it is
    // dropped, and silently, so say so.
    always @(posedge clk)
        if (rst_n && row_in_valid && va1 && sa1 != SL - 1)
            $display("FAIL: a state row arrived while the one before it still had slices to run");
`endif

endmodule

`default_nettype wire
