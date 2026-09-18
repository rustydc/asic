// The Gated DeltaNet units: the causal convolution with its SiLU, the
// per-head gates, and the delta-rule state update.
// Golden model: fabric/layer.py (conv_silu_int, head_gates_int, delta_state_int).

`default_nettype none
`include "fabric_fx.svh"

// ---------------------------------------------------------------------------
// Depthwise causal convolution of K taps and SiLU, L channels per beat.
// The beat carries the channel's new int8 sample, its K-1 previous samples
// (oldest first), its int8 taps and its constants; it returns the int8
// output and the shifted history.  Latency 7.
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
    // S1: accumulate the taps, shift the history.
    reg                 v1;
    reg signed [63:0]   acc1 [0:L-1];
    reg [L*HW-1:0]      hist1;
    reg [L*16-1:0]      mi1, mo1;
    reg [L*6-1:0]       si1, so1;
    integer c, j;
    reg signed [63:0] acc;
    always @(posedge clk) begin
        v1 <= in_valid;
        for (c = 0; c < L; c = c + 1) begin
            acc = 0;
            for (j = 0; j < K - 1; j = j + 1)
                acc = acc + $signed(in_hist[c*HW + j*8 +: 8]) * $signed(in_w[c*K*8 + j*8 +: 8]);
            acc = acc + $signed(in_x[c*8 +: 8]) * $signed(in_w[c*K*8 + (K-1)*8 +: 8]);
            acc1[c] <= acc;
            if (K > 2) hist1[c*HW +: HW] <= {in_x[c*8 +: 8], in_hist[c*HW + 8 +: HW-8]};
            else       hist1[c*HW +: HW] <= in_x[c*8 +: 8];
        end
        mi1 <= mult_in; si1 <= sh_in; mo1 <= mult_out; so1 <= sh_out;
    end
    // S2: to F16.
    reg            v2;
    reg [L*16-1:0] t2;
    reg [L*HW-1:0] hist2;
    reg [L*16-1:0] mo2;
    reg [L*6-1:0]  so2;
    always @(posedge clk) begin
        v2 <= v1;
        for (c = 0; c < L; c = c + 1)
            t2[c*16 +: 16] <= fx_requant(acc1[c], mi1[c*16 +: 16], si1[c*6 +: 6], 16);
        hist2 <= hist1; mo2 <= mo1; so2 <= so1;
    end
    // S3..S6: SiLU per lane; the history and constants wait four cycles.
    wire [L-1:0]    sv;
    wire [L*16-1:0] s6;
    genvar g;
    generate
        for (g = 0; g < L; g = g + 1) begin : g_silu
            fabric_silu #(.LUT_DIR(LUT_DIR)) u_silu (.clk(clk), .valid_in(v2), .t(t2[g*16 +: 16]), .valid_out(sv[g]), .y(s6[g*16 +: 16]));
        end
    endgenerate
    reg [L*HW-1:0] hist3, hist4, hist5, hist6;
    reg [L*16-1:0] mo3, mo4, mo5, mo6;
    reg [L*6-1:0]  so3, so4, so5, so6;
    always @(posedge clk) begin
        hist3 <= hist2; hist4 <= hist3; hist5 <= hist4; hist6 <= hist5;
        mo3 <= mo2; mo4 <= mo3; mo5 <= mo4; mo6 <= mo5;
        so3 <= so2; so4 <= so3; so5 <= so4; so6 <= so5;
    end
    // S7: to int8.
    always @(posedge clk) begin
        out_valid <= sv[0];
        out_hist  <= hist6;
        for (c = 0; c < L; c = c + 1)
            out_y[c*8 +: 8] <= fx_requant($signed(s6[c*16 +: 16]), mo6[c*16 +: 16], so6[c*6 +: 6], 8);
    end
endmodule

// ---------------------------------------------------------------------------
// Per-head gates from the fabric's raw accumulators of in_proj_a and in_proj_b:
//   beta  = sigmoid(b * mult_b >> sh_b)
//   decay = exp(-A * softplus(a * mult_a >> sh_a + dt_bias))
// One head per beat, A in Q6.10, dt_bias in F16, both results U16.  Latency 9.
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
    // S1: both to F16.
    reg               v1;
    reg signed [15:0] ta1, tb1, dt1;
    reg [15:0]        ac1;
    always @(posedge clk) begin
        v1  <= in_valid;
        ta1 <= fx_requant(a_acc, mult_a, sh_a, 16);
        tb1 <= fx_requant(b_acc, mult_b, sh_b, 16);
        dt1 <= dt_bias;
        ac1 <= a_coef;
    end
    // S2: dt_bias.
    reg               v2;
    reg signed [15:0] ta2, tb2;
    reg [15:0]        ac2;
    always @(posedge clk) begin
        v2  <= v1;
        ta2 <= fx_sat($signed({{48{ta1[15]}}, ta1}) + $signed({{48{dt1[15]}}, dt1}), 16);
        tb2 <= tb1;
        ac2 <= ac1;
    end
    // S3..S5: softplus and sigmoid side by side.
    wire        spv, sgv;
    wire [15:0] sp5, sg5;
    fabric_softplus #(.LUT_DIR(LUT_DIR)) u_sp (.clk(clk), .valid_in(v2), .t(ta2), .valid_out(spv), .y(sp5));
    fabric_sigmoid  #(.LUT_DIR(LUT_DIR)) u_sg (.clk(clk), .valid_in(v2), .t(tb2), .valid_out(sgv), .y(sg5));
    reg [15:0] ac3, ac4, ac5;
    always @(posedge clk) begin ac3 <= ac2; ac4 <= ac3; ac5 <= ac4; end
    // S6: t = A * softplus in F16 (22 bits).
    reg        v6;
    reg [21:0] t6;
    wire signed [63:0] prod = fx_rnd_shr($signed({48'b0, ac5}) * $signed({48'b0, sp5}), 10);
    always @(posedge clk) begin
        v6 <= spv;
        t6 <= prod[21:0];
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

`default_nettype wire
