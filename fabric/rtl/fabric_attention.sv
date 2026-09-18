// The global layer's vector units: the rotary table and rotation, and the
// online-softmax attention core.
// Golden model: fabric/layer.py (rotary_table_int, rotary_int, attention_int).

`default_nettype none
`include "fabric_fx.svh"

// ---------------------------------------------------------------------------
// sin and cos of pos * inv_freq[j] for the R/2 rotary frequencies of one
// position: the product in Q0.32 turns keeps its fraction, whose top 16
// bits index the sine table (cos is sin a quarter turn on).  start, then
// done after R/2 + 3 cycles; the tables stay valid until the next start.
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
    wire [63:0]  prod = pos * inv_freq[j*32 +: 32];
    wire [15:0]  turn = prod[31:16];
    reg          v1, v2, v3;
    reg [JW-1:0] j1, j2, j3;
    wire [15:0]  s_w, c_w;
    fabric_lut #(.IB(10), .FB(6), .W(16), .FILE({LUT_DIR, "lut_sin.hex"})) u_sin (.clk(clk), .u(turn), .y(s_w));
    fabric_lut #(.IB(10), .FB(6), .W(16), .FILE({LUT_DIR, "lut_sin.hex"})) u_cos (.clk(clk), .u(turn + 16'h4000), .y(c_w));
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0; j <= 0; v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (start) begin busy <= 1'b1; j <= 0; end
            else if (busy) begin
                j <= j + 1'b1;
                if (j == H - 1) busy <= 1'b0;
            end
            v1 <= busy; j1 <= j;
            v2 <= v1;   j2 <= j1;
            v3 <= v2;   j3 <= j2;
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
// streams in L per beat, is buffered, and streams out L per beat two
// cycles after the last input beat.
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
    reg [HD*16-1:0] buffer;
    reg [BW-1:0]    wr, rd, rd_addr;
    reg             draining, rd_valid;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr <= 0; rd <= 0; draining <= 1'b0; rd_valid <= 1'b0;
        end else begin
            rd_valid <= 1'b0;
            if (in_valid) begin
                buffer[wr*L*16 +: L*16] <= in_x;
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
    // P1: rotate the beat's elements.
    reg            v1;
    reg [L*16-1:0] y1;
    integer l, e, p;
    reg signed [63:0] x1, x2, s, c, t;
    always @(posedge clk) begin
        v1 <= rd_valid;
        for (l = 0; l < L; l = l + 1) begin
            e = rd_addr * L + l;
            if (e < H) begin
                x1 = $signed(buffer[e*16 +: 16]);
                x2 = $signed(buffer[(e + H)*16 +: 16]);
                s  = $signed(sin_tab[e*16 +: 16]);
                c  = $signed(cos_tab[e*16 +: 16]);
                t  = fx_sat(fx_rnd_shr(x1 * c - x2 * s, 15), 16);
            end else if (e < R) begin
                p  = e - H;
                x1 = $signed(buffer[p*16 +: 16]);
                x2 = $signed(buffer[e*16 +: 16]);
                s  = $signed(sin_tab[p*16 +: 16]);
                c  = $signed(cos_tab[p*16 +: 16]);
                t  = fx_sat(fx_rnd_shr(x2 * c + x1 * s, 15), 16);
            end else begin
                t  = $signed(buffer[e*16 +: 16]);
            end
            y1[l*16 +: 16] <= t[15:0];
        end
    end
    // P2: requantize.
    always @(posedge clk) begin
        out_valid <= v1;
        for (l = 0; l < L; l = l + 1)
            out_y[l*8 +: 8] <= fx_requant($signed(y1[l*16 +: 16]), mult, shift, 8);
    end
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

    reg [HD*8-1:0]  q_mem [0:G-1];
    reg [HD*8-1:0]  gate_mem [0:G-1];
    reg [OW*HD-1:0] o_mem [0:G-1];
    reg signed [31:0] score [0:G-1];
    reg signed [31:0] m_r [0:G-1];
    reg               m_valid [0:G-1];
    reg [LW-1:0]      l_r [0:G-1];
    reg [15:0]        f_r [0:G-1];
    reg [15:0]        p_r [0:G-1];

    // Beat counters per kind.
    reg [BW-1:0] beat;
    reg [GW-1:0] head;
    // State.
    localparam [2:0] S_ACCEPT = 3'd0, S_EXP = 3'd1, S_APPLY = 3'd2, S_VALUE = 3'd3,
                     S_RECIP = 3'd4, S_OUT = 3'd5, S_DONE = 3'd6;
    reg [2:0] state;
    assign in_ready = (state == S_ACCEPT) || (state == S_VALUE);

    // Score contribution of a key beat for every head.
    integer g, l;
    reg signed [63:0] contrib [0:G-1];
    always @* begin
        for (g = 0; g < G; g = g + 1) begin
            contrib[g] = 0;
            for (l = 0; l < L; l = l + 1)
                contrib[g] = contrib[g] + $signed(q_mem[g][(beat*L + l)*8 +: 8]) * $signed(in_data[l*8 +: 8]);
        end
    end

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

    // Reciprocal for the output.
    reg          rc_start;
    wire         rc_done;
    wire [16:0]  rc_r;
    wire [5:0]   rc_lz;
    reg [LW-1:0] rc_l;
    fabric_recip #(.LW(LW), .LUT_DIR(LUT_DIR)) u_rc (.clk(clk), .start(rc_start), .l(rc_l), .done(rc_done), .r(rc_r), .lz_out(rc_lz));
    reg [16:0] r_hold;
    reg [5:0]  lz_hold;

    // Output pipeline: O1 w and tg, O2..O4 sigmoid, O5 out.
    reg            ov1;
    reg [L*16-1:0] w1;
    reg [L*16-1:0] tg1;
    wire [L-1:0]    sgv;
    wire [L*16-1:0] sg4;
    generate
        for (gg = 0; gg < L; gg = gg + 1) begin : g_sig
            fabric_sigmoid #(.LUT_DIR(LUT_DIR)) u_sg (.clk(clk), .valid_in(ov1), .t(tg1[gg*16 +: 16]), .valid_out(sgv[gg]), .y(sg4[gg*16 +: 16]));
        end
    endgenerate
    reg [L*16-1:0] w2, w3, w4;
    always @(posedge clk) begin w2 <= w1; w3 <= w2; w4 <= w3; end

    reg signed [63:0] sc, dd, ow, tmp;
    reg [GW-1:0] ohead;
    reg [BW-1:0] obeat;
    reg          out_go;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_ACCEPT; beat <= 0; head <= 0; exp_go <= 1'b0; rc_start <= 1'b0; done <= 1'b0;
            ov1 <= 1'b0; out_go <= 1'b0; ohead <= 0; obeat <= 0;
            for (g = 0; g < G; g = g + 1) begin m_valid[g] <= 1'b0; l_r[g] <= 0; score[g] <= 0; end
        end else begin
            exp_go <= 1'b0;
            rc_start <= 1'b0;
            done <= 1'b0;
            ov1 <= 1'b0;
            if (start) begin
                state <= S_ACCEPT; beat <= 0; head <= 0;
                for (g = 0; g < G; g = g + 1) begin
                    m_valid[g] <= 1'b0; l_r[g] <= 0; score[g] <= 0; o_mem[g] <= 0;
                end
            end
            case (state)
                S_ACCEPT: begin
                    if (finish) begin
                        state <= S_RECIP; ohead <= 0; rc_l <= l_r[0]; rc_start <= 1'b1;
                    end else if (in_valid) begin
                        case (in_kind)
                            2'd0: q_mem[head][beat*L*8 +: L*8]    <= in_data;
                            2'd1: gate_mem[head][beat*L*8 +: L*8] <= in_data;
                            2'd2: for (g = 0; g < G; g = g + 1) score[g] <= score[g] + contrib[g][31:0];
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
                    // The score is complete: requantize, compare with the maximum, launch the exponentials.
                    for (g = 0; g < G; g = g + 1) begin
                        sc = fx_rnd_shr($signed({{32{score[g][31]}}, score[g]}) * $signed({48'b0, mult_s}), sh_s);
                        if (!m_valid[g] || sc > $signed({{32{m_r[g][31]}}, m_r[g]})) begin
                            newmax[g] <= 1'b1;
                            dd = m_valid[g] ? (sc - $signed({{32{m_r[g][31]}}, m_r[g]})) : 64'sd0;
                            m_r[g] <= sc[31:0];
                        end else begin
                            newmax[g] <= 1'b0;
                            dd = $signed({{32{m_r[g][31]}}, m_r[g]}) - sc;
                        end
                        d_in[g] <= (dd > 64'sd4194303) ? 22'd4194303 : dd[21:0];
                        score[g] <= 0;
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
                        for (g = 0; g < G; g = g + 1)
                            for (l = 0; l < L; l = l + 1) begin
                                ow = fx_rnd_shr($signed(o_mem[g][(beat*L + l)*OW +: OW]) * $signed({48'b0, f_r[g]}), 16)
                                     + $signed({48'b0, p_r[g]}) * $signed(in_data[l*8 +: 8]);
                                o_mem[g][(beat*L + l)*OW +: OW] <= ow[OW-1:0];
                            end
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
                        w1[l*16 +: 16]  <= fx_sat(fx_rnd_shr($signed(o_mem[ohead][(obeat*L + l)*OW +: OW]) * $signed({47'b0, r_hold}), 7 + LW - lz_hold), 16);
                        tg1[l*16 +: 16] <= fx_requant($signed(gate_mem[ohead][(obeat*L + l)*8 +: 8]), mult_gate, sh_gate, 16);
                    end
                    if (obeat == BEATS - 1) begin
                        if (ohead == G - 1) state <= S_DONE;
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
                    // Let the output pipeline drain before done.
                    obeat <= obeat + 1'b1;
                    if (obeat == 7) begin done <= 1'b1; state <= S_ACCEPT; obeat <= 0; end
                end
                default: state <= S_ACCEPT;
            endcase
        end
    end
    // O5: gate and requantize.
    always @(posedge clk) begin
        out_valid <= sgv[0];
        for (l = 0; l < L; l = l + 1)
            out_data[l*8 +: 8] <= fx_requant($signed(w4[l*16 +: 16]) * $signed({48'b0, sg4[l*16 +: 16]}), mult_o, sh_o, 8);
    end
endmodule

`default_nettype wire
