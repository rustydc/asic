// The controller's gateware, where it is arithmetic: the sampler that turns
// the head dies' lists into a token, and the CRC the ring's packets carry.
// Golden model: fabric/controller.py.  The control -- which context goes
// next, the table of slots -- is the FPGA's soft side and is not here.

`timescale 1ns/1ps
`default_nettype none

// ---------------------------------------------------------------------------
// The sampler.  Two lists arrive, each K rows in descending order of logit
// with the logits as signed fixed point of 10 fraction bits; on finish they
// are merged, the first top_k candidates weighted through the exponential
// table -- exp(-(max - l) / T), T carried as its reciprocal in F16 -- summed,
// cut to the shortest prefix holding top_p of the mass, and drawn from with
// (rnd * total) >> 32, the token being the first candidate whose cumulative
// weight exceeds the draw.  Every step is the model's, so the same random
// word gives the same token.  About 2K + 2 top_k cycles from finish to out.
// ---------------------------------------------------------------------------
module fabric_sampler #(
    parameter int K = 32,                       // rows a list may hold
    parameter     LUT_DIR = "./"
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               clear,             // a new item: forget the lists
    input  wire               in_valid,          // one entry of one list
    input  wire               in_list,           // which head die's
    input  wire [31:0]        in_row,
    input  wire signed [31:0] in_logit,
    input  wire               finish,            // both lists are in
    input  wire [15:0]        inv_t,             // 2^10 / temperature
    input  wire [7:0]         top_k,
    input  wire [15:0]        top_p,             // U16; 65535 keeps every candidate
    input  wire [31:0]        rnd,               // the random word
    output reg                out_valid,
    output reg  [31:0]        out_row,
    output reg  [7:0]         out_index           // among the merged candidates
);
    localparam int FF = 10, N = 2 * K, NW = $clog2(N) + 1;

    // The lists as they arrive.
    reg [31:0]        row0 [0:K-1], row1 [0:K-1];
    reg signed [31:0] lg0 [0:K-1], lg1 [0:K-1];
    reg [NW-1:0]      n0, n1;
    // The merge, the candidates in order, and their cumulative weights.
    reg [31:0]        mrow [0:N-1];
    reg [31:0]        cum  [0:N-1];
    reg [NW-1:0]      p0, p1, m, k, w_i, keep, scan;
    reg signed [31:0] lmax;
    reg [31:0]        total;
    reg [47:0]        pcut;                      // top_p * total
    reg [63:0]        prod;
    reg [31:0]        draw;
    reg               sel1, have, pend;
    reg signed [32:0] diff;
    reg [47:0]        tt;
    reg [21:0]        t;
    reg               ev;
    wire              wv;
    wire [15:0]       wy;

    localparam [2:0] S_IDLE = 0, S_MERGE = 1, S_DRAIN = 2, S_CUT = 3, S_DRAW = 4, S_OUT = 5;
    reg [2:0] state;

    fabric_exp_neg #(.LUT_DIR(LUT_DIR)) u_exp (.clk(clk), .valid_in(ev), .t(t), .valid_out(wv), .y(wy));

    // The next candidate: the larger logit of the two heads, the lower row on a tie.
    always @* begin
        have = (p0 < n0) || (p1 < n1);
        if (p0 < n0 && p1 < n1)
            sel1 = (lg1[p1] > lg0[p0]) || (lg1[p1] == lg0[p0] && row1[p1] < row0[p0]);
        else
            sel1 = (p1 < n1);
    end
    wire signed [31:0] cand_lg  = sel1 ? lg1[p1] : lg0[p0];
    wire [31:0]        cand_row = sel1 ? row1[p1] : row0[p0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; n0 <= 0; n1 <= 0; p0 <= 0; p1 <= 0; m <= 0; k <= 0; w_i <= 0; keep <= 0; scan <= 0;
            out_valid <= 1'b0; ev <= 1'b0; total <= 0; pend <= 1'b0;
        end else begin
            out_valid <= 1'b0;
            ev <= 1'b0;
            if (clear) begin
                n0 <= 0; n1 <= 0; state <= S_IDLE;
            end
            if (in_valid) begin
                if (in_list) begin row1[n1] <= in_row; lg1[n1] <= in_logit; n1 <= n1 + 1'b1; end
                else         begin row0[n0] <= in_row; lg0[n0] <= in_logit; n0 <= n0 + 1'b1; end
            end
            // A weight lands three cycles after its candidate: accumulate in order.
            if (wv) begin
                cum[w_i] <= total + {16'd0, wy};
                total <= total + {16'd0, wy};
                w_i <= w_i + 1'b1;
            end
            case (state)
                S_IDLE: if (finish) begin
                    p0 <= 0; p1 <= 0; m <= 0; w_i <= 0; total <= 0;
                    k <= (top_k < n0 + n1) ? top_k[NW-1:0] : n0 + n1;
                    state <= S_MERGE;
                end
                S_MERGE: begin
                    // One candidate a cycle into the exponential, its row kept.
                    if (m < k && have) begin
                        if (m == 0) lmax <= cand_lg;
                        diff = (m == 0) ? 33'sd0 : ($signed({lmax[31], lmax}) - $signed({cand_lg[31], cand_lg}));
                        tt = $unsigned(diff[31:0]) * inv_t;               // diff >= 0: the lists are descending
                        t <= (tt[47:FF] >= 48'h3FFFFF) ? 22'h3FFFFF : tt[FF +: 22];
                        ev <= 1'b1;
                        mrow[m] <= cand_row;
                        if (sel1) p1 <= p1 + 1'b1; else p0 <= p0 + 1'b1;
                        m <= m + 1'b1;
                    end else begin
                        k <= m;                                            // fewer candidates than asked
                        state <= S_DRAIN;
                    end
                end
                S_DRAIN: if (w_i == k) begin                              // the last weight is in
                    if (total == 0) begin
                        out_row <= mrow[0]; out_index <= 8'd0; out_valid <= 1'b1; state <= S_IDLE;
                    end else begin
                        pcut <= top_p * total;
                        keep <= k; scan <= 0;
                        state <= (top_p == 16'hFFFF) ? S_DRAW : S_CUT;
                    end
                end
                S_CUT: begin
                    // The shortest prefix whose mass reaches top_p of the whole.
                    if ({cum[scan], 16'd0} >= pcut) begin keep <= scan + 1'b1; state <= S_DRAW; end
                    else if (scan + 1 == k) begin keep <= k; state <= S_DRAW; end
                    else scan <= scan + 1'b1;
                end
                S_DRAW: begin
                    prod <= rnd * cum[keep - 1];
                    scan <= 0;
                    state <= S_OUT;
                end
                S_OUT: begin
                    // The first candidate whose cumulative weight exceeds the draw.
                    if (cum[scan] > prod[63:32] || scan + 1 == keep) begin
                        out_row <= mrow[scan]; out_index <= scan; out_valid <= 1'b1; state <= S_IDLE;
                    end else scan <= scan + 1'b1;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule

// ---------------------------------------------------------------------------
// CRC-32 over a stream of little-endian words, the Ethernet polynomial in its
// reflected form as zlib computes it: bit-serial per byte, thirty-two steps
// unrolled into one cycle.  `bytes` says how many of the last word's bytes
// count; `crc` is the finished value after the last word.
// ---------------------------------------------------------------------------
module fabric_crc32 (
    input  wire        clk,
    input  wire        clear,
    input  wire        valid,
    input  wire [31:0] data,
    input  wire [2:0]  bytes,                    // 1 to 4 of data's bytes, low first
    output wire [31:0] crc
);
    reg  [31:0] state;
    integer b, i;
    // clear and a word in the same cycle: the word is taken, from the fresh
    // state, which is what a receiver framing on sop needs.
    reg [31:0] base;
    reg [31:0] cc;
    always @* begin
        base = clear ? 32'hFFFFFFFF : state;
        cc = base;
        for (b = 0; b < 4; b = b + 1)
            if (b < bytes) begin
                cc = cc ^ {24'd0, data[b*8 +: 8]};
                for (i = 0; i < 8; i = i + 1)
                    cc = (cc >> 1) ^ (32'hEDB88320 & {32{cc[0]}});
            end
    end
    always @(posedge clk) begin
        if (clear && !valid) state <= 32'hFFFFFFFF;
        else if (clear || valid) state <= cc;
    end
    assign crc = ~state;
endmodule

`default_nettype wire
