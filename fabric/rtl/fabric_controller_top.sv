// The controller's datapath: a request from the soft side becomes a packet on
// the ring, and the packet that comes back becomes a token.
//
//   request (slot, position, flags, token, sampling parameters, random word)
//     -> the token's row of the embedding table, as the packet's payload
//     -> fabric_ring_tx
//   fabric_ring_rx
//     -> the hidden vector skipped, the head dies' lists parsed out
//     -> fabric_sampler
//     -> the token, with the slot it belongs to
//
// Which context goes next, and the table of slots, are the soft side's and
// are not here; this is what has to be gateware because it is a stream.
// Golden model: fabric/controller.py.

`timescale 1ns/1ps
`default_nettype none

module fabric_controller_top #(
    parameter int D  = 8,                 // hidden elements, int16
    parameter int K  = 32,                // rows a head die's list may hold
    parameter int KIND = 8'h57,
    parameter     LUT_DIR = "./"
) (
    input  wire        clk,
    input  wire        rst_n,
    // a request
    input  wire        req_valid,
    output wire        req_ready,
    input  wire [15:0] req_slot,
    input  wire [31:0] req_position,
    input  wire [7:0]  req_flags,
    input  wire [31:0] req_token,
    input  wire [15:0] req_inv_t,
    input  wire [7:0]  req_top_k,
    input  wire [15:0] req_top_p,
    input  wire [31:0] req_rnd,
    // the embedding table: word-addressed, the answer a cycle later
    output wire        emb_en,
    output wire [31:0] emb_addr,
    input  wire [31:0] emb_data,
    // the ring, out and back
    output wire        l_valid,
    output wire [31:0] l_data,
    output wire        l_sop,
    input  wire        l_ready,
    input  wire        r_valid,
    input  wire [31:0] r_data,
    input  wire        r_sop,
    output wire        r_ready,
    // the token
    output reg         tok_valid,
    output reg  [31:0] tok_row,
    output reg  [15:0] tok_slot,
    output reg  [7:0]  tok_index
);
    localparam int HW = D / 2;            // payload words of the hidden vector
    localparam int HWW = (HW > 1) ? $clog2(HW) + 1 : 2;

    // ---------------------------------------------------------------------
    // Out: the embedding row as the payload.  One word every other cycle,
    // since the table answers a cycle after its address; a real part is read
    // in bursts and this is where that would go.
    // ---------------------------------------------------------------------
    reg          sending, have;
    reg [HWW-1:0] idx;
    reg [31:0]   word;
    reg [31:0]   base;
    reg [15:0]   slot_q;
    reg [7:0]    flags_q;
    reg [15:0]   inv_t_q, top_p_q;
    reg [7:0]    top_k_q;
    reg [31:0]   rnd_q;

    wire hdr_ready, p_ready;
    wire hdr_go = req_valid && req_ready;

    assign req_ready = !sending && hdr_ready;
    assign emb_en    = sending && !have;
    assign emb_addr  = base + {{(32-HWW){1'b0}}, idx};

    fabric_ring_tx u_tx (
        .clk(clk), .rst_n(rst_n),
        .hdr_valid(hdr_go), .hdr_ready(hdr_ready), .hdr_kind(KIND[7:0]), .hdr_flags(req_flags),
        .hdr_context(req_slot), .hdr_position(req_position), .hdr_length(16'(D * 2)),
        .p_valid(have), .p_ready(p_ready), .p_data(word),
        .l_valid(l_valid), .l_data(l_data), .l_sop(l_sop), .l_ready(l_ready));

    reg emb_q;                                              // a read was issued last cycle
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sending <= 1'b0; have <= 1'b0; idx <= 0; emb_q <= 1'b0;
        end else begin
            emb_q <= emb_en;
            if (hdr_go) begin
                sending <= 1'b1; have <= 1'b0; idx <= 0;
                base <= req_token * HW;
                slot_q <= req_slot; flags_q <= req_flags;
                inv_t_q <= req_inv_t; top_k_q <= req_top_k; top_p_q <= req_top_p; rnd_q <= req_rnd;
            end else begin
                // The table answers the cycle after its address, so the word
                // is taken then, not when the read was issued.
                if (emb_q) begin word <= emb_data; have <= 1'b1; end
                if (have && p_ready) begin
                    have <= 1'b0;
                    idx <= idx + 1'b1;
                    if (idx == HW - 1) sending <= 1'b0;
                end
            end
        end
    end

    // ---------------------------------------------------------------------
    // Back: the hidden vector skipped, the lists parsed, the token drawn.
    // A list is a header word pair (die, k, then the log-sum-exp) and k pairs
    // of (row, logit), which is what controller.pack_head_list writes.
    // ---------------------------------------------------------------------
    wire        rh_valid, rp_valid, rp_last, r_done, r_ok;
    wire [7:0]  rh_flags;
    wire [15:0] rh_context, rh_length;
    wire [31:0] rp_data;
    fabric_ring_rx u_rx (
        .clk(clk), .rst_n(rst_n), .l_valid(r_valid), .l_data(r_data), .l_sop(r_sop), .l_ready(r_ready),
        .rx_ready(1'b1), .hdr_valid(rh_valid), .hdr_kind(), .hdr_flags(rh_flags),
        .hdr_context(rh_context), .hdr_position(), .hdr_length(rh_length),
        .p_valid(rp_valid), .p_data(rp_data), .p_last(rp_last), .done(r_done), .ok(r_ok));

    localparam [2:0] P_SKIP = 0, P_LH0 = 1, P_LH1 = 2, P_ROW = 3, P_LOG = 4;
    reg [2:0]  pstate;
    reg [15:0] pw;
    reg [7:0]  lk, lseen;
    reg        ldie;
    reg [31:0] erow;
    reg [15:0] reply_slot;
    reg        reply_sample, reply_ok;

    reg               s_clear, s_in_valid, s_in_list, s_finish;
    reg [31:0]        s_in_row;
    reg signed [31:0] s_in_logit;
    wire              s_out_valid;
    wire [31:0]       s_out_row;
    wire [7:0]        s_out_index;
    fabric_sampler #(.K(K), .LUT_DIR(LUT_DIR)) u_sampler (
        .clk(clk), .rst_n(rst_n), .clear(s_clear), .in_valid(s_in_valid), .in_list(s_in_list),
        .in_row(s_in_row), .in_logit(s_in_logit), .finish(s_finish),
        .inv_t(inv_t_q), .top_k(top_k_q), .top_p(top_p_q), .rnd(rnd_q),
        .out_valid(s_out_valid), .out_row(s_out_row), .out_index(s_out_index));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pstate <= P_SKIP; pw <= 0; s_clear <= 1'b0; s_in_valid <= 1'b0; s_finish <= 1'b0;
            tok_valid <= 1'b0; reply_sample <= 1'b0;
        end else begin
            s_clear <= 1'b0;
            s_in_valid <= 1'b0;
            s_finish <= 1'b0;
            tok_valid <= 1'b0;
            if (r_valid && r_ready && r_sop) begin
                pstate <= P_SKIP; pw <= 0; s_clear <= 1'b1; lseen <= 0;
            end
            if (rh_valid) begin
                reply_slot <= rh_context;
                reply_sample <= rh_flags[0];                 // FLAG_SAMPLE
            end
            if (rp_valid) begin
                pw <= pw + 1'b1;
                case (pstate)
                    P_SKIP: if (pw == HW - 1) pstate <= P_LH0;
                    P_LH0: begin
                        ldie <= rp_data[0];
                        lk <= rp_data[15:8];
                        lseen <= 0;
                        pstate <= P_LH1;
                    end
                    P_LH1: pstate <= (lk == 0) ? P_LH0 : P_ROW;   // the log-sum-exp: the merge needs it, the draw does not
                    P_ROW: begin erow <= rp_data; pstate <= P_LOG; end
                    P_LOG: begin
                        s_in_valid <= 1'b1; s_in_list <= ldie; s_in_row <= erow; s_in_logit <= rp_data;
                        lseen <= lseen + 1'b1;
                        pstate <= (lseen + 1 == lk) ? P_LH0 : P_ROW;
                    end
                    default: pstate <= P_LH0;
                endcase
            end
            if (r_done) begin
                reply_ok <= r_ok;
                if (r_ok && reply_sample) s_finish <= 1'b1;
            end
            if (s_out_valid) begin
                tok_valid <= 1'b1; tok_row <= s_out_row; tok_index <= s_out_index; tok_slot <= reply_slot;
            end
        end
    end
endmodule

`default_nettype wire
