// The ring link: a packet between one die and the next, or between the
// controller and the first die.  Source-synchronous 32-bit words with
// ready/valid, the first word of a packet marked by `sop`, and the CRC-32 of
// everything before it as the last word.  Golden model: fabric/controller.py.
//
// The CRC is a trailer rather than a header field because this is a stream:
// a head die appending its list, and a sender whose payload is the 8 KB
// hidden vector, would otherwise have to hold the whole packet to fill in a
// header the receiver reads first.

`timescale 1ns/1ps
`default_nettype none

// ---------------------------------------------------------------------------
// A header and a payload stream out as a framed packet.
// ---------------------------------------------------------------------------
module fabric_ring_tx (
    input  wire        clk,
    input  wire        rst_n,
    // the packet to send: taken when hdr_ready, then its payload words
    input  wire        hdr_valid,
    output wire        hdr_ready,
    input  wire [7:0]  hdr_kind,
    input  wire [7:0]  hdr_flags,
    input  wire [15:0] hdr_context,
    input  wire [31:0] hdr_position,
    input  wire [15:0] hdr_length,      // payload bytes, a multiple of four
    input  wire        p_valid,
    output wire        p_ready,
    input  wire [31:0] p_data,
    // the link
    output wire        l_valid,
    output wire [31:0] l_data,
    output wire        l_sop,
    input  wire        l_ready
);
    localparam [2:0] S_IDLE = 0, S_W0 = 1, S_W1 = 2, S_W2 = 3, S_PAY = 4, S_CRC = 5;
    reg [2:0]  state;
    reg [7:0]  kind, flags;
    reg [15:0] context_r, length_r;
    reg [31:0] position_r;
    reg [15:0] left;                     // payload words still to send

    wire [31:0] crc;
    wire        take = l_valid && l_ready;
    fabric_crc32 u_crc (.clk(clk), .clear(state == S_IDLE), .valid(take && state != S_CRC),
                        .data(l_data), .bytes(3'd4), .crc(crc));

    assign hdr_ready = (state == S_IDLE);
    assign p_ready   = (state == S_PAY) && l_ready;
    assign l_valid   = (state != S_IDLE) && ((state != S_PAY) || p_valid);
    assign l_sop     = (state == S_W0);
    assign l_data    = (state == S_W0)  ? {context_r, flags, kind}
                     : (state == S_W1)  ? position_r
                     : (state == S_W2)  ? {16'd0, length_r}
                     : (state == S_PAY) ? p_data : crc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; left <= 0;
        end else case (state)
            S_IDLE: if (hdr_valid) begin
                kind <= hdr_kind; flags <= hdr_flags; context_r <= hdr_context;
                position_r <= hdr_position; length_r <= hdr_length;
                left <= hdr_length[15:2];
                state <= S_W0;
            end
            S_W0: if (take) state <= S_W1;
            S_W1: if (take) state <= S_W2;
            S_W2: if (take) state <= (left == 0) ? S_CRC : S_PAY;
            S_PAY: if (take) begin
                left <= left - 1'b1;
                if (left == 1) state <= S_CRC;
            end
            S_CRC: if (take) state <= S_IDLE;
            default: state <= S_IDLE;
        endcase
    end
endmodule

// ---------------------------------------------------------------------------
// A framed packet in, its header and payload out, and whether the CRC held.
// ---------------------------------------------------------------------------
module fabric_ring_rx (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        l_valid,
    input  wire [31:0] l_data,
    input  wire        l_sop,
    output wire        l_ready,
    input  wire        rx_ready,          // the consumer can take a word
    output reg         hdr_valid,         // a cycle, when the header is complete
    output reg  [7:0]  hdr_kind,
    output reg  [7:0]  hdr_flags,
    output reg  [15:0] hdr_context,
    output reg  [31:0] hdr_position,
    output reg  [15:0] hdr_length,
    output reg         p_valid,
    output reg  [31:0] p_data,
    output reg         p_last,
    output reg         done,              // a cycle, after the trailer
    output reg         ok                 // with done: the CRC held
);
    localparam [2:0] S_SOP = 0, S_W1 = 1, S_W2 = 2, S_PAY = 3, S_TRL = 4;
    reg [2:0]  state;
    reg [15:0] left;

    wire [31:0] crc;
    wire        take = l_valid && l_ready;
    assign l_ready = rx_ready;
    // A packet's first word restarts the CRC wherever the receiver had got to,
    // so a truncated packet cannot poison the next one.  The restart and the
    // word itself are the same cycle, which fabric_crc32 takes: clear wins,
    // so sop is fed by the state machine's own first step instead.
    wire sop_take = take && l_sop && (state == S_SOP);
    fabric_crc32 u_crc (.clk(clk), .clear(sop_take), .valid(take && state != S_TRL),
                        .data(l_data), .bytes(3'd4), .crc(crc));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_SOP; left <= 0; hdr_valid <= 1'b0; p_valid <= 1'b0; done <= 1'b0; ok <= 1'b0;
        end else begin
            hdr_valid <= 1'b0;
            p_valid <= 1'b0;
            done <= 1'b0;
            case (state)
                S_SOP: if (take && l_sop) begin
                    hdr_kind <= l_data[7:0]; hdr_flags <= l_data[15:8]; hdr_context <= l_data[31:16];
                    state <= S_W1;
                end
                S_W1: if (take) begin hdr_position <= l_data; state <= S_W2; end
                S_W2: if (take) begin
                    hdr_length <= l_data[15:0];
                    left <= l_data[15:2];
                    hdr_valid <= 1'b1;
                    state <= (l_data[15:2] == 0) ? S_TRL : S_PAY;
                end
                S_PAY: if (take) begin
                    p_valid <= 1'b1; p_data <= l_data; p_last <= (left == 1);
                    left <= left - 1'b1;
                    if (left == 1) state <= S_TRL;
                end
                S_TRL: if (take) begin
                    done <= 1'b1;
                    ok <= (l_data == crc);
                    state <= S_SOP;
                end
                default: state <= S_SOP;
            endcase
        end
    end
endmodule

`default_nettype wire
