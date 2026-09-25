// A layer die's front end: the ring in, the engine, the ring out.
//
// Packets arrive from the die before (or the controller) on fabric_ring_rx.
// Each is a work item -- one token, or a chunk of a prompt's tokens
// (fabric/controller.py) -- and takes a lane: its hidden vectors are written
// into the engine's vector buffer at that lane's input, and its slot, flags,
// position and token count are kept.  The engine runs a program over up to
// LANES packets at once, the tokens of different contexts interleaved (the
// sequencer's streams), so the lanes are gathered into a batch, and the
// batch runs when it is full, when a packet that cannot join it arrives, or
// when the link has been quiet for GATHER_WAIT cycles.  Then each lane's
// packet goes on to the next die with its output vectors in place of the
// input, the header otherwise as it came.
//
// What the engine is given, per batch:
//   * the program: its first step and length, from a table indexed by the
//     batch's shape -- a token a lane or a chunk a lane, and how many lanes
//     -- since the dies hold a program for each (fabric/engine.py lays
//     each one out, and the table carries where each lane's input and
//     output are in the vector buffer for that program);
//   * each lane's slot as a page of the die's memory: the packet's context
//     field is the controller's slot number, and a slot is SLOT_PAGES pages
//     from PAGE_BASE on;
//   * FIRST per lane, from the packet's flag.
//
// The vector buffer is the engine's: this unit uses the memory unit's ports
// while the engine is idle (`v_sel`), so a packet's payload waits on the
// link while a batch runs -- a few thousand cycles against a few hundred
// thousand a token spends in the die's layers.
//
// A packet whose CRC fails takes no lane: its vectors are overwritten by the
// next.  A packet that is not a work item of a shape the die has a program
// for is read off the link and dropped.  Both are counted.

`timescale 1ns/1ps
`default_nettype none

module fabric_die_link #(
    parameter int D           = 8,              // hidden elements a token, int16; a multiple of eight
    parameter int CHUNK       = 1,              // the chunk the die has programs for, besides a single token
    parameter int LANES       = 4,              // packets a batch, at most four
    parameter int SLOT_PAGES  = 16,             // a slot's pages of the die's memory
    parameter int PAGE_BASE   = 0,
    parameter int GATHER_WAIT = 16,             // quiet cycles before a part-full batch runs
    parameter int AW          = 24,             // vector-buffer address bits
    parameter int KIND        = 8'h57,
    parameter     TABLE_FILE  = "die_table.hex"
) (
    input  wire              clk,
    input  wire              rst_n,
    // the ring in
    input  wire              u_valid,
    input  wire [31:0]       u_data,
    input  wire              u_sop,
    output wire              u_ready,
    // the ring out
    output wire              d_valid,
    output wire [31:0]       d_data,
    output wire              d_sop,
    input  wire              d_ready,
    // the engine
    output reg               e_start,
    output reg  [15:0]       e_pc,
    output reg  [15:0]       e_steps,
    output reg  [3:0]        e_first,
    output reg  [4*21-1:0]   e_slot_page,
    input  wire              e_done,
    // the vector buffer, while the engine is idle
    output wire              v_sel,
    output reg               v_wr_en,
    output reg  [AW-1:0]     v_wr_addr,
    output reg  [127:0]      v_wr_data,
    output wire              v_rd_en,
    output wire [AW-1:0]     v_rd_addr,
    input  wire [127:0]      v_rd_data,         // the cycle after its address
    // what was dropped
    output reg  [15:0]       crc_errors,
    output reg  [15:0]       malformed
);
    localparam int TW = D / 2;                  // payload words a token
    localparam int TB = D / 8;                  // vector-buffer beats a token

    // ---------------------------------------------------------------------
    // The program table: an entry a batch shape, {chunk, lanes - 1}.
    //   [15:0] first step   [31:16] steps
    //   [32 + 24k +: 24] lane k's input   [128 + 24k +: 24] lane k's output
    // ---------------------------------------------------------------------
    reg [255:0] table_mem [0:7];
    initial if (TABLE_FILE != "") $readmemh(TABLE_FILE, table_mem);
    reg [255:0] entry;

    // ---------------------------------------------------------------------
    // The lanes.
    // ---------------------------------------------------------------------
    reg [2:0]  lanes;                           // lanes gathered
    reg        cls;                             // the batch's shape: a chunk a lane
    reg [7:0]  l_flags   [0:3];
    reg [15:0] l_context [0:3];
    reg [31:0] l_position[0:3];
    reg [7:0]  l_tokens  [0:3];
    reg [7:0]  h_flags, h_tokens;               // the packet coming in, until it takes its lane
    reg [15:0] h_context;
    reg [31:0] h_position;

    // ---------------------------------------------------------------------
    // In.
    // ---------------------------------------------------------------------
    wire        rh_valid, rp_valid, rp_last, r_done, r_ok;
    wire [7:0]  rh_kind, rh_flags, rh_tokens;
    wire [15:0] rh_context;
    wire [31:0] rh_position, rp_data;
    wire [23:0] rh_length;
    reg         rx_ready;
    fabric_ring_rx u_rx (
        .clk(clk), .rst_n(rst_n), .l_valid(u_valid), .l_data(u_data), .l_sop(u_sop), .l_ready(u_ready),
        .rx_ready(rx_ready), .hdr_valid(rh_valid), .hdr_kind(rh_kind), .hdr_flags(rh_flags),
        .hdr_context(rh_context), .hdr_position(rh_position), .hdr_tokens(rh_tokens), .hdr_length(rh_length),
        .p_valid(rp_valid), .p_data(rp_data), .p_last(rp_last), .done(r_done), .ok(r_ok));

    localparam [1:0] R_HDR = 0, R_PAY = 1, R_DROP = 2, R_HOLD = 3;
    localparam [2:0] X_IDLE = 0, X_LOOK = 1, X_START = 2, X_RUN = 3, X_THDR = 4, X_TPAY = 5, X_TEND = 6;
    reg [1:0]  rstate;
    reg [2:0]  xstate;
    reg [1:0]  wsub;                            // a word's place in its beat
    reg [AW-1:0] waddr;
    reg [15:0] quiet;

    // A header the die has a program for: a work item of one token or of the
    // chunk, its length exactly the vectors.
    wire        shape_ok = (rh_kind == KIND[7:0]) && (rh_tokens == 8'd1 || rh_tokens == CHUNK[7:0])
                           && (rh_length == 24'(rh_tokens) * 24'(2 * D));
    wire        rh_cls   = (rh_tokens != 8'd1);
    // It joins the batch being gathered: the engine is idle, there is room,
    // and it is the batch's shape (or the batch is empty).
    wire        joins    = (xstate == X_IDLE) && (lanes < LANES) && (lanes == 0 || rh_cls == cls);

    // The payload is read off the link only into a lane or to be dropped; a
    // header is always taken, and its payload held there until it can be.
    always @* begin
        case (rstate)
            R_HDR:   rx_ready = !rh_valid || !shape_ok || joins;
            R_PAY:   rx_ready = 1'b1;
            R_DROP:  rx_ready = 1'b1;
            default: rx_ready = 1'b0;
        endcase
    end

    // ---------------------------------------------------------------------
    // Out.
    // ---------------------------------------------------------------------
    reg         t_hdr;
    reg  [2:0]  tlane;
    reg  [23:0] tleft;                          // payload words of the lane still to send
    reg  [23:0] tfetch;                         // and its beats still to read
    reg  [AW-1:0] raddr;
    reg         rbusy;                          // a beat is on its way from the buffer
    reg         have;                           // `beat` holds words not yet sent
    reg  [127:0] beat;
    reg  [1:0]  tsub;
    wire        t_hdr_ready, t_p_ready;
    fabric_ring_tx u_tx (
        .clk(clk), .rst_n(rst_n),
        .hdr_valid(t_hdr), .hdr_ready(t_hdr_ready), .hdr_kind(KIND[7:0]), .hdr_flags(l_flags[tlane[1:0]]),
        .hdr_context(l_context[tlane[1:0]]), .hdr_position(l_position[tlane[1:0]]), .hdr_tokens(l_tokens[tlane[1:0]]),
        .hdr_length(24'(l_tokens[tlane[1:0]]) * 24'(2 * D)),
        .p_valid(have), .p_ready(t_p_ready), .p_data(beat[32*tsub +: 32]),
        .l_valid(d_valid), .l_data(d_data), .l_sop(d_sop), .l_ready(d_ready));

    // A beat is fetched when the last one's words are all but gone, so the
    // link sees a word a cycle.
    wire        take_word = have && t_p_ready;
    wire        refill    = (xstate == X_TPAY) && !rbusy && (tfetch != 0) && (!have || (take_word && tsub == 2'd3));
    assign v_rd_en   = refill;
    assign v_rd_addr = raddr;
    assign v_sel     = (xstate == X_IDLE) || (xstate == X_THDR) || (xstate == X_TPAY) || (xstate == X_TEND);

    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rstate <= R_HDR; xstate <= X_IDLE; lanes <= 0; cls <= 1'b0; wsub <= 0; quiet <= 0;
            v_wr_en <= 1'b0; e_start <= 1'b0; e_pc <= 0; e_steps <= 0; e_first <= 0; e_slot_page <= 0;
            t_hdr <= 1'b0; tlane <= 0; tleft <= 0; tfetch <= 0; rbusy <= 1'b0; have <= 1'b0; tsub <= 0;
            crc_errors <= 0; malformed <= 0;
        end else begin
            v_wr_en <= 1'b0;
            e_start <= 1'b0;

            // ---- in ----
            case (rstate)
                R_HDR: if (rh_valid) begin
                    if (!shape_ok) begin
                        malformed <= malformed + 1'b1;
                        rstate <= R_DROP;
                    end else begin
                        h_flags <= rh_flags; h_context <= rh_context; h_position <= rh_position; h_tokens <= rh_tokens;
                        wsub <= 0;
                        if (joins) begin
                            if (lanes == 0) cls <= rh_cls;
                            waddr <= entry_in(lanes, rh_cls);
                            rstate <= R_PAY;
                        end else
                            rstate <= R_HOLD;               // its payload waits on the link for the next batch
                    end
                end
                R_PAY: begin
                    if (rp_valid) begin
                        v_wr_data[32*wsub +: 32] <= rp_data;
                        wsub <= wsub + 1'b1;
                        if (wsub == 2'd3) begin
                            v_wr_en <= 1'b1; v_wr_addr <= waddr;
                            waddr <= waddr + 16;
                        end
                    end
                    if (r_done) begin
                        if (r_ok) begin
                            l_flags[lanes[1:0]] <= h_flags; l_context[lanes[1:0]] <= h_context;
                            l_position[lanes[1:0]] <= h_position; l_tokens[lanes[1:0]] <= h_tokens;
                            lanes <= lanes + 1'b1;
                        end else
                            crc_errors <= crc_errors + 1'b1;
                        rstate <= R_HDR;
                    end
                end
                R_DROP: if (r_done) rstate <= R_HDR;
                R_HOLD: ;                                   // released below, when a batch can take it
                default: rstate <= R_HDR;
            endcase

            // ---- the batch ----
            case (xstate)
                X_IDLE: begin
                    quiet <= (rstate == R_HDR && !rh_valid && !u_valid) ? quiet + 1'b1 : 16'd0;
                    if (rstate == R_HOLD && lanes == 0) begin
                        // The held packet starts the next batch.
                        cls <= (h_tokens != 8'd1);
                        waddr <= entry_in(3'd0, h_tokens != 8'd1);
                        rstate <= R_PAY;
                    end else if (lanes != 0 && ((rstate == R_HOLD) ||
                                                (rstate == R_HDR && !rh_valid && (lanes == LANES || quiet >= GATHER_WAIT)))) begin
                        entry <= table_mem[{cls, 2'(lanes - 1)}];
                        xstate <= X_LOOK;
                    end
                end
                X_LOOK: begin
                    e_pc <= entry[15:0]; e_steps <= entry[31:16];
                    for (k = 0; k < 4; k = k + 1) begin
                        e_first[k] <= (k < lanes) && l_flags[k][1];                // FLAG_FIRST
                        e_slot_page[21*k +: 21] <= 21'(PAGE_BASE) + ((k < lanes) ? 21'(l_context[k]) * 21'(SLOT_PAGES) : 21'd0);
                    end
                    xstate <= X_START;
                end
                X_START: begin e_start <= 1'b1; xstate <= X_RUN; end
                X_RUN: if (e_done) begin tlane <= 0; t_hdr <= 1'b1; xstate <= X_THDR; end
                X_THDR: if (t_hdr_ready) begin
                    // The tx takes the header this cycle; the lane's words follow.
                    t_hdr <= 1'b0;
                    tleft <= 24'(l_tokens[tlane[1:0]]) * 24'(TW);
                    tfetch <= 24'(l_tokens[tlane[1:0]]) * 24'(TB);
                    raddr <= entry[128 + 24*tlane +: 24];
                    have <= 1'b0; rbusy <= 1'b0; tsub <= 0;
                    xstate <= X_TPAY;
                end
                X_TPAY: begin
                    if (refill) begin rbusy <= 1'b1; raddr <= raddr + 16; tfetch <= tfetch - 1'b1; end
                    if (take_word) begin
                        tsub <= tsub + 1'b1;
                        tleft <= tleft - 1'b1;
                        if (tsub == 2'd3) have <= 1'b0;
                    end
                    if (rbusy) begin beat <= v_rd_data; have <= 1'b1; rbusy <= 1'b0; end
                    if (tleft == 1 && take_word) xstate <= X_TEND;
                end
                X_TEND: if (t_hdr_ready) begin                // the tx has sent the CRC
                    if (tlane + 1 == lanes) begin
                        lanes <= 0; xstate <= X_IDLE;
                    end else begin
                        tlane <= tlane + 1'b1; t_hdr <= 1'b1; xstate <= X_THDR;
                    end
                end
                default: xstate <= X_IDLE;
            endcase
        end
    end

    // Lane k's input for a batch of the given shape.  The table's entries of
    // one shape agree on a lane's input whatever the batch's size, since a
    // lane is written before the batch's size is known: fabric/engine.py
    // lays the programs out so.
    function automatic [AW-1:0] entry_in(input [2:0] k, input c);
        reg [255:0] e;
        begin
            e = table_mem[{c, 2'(LANES - 1)}];
            entry_in = e[32 + 24*k +: AW];
        end
    endfunction
endmodule

`default_nettype wire
