// A layer die's front end: the ring in, the engine's lanes, the ring out.
//
// Packets arrive from the die before (or the controller) on fabric_ring_rx.
// Each is a work item -- one token, or a chunk of a prompt's tokens
// (fabric/controller.py) -- and takes a lane of the engine as it arrives:
// its hidden vectors are written into the vector buffer at that lane's
// input, and its slot, flags, position and token count are kept.  The engine
// runs its lanes at once, each lane's steps issuing as they are ready
// (fabric_sequencer), so there is nothing to gather: a packet waits only for
// a free lane, and for a lane that holds its own context to finish -- two
// tokens of one context cannot run at once on its state, and a prompt's
// packets come back to back from one slot.  When a lane is done its packet
// goes on to the next die with its output vectors in place of the input,
// the header otherwise as it came, the lanes' packets in the order they
// finished: another context's may overtake, a context's own never do.
//
// What the engine is given, per packet: a run for each of the die's layers,
// pushed to the packet's lane one a cycle --
//   * the program: its first step and length, from a table indexed by the
//     layer's kind and the packet's shape (a token or a chunk), since every
//     lane's store holds the die's programs at the same steps
//     (fabric/engine.py lays each lane's out, and the table carries where
//     each lane's vectors are in the vector buffer);
//   * the layer, and where its part of the slot starts (the layer table);
// and with them the lane's token: its slot as a page of the die's memory --
// the packet's context field is the controller's slot number, and a slot is
// SLOT_PAGES pages from PAGE_BASE on -- its FIRST and its position.
//
// The link has ports of its own on the vector buffer, a write and a read:
// a packet is written into a lane that is not running and read out of one
// that is done while the other lanes run, and a lane's buffers are in banks
// of its own, so nothing the engine does meets them.
//
// A packet whose CRC fails gives its lane back: its vectors are overwritten
// by the next.  A packet that is not a work item of a shape the die has a
// program for is read off the link and dropped.  Both are counted.

`timescale 1ns/1ps
`default_nettype none

module fabric_die_link #(
    parameter int D           = 8,              // hidden elements a token, int16; a multiple of eight
    parameter int CHUNK       = 1,              // the chunk the die has programs for, besides a single token
    parameter int LANES       = 4,              // the engine's lanes, at most four
    parameter int SLOT_PAGES  = 16,             // a slot's pages of the die's memory
    parameter int PAGE_BASE   = 0,
    parameter int AW          = 24,             // vector-buffer address bits
    parameter int KIND        = 8'h57,
    parameter int LAYERS      = 1,              // layers a token takes on this die, a run each: four on a layer die
    parameter     TABLE_FILE  = "die_table.hex",
    parameter     LAYER_FILE  = ""              // each layer's kind and its part of a slot (below); none: one layer at the slot's start
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
    // the engine's lanes: a run pushed to one, and each one's done
    output reg               e_push,
    output reg  [1:0]        e_lane,
    output reg  [15:0]       e_pc,
    output reg  [15:0]       e_steps,
    output reg  [1:0]        e_layer,
    output reg  [20:0]       e_page,
    output reg  [3:0]        e_set,
    output reg  [3:0]        e_first,
    output reg  [4*21-1:0]   e_slot_page,
    output reg  [4*32-1:0]   e_position,
    input  wire [3:0]        e_done,
    // the link's own ports on the vector buffer
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
    // The program table: an entry a layer kind and packet shape, {kind,
    // chunk}.  A kind is a program: a die's three recurrent layers run one
    // program and its global layer the other, each layer on its own bank of
    // weights and constants (the engine's layer) and its own part of the
    // slot (the layer table).  A lane's vectors are in place: its input and
    // its output are one buffer, whatever the layer.
    //   [15:0] first step   [31:16] steps
    //   [32 + 24k +: 24] lane k's input   [128 + 24k +: 24] lane k's output
    // ---------------------------------------------------------------------
    reg [255:0] table_mem [0:3];
    initial if (TABLE_FILE != "") $readmemh(TABLE_FILE, table_mem);
    // The layer table: [0] the layer's kind, [21:1] the page its part of a
    // slot starts at, from the slot's first.
    reg [31:0]  layer_mem [0:3];
    integer     li;
    initial begin
        for (li = 0; li < 4; li = li + 1) layer_mem[li] = 0;
        if (LAYER_FILE != "") $readmemh(LAYER_FILE, layer_mem);
    end

    // ---------------------------------------------------------------------
    // The lanes.
    // ---------------------------------------------------------------------
    localparam [2:0] L_FREE = 0, L_LOAD = 1, L_PUSH = 2, L_RUN = 3, L_OUT = 4;
    reg [2:0]  l_st      [0:3];
    reg        l_cls     [0:3];                 // a chunk
    reg [7:0]  l_flags   [0:3];
    reg [15:0] l_context [0:3];
    reg [31:0] l_position[0:3];
    reg [7:0]  l_tokens  [0:3];
    reg [15:0] l_seq     [0:3];                 // the order the lanes finished in
    reg [15:0] seq;
    reg [7:0]  h_flags, h_tokens;               // the packet coming in
    reg [15:0] h_context;
    reg [31:0] h_position;
    reg        h_cls;
    reg [1:0]  rlane;                           // the lane it is going into

    // A free lane, the lowest; and whether a context is in a lane that has
    // not finished with its state.
    reg        any_free;
    reg [1:0]  free_lane;
    integer    fk;
    always @* begin
        any_free = 1'b0; free_lane = 0;
        for (fk = LANES - 1; fk >= 0; fk = fk - 1)
            if (l_st[fk] == L_FREE) begin any_free = 1'b1; free_lane = 2'(fk); end
    end

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
    localparam [1:0] X_IDLE = 0, X_THDR = 1, X_TPAY = 2, X_TEND = 3;
    reg [1:0]  rstate, xstate;
    reg [1:0]  wsub;                            // a word's place in its beat
    reg [AW-1:0] waddr;

    // A header the die has a program for: a work item of one token or of the
    // chunk, its length exactly the vectors.
    wire        shape_ok = (rh_kind == KIND[7:0]) && (rh_tokens == 8'd1 || rh_tokens == CHUNK[7:0])
                           && (rh_length == 24'(rh_tokens) * 24'(2 * D));
    wire        rh_cls   = (rh_tokens != 8'd1);
    // Written out here rather than as a function of the context: what a
    // function reads is not in an always block's sensitivity, only what it
    // is passed, and a stale answer here holds a packet for good.
    reg         rh_busy, h_busy;
    integer     bk;
    always @* begin
        rh_busy = 1'b0; h_busy = 1'b0;
        for (bk = 0; bk < LANES; bk = bk + 1)
            if (l_st[bk] == L_LOAD || l_st[bk] == L_PUSH || l_st[bk] == L_RUN) begin
                if (l_context[bk] == rh_context) rh_busy = 1'b1;
                if (l_context[bk] == h_context)  h_busy  = 1'b1;
            end
    end
    wire        takes    = any_free && !rh_busy;
    wire        h_takes  = any_free && !h_busy;

    // The payload is read off the link only into a lane or to be dropped; a
    // header is always taken, and its payload held there until it can be.
    always @* begin
        case (rstate)
            R_HDR:   rx_ready = !rh_valid || !shape_ok || takes;
            R_PAY:   rx_ready = 1'b1;
            R_DROP:  rx_ready = 1'b1;
            default: rx_ready = 1'b0;
        endcase
    end

    // ---------------------------------------------------------------------
    // Out.
    // ---------------------------------------------------------------------
    reg         t_hdr;
    reg  [1:0]  tlane;
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
        .hdr_valid(t_hdr), .hdr_ready(t_hdr_ready), .hdr_kind(KIND[7:0]), .hdr_flags(l_flags[tlane]),
        .hdr_context(l_context[tlane]), .hdr_position(l_position[tlane]), .hdr_tokens(l_tokens[tlane]),
        .hdr_length(24'(l_tokens[tlane]) * 24'(2 * D)),
        .p_valid(have), .p_ready(t_p_ready), .p_data(beat[32*tsub +: 32]),
        .l_valid(d_valid), .l_data(d_data), .l_sop(d_sop), .l_ready(d_ready));

    // A beat is fetched when the last one's words are all but gone, so the
    // link sees a word a cycle.
    wire        take_word = have && t_p_ready;
    wire        refill    = (xstate == X_TPAY) && !rbusy && (tfetch != 0) && (!have || (take_word && tsub == 2'd3));
    assign v_rd_en   = refill;
    assign v_rd_addr = raddr;

    // The lane done longest: the next to leave.
    reg        any_out;
    reg [1:0]  out_lane;
    integer    ok_;
    always @* begin
        any_out = 1'b0; out_lane = 0;
        for (ok_ = 0; ok_ < LANES; ok_ = ok_ + 1)
            if (l_st[ok_] == L_OUT && (!any_out || l_seq[ok_] < l_seq[out_lane])) begin any_out = 1'b1; out_lane = 2'(ok_); end
    end

    // ---------------------------------------------------------------------
    // The pushes: a lane whose packet is in gets its layers, a run a cycle.
    // Its queue is empty -- it was free -- and holds a token's layers.
    // ---------------------------------------------------------------------
    reg        pbusy;
    reg [1:0]  plane, play;
    reg        any_push;
    reg [1:0]  push_lane;
    integer    pk;
    always @* begin
        any_push = 1'b0; push_lane = 0;
        for (pk = LANES - 1; pk >= 0; pk = pk - 1)
            if (l_st[pk] == L_PUSH) begin any_push = 1'b1; push_lane = 2'(pk); end
    end
    wire [255:0] p_entry = table_mem[{layer_mem[play][0], l_cls[plane]}];

    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rstate <= R_HDR; xstate <= X_IDLE; wsub <= 0; seq <= 0; pbusy <= 1'b0; plane <= 0; play <= 0;
            v_wr_en <= 1'b0; e_push <= 1'b0; e_lane <= 0; e_pc <= 0; e_steps <= 0; e_layer <= 0; e_page <= 0; e_set <= 0;
            e_first <= 0; e_slot_page <= 0; e_position <= 0;
            t_hdr <= 1'b0; tlane <= 0; tleft <= 0; tfetch <= 0; rbusy <= 1'b0; have <= 1'b0; tsub <= 0;
            crc_errors <= 0; malformed <= 0;
            for (k = 0; k < 4; k = k + 1) begin l_st[k] <= L_FREE; l_seq[k] <= 0; l_context[k] <= 0; end
        end else begin
            v_wr_en <= 1'b0;
            e_push <= 1'b0;

            // ---- in ----
            case (rstate)
                R_HDR: if (rh_valid) begin
                    if (!shape_ok) begin
                        malformed <= malformed + 1'b1;
                        rstate <= R_DROP;
                    end else begin
                        h_flags <= rh_flags; h_context <= rh_context; h_position <= rh_position; h_tokens <= rh_tokens; h_cls <= rh_cls;
                        wsub <= 0;
                        if (takes) begin
                            rlane <= free_lane; l_st[free_lane] <= L_LOAD; l_context[free_lane] <= rh_context;
                            waddr <= table_mem[{layer_mem[0][0], rh_cls}][32 + 24*free_lane +: AW];
                            rstate <= R_PAY;
                        end else
                            rstate <= R_HOLD;               // its payload waits on the link for a lane
                    end
                end
                R_HOLD: if (h_takes) begin
                    rlane <= free_lane; l_st[free_lane] <= L_LOAD; l_context[free_lane] <= h_context;
                    waddr <= table_mem[{layer_mem[0][0], h_cls}][32 + 24*free_lane +: AW];
                    rstate <= R_PAY;
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
                            l_flags[rlane] <= h_flags; l_position[rlane] <= h_position; l_tokens[rlane] <= h_tokens; l_cls[rlane] <= h_cls;
                            l_st[rlane] <= L_PUSH;
                        end else begin
                            crc_errors <= crc_errors + 1'b1;
                            l_st[rlane] <= L_FREE;
                        end
                        rstate <= R_HDR;
                    end
                end
                R_DROP: if (r_done) rstate <= R_HDR;
                default: rstate <= R_HDR;
            endcase

            // ---- the pushes ----
            if (!pbusy) begin
                if (any_push) begin plane <= push_lane; play <= 0; pbusy <= 1'b1; end
            end else begin
                e_push <= 1'b1; e_lane <= plane;
                e_pc <= p_entry[15:0]; e_steps <= p_entry[31:16]; e_layer <= play; e_page <= layer_mem[play][21:1];
                e_set <= 4'b0001 << plane;
                for (k = 0; k < 4; k = k + 1) begin
                    e_first[k] <= l_flags[plane][1];                                   // FLAG_FIRST
                    e_slot_page[21*k +: 21] <= 21'(PAGE_BASE) + 21'(l_context[plane]) * 21'(SLOT_PAGES);
                    e_position[32*k +: 32] <= l_position[plane];
                end
                if (play == 2'(LAYERS - 1)) begin l_st[plane] <= L_RUN; pbusy <= 1'b0; end
                else play <= play + 1'b1;
            end

            // ---- done ----
            for (k = 0; k < LANES; k = k + 1)
                if (e_done[k] && l_st[k] == L_RUN) begin l_st[k] <= L_OUT; l_seq[k] <= seq; end
            if (|e_done) seq <= seq + 1'b1;

            // ---- out ----
            case (xstate)
                X_IDLE: if (any_out) begin tlane <= out_lane; t_hdr <= 1'b1; xstate <= X_THDR; end
                X_THDR: if (t_hdr_ready) begin
                    // The tx takes the header this cycle; the lane's words follow.
                    t_hdr <= 1'b0;
                    tleft <= 24'(l_tokens[tlane]) * 24'(TW);
                    tfetch <= 24'(l_tokens[tlane]) * 24'(TB);
                    raddr <= table_mem[{layer_mem[LAYERS - 1][0], l_cls[tlane]}][128 + 24*tlane +: AW];
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
                    l_st[tlane] <= L_FREE; xstate <= X_IDLE;
                end
                default: xstate <= X_IDLE;
            endcase
        end
    end
`ifndef FABRIC_SYNTH
    initial if (LAYERS > 4) $display("FAIL: a lane's queue holds four runs; LAYERS is %0d", LAYERS);
`endif
endmodule

`default_nettype wire
