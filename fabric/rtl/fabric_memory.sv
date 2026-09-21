// The memory side of a layer: the units that use the die's one memory port.
//
// Memory port, one per requester:
//   req_valid / req_ready, req_write, req_addr[31:0] (byte address aligned
//   to a beat), req_beats[11:0] (up to 4095); write beats follow the request on
//   wdata_valid / wdata_ready / wdata; read beats return in order on
//   rdata_valid / rdata.  One request in flight per requester.
//
// Golden model: fabric/memory.py.

`default_nettype none
`include "fabric_fx.svh"

// ---------------------------------------------------------------------------
// Behavioural memory for the testbenches: WORDS beats, a read latency, a
// hex image in and the same array for the checks.  Not for synthesis.
// ---------------------------------------------------------------------------
module fabric_mem_model #(
    parameter int DW    = 128,
    parameter int WORDS = 4096,
    parameter int LAT   = 4,
    parameter     FILE  = ""
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          req_valid,
    output wire          req_ready,
    input  wire          req_write,
    input  wire [31:0]   req_addr,
    input  wire [11:0]   req_beats,
    input  wire          wdata_valid,
    output wire          wdata_ready,
    input  wire [DW-1:0] wdata,
    output reg           rdata_valid,
    output reg  [DW-1:0] rdata
);
    localparam int AW = $clog2(WORDS);
    reg [DW-1:0] mem [0:WORDS-1];
    initial begin
        if (FILE != "") $readmemh(FILE, mem);
    end
    reg        busy, is_write;
    reg [AW-1:0] addr;
    reg [11:0] left;
    reg [3:0]  wait_r;
    assign req_ready   = !busy;
    assign wdata_ready = busy && is_write;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0; rdata_valid <= 1'b0; is_write <= 1'b0; left <= 0; wait_r <= 0;
        end else begin
            rdata_valid <= 1'b0;
            if (!busy) begin
                if (req_valid) begin
                    busy <= 1'b1; is_write <= req_write; addr <= req_addr[AW+3:4]; left <= req_beats; wait_r <= LAT;
                end
            end else if (is_write) begin
                if (wdata_valid) begin
                    mem[addr] <= wdata;
                    addr <= addr + 1'b1;
                    left <= left - 1'b1;
                    if (left == 1) busy <= 1'b0;
                end
            end else begin
                if (wait_r != 0) wait_r <= wait_r - 1'b1;
                else begin
                    rdata_valid <= 1'b1;
                    rdata <= mem[addr];
                    addr <= addr + 1'b1;
                    left <= left - 1'b1;
                    if (left == 1) busy <= 1'b0;
                end
            end
        end
    end
endmodule

// ---------------------------------------------------------------------------
// N requesters onto one port, round robin, the port held for a whole
// transaction.  Read data is steered to the requester that owns it.
// ---------------------------------------------------------------------------
module fabric_mem_arbiter #(
    parameter int N  = 2,
    parameter int DW = 128
) (
    input  wire            clk,
    input  wire            rst_n,
    // requesters
    input  wire [N-1:0]    r_req_valid,
    output wire [N-1:0]    r_req_ready,
    input  wire [N-1:0]    r_req_write,
    input  wire [N*32-1:0] r_req_addr,
    input  wire [N*12-1:0] r_req_beats,
    input  wire [N-1:0]    r_wdata_valid,
    output wire [N-1:0]    r_wdata_ready,
    input  wire [N*DW-1:0] r_wdata,
    output wire [N-1:0]    r_rdata_valid,
    output wire [DW-1:0]   r_rdata,
    // memory
    output wire            m_req_valid,
    input  wire            m_req_ready,
    output wire            m_req_write,
    output wire [31:0]     m_req_addr,
    output wire [11:0]     m_req_beats,
    output wire            m_wdata_valid,
    input  wire            m_wdata_ready,
    output wire [DW-1:0]   m_wdata,
    input  wire            m_rdata_valid,
    input  wire [DW-1:0]   m_rdata
);
    localparam int IW = $clog2(N) + 1;
    reg          locked;
    reg [IW-1:0] owner, last;
    reg          own_write;
    reg [11:0]   left;
    // Pick the next requester after `last` with a request.
    integer i;
    reg [IW-1:0] pick;
    reg          found;
    always @* begin
        pick = 0; found = 1'b0;
        for (i = 1; i <= N; i = i + 1)
            if (!found && r_req_valid[(last + i) % N]) begin pick = (last + i) % N; found = 1'b1; end
    end
    wire grant = !locked && found && m_req_ready;
    assign m_req_valid = grant;
    assign m_req_write = r_req_write[pick];
    assign m_req_addr  = r_req_addr[pick*32 +: 32];
    assign m_req_beats = r_req_beats[pick*12 +: 12];
    genvar g;
    generate
        for (g = 0; g < N; g = g + 1) begin : g_r
            assign r_req_ready[g]   = grant && (pick == g);
            assign r_wdata_ready[g] = locked && own_write && (owner == g) && m_wdata_ready;
            assign r_rdata_valid[g] = locked && !own_write && (owner == g) && m_rdata_valid;
        end
    endgenerate
    assign m_wdata_valid = locked && own_write && r_wdata_valid[owner];
    assign m_wdata       = r_wdata[owner*DW +: DW];
    assign r_rdata       = m_rdata;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            locked <= 1'b0; owner <= 0; last <= N - 1; own_write <= 1'b0; left <= 0;
        end else begin
            if (grant) begin
                locked <= 1'b1; owner <= pick; last <= pick; own_write <= r_req_write[pick]; left <= r_req_beats[pick*12 +: 12];
            end else if (locked) begin
                if ((own_write && m_wdata_valid && m_wdata_ready) || (!own_write && m_rdata_valid)) begin
                    left <= left - 1'b1;
                    if (left == 1) locked <= 1'b0;
                end
            end
        end
    end
endmodule

// ---------------------------------------------------------------------------
// Rows of ROW_BITS between memory and a row-streaming unit such as the delta
// engine.  Read: from rd_start, ROWS rows from rd_base, each a burst of
// ROW_BITS/DW beats, presented whole on row_out.  Write: rows arriving on
// row_in queue in a FIFO of ROWS entries and are written from wr_base in
// order; wr_done after the last.  Two ports, so the read of the next head
// can overlap the write of this one behind an arbiter.
// ---------------------------------------------------------------------------
module fabric_row_dma #(
    parameter int ROW_BITS = 2048,
    parameter int DW       = 128,
    parameter int ROWS     = 128
) (
    input  wire                clk,
    input  wire                rst_n,
    // read side
    input  wire                rd_start,
    input  wire [31:0]         rd_base,
    output reg                 rd_done,
    output reg                 row_out_valid,
    output reg  [ROW_BITS-1:0] row_out,
    output reg                 rd_req_valid,
    input  wire                rd_req_ready,
    output wire [31:0]         rd_req_addr,
    output wire [11:0]         rd_req_beats,
    input  wire                rd_rdata_valid,
    input  wire [DW-1:0]       rd_rdata,
    // write side
    input  wire                wr_start,
    input  wire [31:0]         wr_base,
    output reg                 wr_done,
    input  wire                row_in_valid,
    input  wire [ROW_BITS-1:0] row_in,
    output reg                 wr_req_valid,
    input  wire                wr_req_ready,
    output wire [31:0]         wr_req_addr,
    output wire [11:0]         wr_req_beats,
    output wire                wr_wdata_valid,
    input  wire                wr_wdata_ready,
    output wire [DW-1:0]       wr_wdata
);
    localparam int BPR = ROW_BITS / DW;                // beats per row
    localparam int BW  = $clog2(BPR) + 1;
    localparam int RW  = $clog2(ROWS) + 1;
    localparam int ROW_BYTES = ROW_BITS / 8;

    // Read.
    reg          rd_busy;
    reg [RW-1:0] rd_row;
    reg [BW-1:0] rd_beat;
    reg          rd_inflight;
    assign rd_req_addr  = rd_base + rd_row * ROW_BYTES;
    assign rd_req_beats = BPR;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_busy <= 1'b0; rd_row <= 0; rd_beat <= 0; rd_inflight <= 1'b0; rd_req_valid <= 1'b0;
            rd_done <= 1'b0; row_out_valid <= 1'b0;
        end else begin
            rd_done <= 1'b0;
            row_out_valid <= 1'b0;
            if (rd_start) begin
                rd_busy <= 1'b1; rd_row <= 0; rd_beat <= 0; rd_inflight <= 1'b0; rd_req_valid <= 1'b1;
            end else if (rd_busy) begin
                if (rd_req_valid && rd_req_ready) begin
                    rd_req_valid <= 1'b0; rd_inflight <= 1'b1; rd_beat <= 0;
                end
                if (rd_inflight && rd_rdata_valid) begin
                    row_out[rd_beat*DW +: DW] <= rd_rdata;
                    rd_beat <= rd_beat + 1'b1;
                    if (rd_beat == BPR - 1) begin
                        row_out_valid <= 1'b1;
                        rd_inflight <= 1'b0;
                        rd_row <= rd_row + 1'b1;
                        if (rd_row == ROWS - 1) begin rd_busy <= 1'b0; rd_done <= 1'b1; end
                        else rd_req_valid <= 1'b1;
                    end
                end
            end
        end
    end

    // Write: FIFO of rows, then one burst per row.
    reg [ROW_BITS-1:0] fifo [0:ROWS-1];
    reg [RW-1:0] f_wr, f_rd;
    reg          wr_busy;
    reg [RW-1:0] wr_row;
    reg [BW-1:0] wr_beat;
    reg          wr_streaming;
    wire         f_empty = (f_wr == f_rd);
    assign wr_req_addr    = wr_base + wr_row * ROW_BYTES;
    assign wr_req_beats   = BPR;
    assign wr_wdata_valid = wr_streaming;
    assign wr_wdata       = fifo[f_rd][wr_beat*DW +: DW];
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            f_wr <= 0; f_rd <= 0; wr_busy <= 1'b0; wr_row <= 0; wr_beat <= 0; wr_streaming <= 1'b0;
            wr_req_valid <= 1'b0; wr_done <= 1'b0;
        end else begin
            wr_done <= 1'b0;
            if (row_in_valid) begin
                fifo[f_wr] <= row_in;
                f_wr <= f_wr + 1'b1;
            end
            if (wr_start) begin
                wr_busy <= 1'b1; wr_row <= 0; wr_streaming <= 1'b0; wr_req_valid <= 1'b0;
            end else if (wr_busy) begin
                if (!wr_streaming && !wr_req_valid && !f_empty) wr_req_valid <= 1'b1;
                if (wr_req_valid && wr_req_ready) begin
                    wr_req_valid <= 1'b0; wr_streaming <= 1'b1; wr_beat <= 0;
                end
                if (wr_streaming && wr_wdata_ready) begin
                    wr_beat <= wr_beat + 1'b1;
                    if (wr_beat == BPR - 1) begin
                        wr_streaming <= 1'b0;
                        f_rd <= f_rd + 1'b1;
                        wr_row <= wr_row + 1'b1;
                        if (wr_row == ROWS - 1) begin wr_busy <= 1'b0; wr_done <= 1'b1; f_wr <= 0; f_rd <= 0; end
                    end
                end
            end
        end
    end
endmodule

// ---------------------------------------------------------------------------
// Top-K by streaming insertion: K entries sorted by score, a candidate
// enters above the first entry it strictly beats, so an earlier candidate
// keeps its place against an equal later one.  `clear` empties it;
// `finish` streams the entries out in rank order.
// ---------------------------------------------------------------------------
module fabric_topk #(
    parameter int K   = 32,
    parameter int IDW = 16,
    parameter int SW  = 32
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 clear,
    input  wire                 cand_valid,
    input  wire [IDW-1:0]       cand_id,
    input  wire signed [SW-1:0] cand_score,
    input  wire                 finish,
    output reg                  out_valid,
    output reg  [IDW-1:0]       out_id,
    output reg  signed [SW-1:0] out_score,
    output reg                  out_last,
    output reg                  done
);
    localparam int KW = $clog2(K) + 1;
    reg [IDW-1:0]       ids [0:K-1];
    reg signed [SW-1:0] scores [0:K-1];
    reg [K-1:0]         valid;
    reg [KW-1:0]        count;
    // Insertion position: the first slot that is empty or strictly beaten.
    integer j;
    reg [K-1:0] beat_r;
    always @* for (j = 0; j < K; j = j + 1) beat_r[j] = !valid[j] || (cand_score > scores[j]);
    reg          walking;
    reg [KW-1:0] walk;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid <= 0; count <= 0; walking <= 1'b0; walk <= 0; out_valid <= 1'b0; out_last <= 1'b0; done <= 1'b0;
        end else begin
            out_valid <= 1'b0; out_last <= 1'b0; done <= 1'b0;
            if (clear) begin
                valid <= 0; count <= 0; walking <= 1'b0;
            end else if (cand_valid) begin
                for (j = K - 1; j >= 0; j = j - 1) begin
                    if (beat_r[j]) begin
                        // Slot j moves down unless it is the first beaten slot, which takes the candidate.
                        if (j == 0 || !beat_r[j-1]) begin
                            ids[j] <= cand_id; scores[j] <= cand_score; valid[j] <= 1'b1;
                        end else if (j > 0) begin
                            ids[j] <= ids[j-1]; scores[j] <= scores[j-1]; valid[j] <= valid[j-1];
                        end
                    end
                end
                if (|beat_r && count != K) count <= count + 1'b1;
            end else if (finish) begin
                walking <= 1'b1; walk <= 0;
                if (count == 0) begin walking <= 1'b0; done <= 1'b1; end
            end else if (walking) begin
                out_valid <= 1'b1; out_id <= ids[walk]; out_score <= scores[walk];
                walk <= walk + 1'b1;
                if (walk == count - 1) begin out_last <= 1'b1; walking <= 1'b0; done <= 1'b1; end
            end
        end
    end
endmodule

// ---------------------------------------------------------------------------
// Index scan: for blocks 0 .. n-1 read the index record (code beats then the
// scale beat) and score it against the query's codes,
//   score = scale * sum (2q - 15)(2k - 15),
// emitting one candidate per block.  done after the last candidate.
// ---------------------------------------------------------------------------
module fabric_index_scan #(
    parameter int DW   = 128,
    parameter int IDIM = 128,
    parameter int IDW  = 16,
    parameter int RPB  = 25                                // records per request (a page)
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              start,
    input  wire [31:0]       base,
    input  wire [IDW-1:0]    n_blocks,
    input  wire [IDIM*4-1:0] q_codes,
    output reg               done,
    output reg               cand_valid,
    output reg  [IDW-1:0]    cand_id,
    output reg  signed [31:0] cand_score,
    output reg               req_valid,
    input  wire              req_ready,
    output wire [31:0]       req_addr,
    output wire [11:0]       req_beats,
    input  wire              rdata_valid,
    input  wire [DW-1:0]     rdata
);
    localparam int CPB  = DW / 4;                          // codes per beat
    localparam int CB   = (IDIM + CPB - 1) / CPB;          // code beats
    localparam int REC  = (CB + 1) * (DW / 8);             // record bytes
    localparam int BW   = $clog2(CB + 1) + 1;
    localparam int RW   = $clog2(RPB + 1);
    reg              busy, inflight;
    reg [IDW-1:0]    blk;                                  // the block being scored
    reg [IDW-1:0]    first;                                // the first block of the request
    reg [RW-1:0]     count, got;                           // records in the request, records finished
    reg [BW-1:0]     beat;
    reg signed [31:0] acc;
    wire [IDW-1:0]   left = n_blocks - blk;
    wire [RW-1:0]    want = (left > RPB) ? RPB[RW-1:0] : left[RW-1:0];
    assign req_addr  = base + blk * REC;
    assign req_beats = want * (CB + 1);
    // Partial dot product of this beat's codes.  Each factor is in [-15, 15]
    // and each product in [-225, 225], so the beat's sum needs 14 bits, not
    // 64; the sum is a balanced tree, since a chain of CPB adds is CPB carry
    // chains deep and ABC cannot restructure them.
    localparam int PWID = 16;
    localparam int LV   = $clog2(CPB);
    localparam int CPP  = 1 << LV;                         // the tree's width, CPB padded
    integer c, lv;
    reg signed [PWID-1:0] tree [0:LV][0:CPP-1];
    always @* begin
        // An assignment, not a conditional expression: an unsigned zero in the
        // other arm would make the whole expression unsigned.
        for (c = 0; c < CPP; c = c + 1) begin
            tree[0][c] = 0;
            if (c < CPB && beat * CPB + c < IDIM)
                tree[0][c] = (2 * $signed({1'b0, q_codes[(beat*CPB + c)*4 +: 4]}) - 16'sd15)
                             * (2 * $signed({1'b0, rdata[c*4 +: 4]}) - 16'sd15);
        end
        for (lv = 1; lv <= LV; lv = lv + 1)
            for (c = 0; c < (CPP >> lv); c = c + 1)
                tree[lv][c] = tree[lv-1][2*c] + tree[lv-1][2*c+1];
    end
    wire signed [PWID-1:0] part = tree[LV][0];
    wire signed [63:0] final_score = acc * $signed({56'b0, rdata[7:0]});
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0; inflight <= 1'b0; blk <= 0; first <= 0; count <= 0; got <= 0; beat <= 0; acc <= 0;
            req_valid <= 1'b0; done <= 1'b0; cand_valid <= 1'b0;
        end else begin
            done <= 1'b0;
            cand_valid <= 1'b0;
            if (start) begin
                blk <= 0; beat <= 0; acc <= 0; inflight <= 1'b0;
                if (n_blocks == 0) done <= 1'b1;
                else begin busy <= 1'b1; req_valid <= 1'b1; end
            end else if (busy) begin
                if (req_valid && req_ready) begin
                    req_valid <= 1'b0; inflight <= 1'b1; beat <= 0; acc <= 0; count <= want; got <= 0;
                end
                if (inflight && rdata_valid) begin
                    if (beat < CB) begin
                        acc <= acc + {{(32-PWID){part[PWID-1]}}, part};
                        beat <= beat + 1'b1;
                    end else begin
                        cand_valid <= 1'b1; cand_id <= blk; cand_score <= final_score[31:0];
                        beat <= 0; acc <= 0;
                        blk <= blk + 1'b1;
                        got <= got + 1'b1;
                        if (got == count - 1) begin
                            inflight <= 1'b0;
                            if (blk == n_blocks - 1) begin busy <= 1'b0; done <= 1'b1; end
                            else req_valid <= 1'b1;
                        end
                    end
                end
            end
        end
    end
endmodule

// ---------------------------------------------------------------------------
// Record reader: each request names `count` consecutive key-then-value
// records of one KV head (a page of the head-major window, or one block
// record); they are read in one burst into a buffer of MAXR records,
// unpacked from KV_BITS to int8, and streamed record by record to the
// attention core as HD/L key beats (kind 2) then HD/L value beats (kind 3),
// honouring its ready.  Requests arrive on a valid/ready queue; rec_done
// pulses per record.
// ---------------------------------------------------------------------------
module fabric_record_reader #(
    parameter int DW      = 128,
    parameter int HD      = 256,
    parameter int KV_BITS = 8,
    parameter int L       = 64,
    parameter int MAXR    = 8                              // records per request at most
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           addr_valid,
    output wire           addr_ready,
    input  wire [31:0]    addr,
    input  wire [7:0]     addr_count,
    output reg            req_valid,
    input  wire           req_ready,
    output reg  [31:0]    req_addr,
    output reg  [11:0]    req_beats,
    input  wire           rdata_valid,
    input  wire [DW-1:0]  rdata,
    output reg            out_valid,
    input  wire           out_ready,
    output reg  [1:0]     out_kind,
    output reg  [L*8-1:0] out_data,
    output reg            rec_done
);
    localparam int HALF_BEATS = (HD * KV_BITS + DW - 1) / DW;
    localparam int REC_BEATS  = 2 * HALF_BEATS;
    localparam int EPB        = DW / KV_BITS;             // elements per beat
    localparam int OUT_BEATS  = 2 * (HD / L);
    localparam int BW         = $clog2(REC_BEATS) + 1;
    localparam int OW         = $clog2(OUT_BEATS) + 1;
    localparam int RW         = $clog2(MAXR) + 1;
    // The records are a memory, not a register array.  Read as registers,
    // "record rrec, beat ob" is one mux over MAXR * 2 * HD bytes whose first
    // select bit drives five hundred loads -- two of the unit's two and a half
    // nanoseconds -- and in silicon this is a small SRAM anyway.  Its word is
    // the beat that arrives, so a write is a word; a beat out is L of the
    // word's bytes, and the address leads the data by a cycle.
    localparam int MW         = EPB * 8;                   // memory word: one unpacked beat
    localparam int OPW        = (EPB > L) ? EPB / L : 1;   // out beats a word holds
    localparam int MD         = MAXR * REC_BEATS;
    localparam int MAW        = (MD > 1) ? $clog2(MD) : 1;
    reg              busy, inflight, emitting;
    reg [BW-1:0]     beat;
    reg [OW-1:0]     ob;
    reg [RW-1:0]     count, wrec, rrec;                    // records in the request, filled, emitted
    assign addr_ready = !busy;
    // Unpack one beat to EPB int8 elements.
    integer e;
    reg [EPB*8-1:0] unpacked;
    always @* begin
        for (e = 0; e < EPB; e = e + 1)
            if (KV_BITS == 8) unpacked[e*8 +: 8] = rdata[e*8 +: 8];
            else              unpacked[e*8 +: 8] = {rdata[e*4 +: 4], 4'b0};
    end
    wire out_fire = out_valid && out_ready;
    wire have_rec = (wrec != rrec);                        // a filled record awaits emission
    // The beat the memory is asked for is the one that will be wanted: the
    // next if this cycle takes a word, the first if emission starts here.
    // Each half of a record starts at a word of its own -- HD need not be a
    // whole number of beats (24 elements of 8 bits is a beat and a half), and
    // the beats that arrive are what the halves are padded to.
    localparam int HB2 = OUT_BEATS / 2;                    // out beats in a half
    wire            take   = emitting && (!out_valid || out_fire);
    wire            begins = busy && !emitting && !out_valid && have_rec;
    wire [OW-1:0]   ob_a   = begins ? {OW{1'b0}}
                                    : ((take && ob != OUT_BEATS - 1) ? ob + 1'b1 : ob);
    wire            hi_a   = (ob_a >= HB2);
    wire [OW-1:0]   eo_a   = hi_a ? (ob_a - HB2[OW-1:0]) : ob_a;
    wire [MAW-1:0]  rd_w   = (rrec * REC_BEATS + (hi_a ? HALF_BEATS : 0) + (eo_a * L) / EPB);
    wire [OW-1:0]   eo     = (ob >= HB2) ? (ob - HB2[OW-1:0]) : ob;
    wire [MAW-1:0]  wr_w   = (wrec * REC_BEATS + beat);
    wire [MW-1:0]   rec_q;
    fabric_sram #(.W(MW), .D(MD), .NRD(1), .NWR(1), .MB(MW)) u_recs (
        .clk(clk), .rd_en(1'b1), .rd_addr(rd_w), .rd_data(rec_q),
        .wr_en(inflight && rdata_valid), .wr_addr(wr_w), .wr_data(unpacked), .wr_mask(1'b1));
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0; inflight <= 1'b0; emitting <= 1'b0; beat <= 0; ob <= 0; count <= 0; wrec <= 0; rrec <= 0;
            req_valid <= 1'b0; req_addr <= 0; req_beats <= 0; out_valid <= 1'b0; rec_done <= 1'b0;
        end else begin
            rec_done <= 1'b0;
            if (!busy && addr_valid) begin
                busy <= 1'b1; req_valid <= 1'b1; req_addr <= addr; req_beats <= addr_count * REC_BEATS;
                count <= addr_count[RW-1:0]; beat <= 0; wrec <= 0; rrec <= 0;
            end
            if (req_valid && req_ready) begin req_valid <= 1'b0; inflight <= 1'b1; end
            if (inflight && rdata_valid) begin
                // Beat b of record wrec is word wrec*REC_BEATS + b.
                if (beat == REC_BEATS - 1) begin
                    beat <= 0; wrec <= wrec + 1'b1;
                    if (wrec == count - 1) inflight <= 1'b0;
                end else beat <= beat + 1'b1;
            end
            if (emitting) begin
                if (!out_valid || out_fire) begin
                    out_valid <= 1'b1;
                    out_kind  <= (ob < OUT_BEATS / 2) ? 2'd2 : 2'd3;
                    out_data  <= rec_q[((eo % OPW) * L * 8) +: L*8];
                    ob <= ob + 1'b1;
                    if (ob == OUT_BEATS - 1) emitting <= 1'b0;
                end
            end else if (out_fire) begin
                out_valid <= 1'b0;
                rec_done <= 1'b1;
                rrec <= rrec + 1'b1;
                if (rrec == count - 1) busy <= 1'b0;
            end else if (busy && !out_valid && have_rec) begin
                emitting <= 1'b1; ob <= 0;
            end
        end
    end
endmodule

// ---------------------------------------------------------------------------
// The global layer's append: this token's keys and values into its window
// slot, the block sums, and at a block's end the block record and its index
// record.  `start` latches the token with the context's running sums
// (int16 per element, kept per context in memory by the caller); `done`
// when every write has been issued and accepted, with the new sums on the
// out ports (zero after a block's end).
//
//   window record   n * W + (pos mod W)       key then value, KV_BITS (head-major)
//   block record    block * NKV + n           the rounded means, KV_BITS
//   index record    block                     4-bit codes of the L2-normalised
//                                             mean of index_k, then its scale
// ---------------------------------------------------------------------------
module fabric_kv_append #(
    parameter int DW      = 128,
    parameter int HD      = 256,
    parameter int NKV     = 4,
    parameter int IDIM    = 128,
    parameter int BS      = 16,
    parameter int KV_BITS = 8,
    parameter int W       = 512,
    parameter     LUT_DIR = "./"
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                start,
    input  wire [31:0]         pos,
    input  wire [31:0]         window_base,
    input  wire [31:0]         block_base,
    input  wire [31:0]         index_base,
    input  wire [NKV*HD*8-1:0] k_rows,
    input  wire [NKV*HD*8-1:0] v_rows,
    input  wire [IDIM*8-1:0]   idx_k,
    // The block sums are a memory here, and they arrive and leave as the DMA
    // moves them: a beat in as it reads one, a beat out as it writes one back.
    // What this replaces was the whole vector on wires, 35 kbit at the 9B
    // geometry, with every element updated and averaged in one cycle.
    input  wire                s_in_valid,
    input  wire [15:0]         s_in_addr,
    input  wire [DW-1:0]       s_in_data,
    input  wire [15:0]         s_out_addr,
    output wire [DW-1:0]       s_out_data,
    output reg                 done,
    output reg                 req_valid,
    input  wire                req_ready,
    output reg  [31:0]         req_addr,
    output wire [11:0]         req_beats,
    output wire                wdata_valid,
    input  wire                wdata_ready,
    output wire [DW-1:0]       wdata
);
    localparam int HALF_BEATS = (HD * KV_BITS + DW - 1) / DW;
    localparam int REC_BEATS  = 2 * HALF_BEATS;
    localparam int REC_BYTES  = REC_BEATS * DW / 8;
    localparam int EPB        = DW / KV_BITS;
    localparam int CPB        = DW / 4;
    localparam int CB         = (IDIM + CPB - 1) / CPB;
    localparam int IREC_BEATS = CB + 1;
    localparam int IREC_BYTES = IREC_BEATS * DW / 8;
    localparam int SUMW       = 8 + $clog2(BS);
    localparam int LOG_BS     = $clog2(BS);
    localparam int NL         = 8;                        // norm lanes
    localparam int SWB        = 15 + $clog2(IDIM);        // sum of IDIM int8 squares
    localparam int SW         = SWB + (SWB % 2);

    reg [NKV*HD*8-1:0]  k_r, v_r;
    reg [IDIM*8-1:0]    idx_r;
    reg [31:0]          pos_r;
    localparam int SPB    = DW / 16;                    // sums to a beat
    localparam int NKSUM  = NKV * HD;                   // the keys' sums, then the values', then the index's
    localparam int NSUM   = 2 * NKSUM + IDIM;
    localparam int SBEATS = (NSUM + SPB - 1) / SPB;
    localparam int BLKW   = NKV * REC_BEATS;            // the block record, a beat to a word
    localparam int IBARW  = (IDIM + NL - 1) / NL;
    localparam int SAW    = (SBEATS > 1) ? $clog2(SBEATS) : 1;
    localparam int BAW    = (BLKW > 1) ? $clog2(BLKW) : 1;
    localparam int IAW    = (IBARW > 1) ? $clog2(IBARW) : 1;

    // The record being written: packed key and value halves of one head.
    // A record is emitted a beat at a time, so only the beat is held: what
    // this replaces was the whole record in flops, packed in one cycle by
    // HALF_BEATS packers, and read out by a beat counter that had to decode
    // into every beat's write enables.
    reg [DW-1:0] wb_q;
    reg [11:0] beats_r;
    assign req_beats = beats_r;

    // Packing of an int8 vector into a half: element e at bit KV_BITS*e.
    // An int8 rounded down to four bits: nine bits carries the round and the
    // saturation is a compare, where the 64-bit helpers were a lane each of
    // 64-bit adder and shifter.
    function automatic [DW-1:0] pack_beat(input [NKV*HD*8-1:0] rows, input integer head, input integer b);
        integer e;
        reg signed [8:0] q;
        begin
            pack_beat = 0;
            for (e = 0; e < EPB; e = e + 1)
                if (b * EPB + e < HD) begin
                    if (KV_BITS == 8) pack_beat[e*8 +: 8] = rows[(head*HD + b*EPB + e)*8 +: 8];
                    else begin
                        q = ($signed(rows[(head*HD + b*EPB + e)*8 +: 8]) + 9'sd8) >>> 4;
                        if (q > 9'sd7) q = 9'sd7;
                        if (q < -9'sd8) q = -9'sd8;
                        pack_beat[e*4 +: 4] = q[3:0];
                    end
                end
        end
    endfunction

    // The unit vector's largest magnitude, as a balanced tree: the chain of
    // compare-selects this replaces was IDIM deep, all of it in one cycle.
    localparam int MLV = $clog2(IDIM), MPP = 1 << MLV;
    reg [8:0] mtree [0:MLV][0:MPP-1];
    integer ml, mc;
    always @* begin
        for (mc = 0; mc < MPP; mc = mc + 1) begin
            mtree[0][mc] = 9'd1;
            if (mc < IDIM)
                mtree[0][mc] = unit[mc*8+7] ? (9'd256 - {1'b0, unit[mc*8 +: 8]}) : {1'b0, unit[mc*8 +: 8]};
        end
        for (ml = 1; ml <= MLV; ml = ml + 1)
            for (mc = 0; mc < (MPP >> ml); mc = mc + 1)
                mtree[ml][mc] = (mtree[ml-1][2*mc] > mtree[ml-1][2*mc+1]) ? mtree[ml-1][2*mc] : mtree[ml-1][2*mc+1];
    end
    wire [8:0] unit_absmax = (mtree[MLV][0] > 9'd1) ? mtree[MLV][0] : 9'd1;

    // The block sums, and the means taken from them.
    //
    // A beat of the sums arrives as the DMA reads it: eight sums, and the
    // eight rows that belong with them are added there and then, so the
    // update that was NKV*HD adders in one cycle is eight.  A beat lies
    // wholly inside one of the three regions and, in the first two, inside
    // one head and one record beat, because HD and NKV*HD are multiples of
    // eight -- so the means it yields go straight into the block record beat
    // they belong to, and kbar and vbar never exist.
    wire [SAW-1:0]  s_rd_addr = s_out_addr[SAW-1:0];
    wire [DW-1:0]   s_rd_data;
    reg  [DW-1:0]   s_wr_data;
    assign s_out_data = block_end ? {DW{1'b0}} : s_rd_data;   // a closed block starts again at zero
    fabric_sram #(.W(DW), .D(SBEATS), .MB(DW)) u_sums (
        .clk(clk), .rd_en(1'b1), .rd_addr(s_rd_addr), .rd_data(s_rd_data),
        .wr_en(s_in_valid), .wr_addr(s_in_addr[SAW-1:0]), .wr_data(s_wr_data), .wr_mask(1'b1));

    reg  [BAW-1:0]  blk_raddr;
    wire [DW-1:0]   blk_rdata;
    reg  [DW-1:0]   blk_next;
    reg  [EPB-1:0]  blk_mask;                           // the codes this beat owns
    reg  [BAW-1:0]  nidx;
    wire            blk_we;
    fabric_sram #(.W(DW), .D(BLKW), .NRD(1), .NWR(1), .MB(KV_BITS)) u_blk (
        .clk(clk), .rd_en(1'b1), .rd_addr(blk_raddr), .rd_data(blk_rdata),
        .wr_en(blk_we), .wr_addr(nidx), .wr_data(blk_next), .wr_mask(blk_mask));

    reg  [IAW-1:0]  ibar_raddr;
    wire [NL*8-1:0] ibar_rdata;
    reg  [NL*8-1:0] ibar_next;
    reg  [IAW-1:0]  iidx;
    wire            ibar_we;
    fabric_sram #(.W(NL*8), .D(IBARW), .MB(NL*8)) u_ibar (
        .clk(clk), .rd_en(1'b1), .rd_addr(ibar_raddr), .rd_data(ibar_rdata),
        .wr_en(ibar_we), .wr_addr(iidx), .wr_data(ibar_next), .wr_mask(1'b1));

    // One arriving beat: the eight sums updated, and if the block closes here,
    // their means in the shape the record wants them.
    // The sums are SUMW bits and their means a byte, so a round needs one bit
    // more than the sum and a code nine, not sixty-four apiece across SPB lanes.
    localparam [SUMW:0] HALF_BS = BS >> 1;      // the round the mean adds before its shift
    integer se;
    reg signed [SUMW:0]   sacc, srnd;
    reg signed [8:0]      smean;
    reg signed [7:0]      srow;                 // one row's element, and signed: a
                                                // part-select of an integer is not
    integer sflat, sflat0, shead, swithin, spos;
    always @* begin
        // A beat lies wholly in one region, and in the first two wholly in one
        // head and one record beat, so its word is known before the elements.
        sflat0 = s_in_addr * SPB;
        if (sflat0 < NKSUM)
            nidx = (sflat0 / HD) * REC_BEATS + (sflat0 % HD) / EPB;
        else if (sflat0 < 2*NKSUM)
            nidx = ((sflat0 - NKSUM) / HD) * REC_BEATS + HALF_BEATS + ((sflat0 - NKSUM) % HD) / EPB;
        else
            nidx = 0;
        blk_next = 0;
        blk_mask = 0;
        ibar_next = 0;
        iidx = 0;
        for (se = 0; se < SPB; se = se + 1) begin
            sflat = sflat0 + se;
            srow = 0;
            if (sflat < NKSUM)        srow = $signed(k_rows[sflat*8 +: 8]);
            else if (sflat < 2*NKSUM) srow = $signed(v_rows[(sflat - NKSUM)*8 +: 8]);
            else if (sflat < NSUM)    srow = $signed(idx_k[(sflat - 2*NKSUM)*8 +: 8]);
            sacc = $signed(s_in_data[se*16 +: SUMW]) + srow;
            s_wr_data[se*16 +: 16] = {{(16-SUMW){sacc[SUMW-1]}}, sacc[SUMW-1:0]};
            srnd  = $signed(sacc[SUMW-1:0]) + $signed(HALF_BS);
            smean = srnd >>> LOG_BS;
            if (sflat < 2*NKSUM) begin
                swithin = (sflat < NKSUM) ? (sflat % HD) : ((sflat - NKSUM) % HD);
                spos    = swithin % EPB;
                blk_mask[spos] = 1'b1;                 // a beat owns SPB of the word's codes
                if (KV_BITS == 8) blk_next[spos*8 +: 8] = smean[7:0];
                else begin
                    smean = ($signed(smean[7:0]) + 9'sd8) >>> 4;
                    if (smean > 9'sd7) smean = 9'sd7;
                    if (smean < -9'sd8) smean = -9'sd8;
                    blk_next[spos*4 +: 4] = smean[3:0];
                end
            end else if (sflat < NSUM) begin
                ibar_next[(sflat - 2*NKSUM) % NL * 8 +: 8] = smean[7:0];
                iidx = (sflat - 2*NKSUM) / NL;
            end
        end
    end
    assign blk_we  = s_in_valid && block_end && (s_in_addr * SPB < 2*NKSUM);
    assign ibar_we = s_in_valid && block_end && (s_in_addr * SPB >= 2*NKSUM) && (s_in_addr * SPB < NSUM);


    integer j;
    reg signed [63:0] t;

    // Index unit vector: the norm unit over the mean, then codes.
    reg             nv_in;
    wire [NL*8-1:0] n_x = ibar_rdata;
    wire            nv_out;
    wire [NL*8-1:0] n_y;
    fabric_rmsnorm #(.D(IDIM), .XW(8), .OW(8), .L(NL), .SW(SW), .LUT_DIR(LUT_DIR)) u_norm (
        .clk(clk), .rst_n(rst_n), .in_valid(nv_in), .n_beats((IDIM / NL)), .in_x(n_x), .in_gain({NL{16'd1}}), .mult(16'd1), .shift(6'd7),
        .eps({SW{1'b0}}), .out_valid(nv_out), .out_y(n_y));
    reg [IDIM*8-1:0] unit;
    reg [7:0]  scale;
    reg        rc_start;
    wire       rc_done;
    wire [16:0] rc_r;
    wire [5:0]  rc_lz;
    fabric_recip #(.LW(9), .LUT_DIR(LUT_DIR)) u_rc (.clk(clk), .start(rc_start), .l({scale, 1'b0}), .done(rc_done), .r(rc_r), .lz_out(rc_lz));

    // The index record's codes are packed LI a cycle, not a beat a cycle.  A
    // whole beat was CPB multiplies and CPB variable shifts in one cycle off
    // one reciprocal, one scale and one shift: nineteen of the append's
    // twenty-one nanoseconds were three nets of five hundred-odd loads driven
    // by minimum-size cells, which is what a value read by every lane comes
    // to.  Lanes are what every other vector unit here has, and they cost
    // LSTEPS cycles a beat -- tens over a token of two hundred thousand.
    localparam int LI     = (CPB > 4) ? 4 : CPB;          // codes packed a cycle
    localparam int LSTEPS = (CPB + LI - 1) / LI;          // cycles a beat
    reg        ipack;                                     // a beat is being packed
    reg [7:0]  lane;

    localparam [3:0] S_IDLE = 0, S_WIN = 1, S_WIN_DATA = 2, S_BLK = 3, S_BLK_DATA = 4, S_NORM_IN = 5, S_NORM_OUT = 6,
                     S_SCALE = 7, S_CODES = 8, S_IREC = 9, S_IREC_DATA = 10, S_DONE = 11;
    reg [3:0]  state;
    reg [3:0]  head;
    reg [7:0]  beat;
    reg [7:0]  nbeat;
    wire       block_end = (pos[LOG_BS-1:0] == BS - 1);
    assign wdata_valid = (state == S_WIN_DATA) || (state == S_BLK_DATA) || (state == S_IREC_DATA && !ipack);
    assign wdata       = (state == S_BLK_DATA) ? blk_rdata : wb_q;

    function automatic [DW-1:0] rec_beat(input [NKV*HD*8-1:0] kr, input [NKV*HD*8-1:0] vr,
                                         input integer hd, input integer bt);
        rec_beat = (bt < HALF_BEATS) ? pack_beat(kr, hd, bt) : pack_beat(vr, hd, bt - HALF_BEATS);
    endfunction
    // The widths the values actually need.  An int8 plus an unsigned byte is
    // ten bits, fifteen times that is fourteen, and that by the 16-bit
    // reciprocal is thirty.  Written through the 64-bit helpers it was CPB
    // lanes of a 64 by 64 multiply, twice over, in one cycle -- the same
    // mistake the index scan's chain of 64-bit adds was.
    function automatic [LI*4-1:0] idx_lanes(input integer bt, input integer ln);
        integer e, idx, sh;
        reg signed [9:0]  a;
        reg signed [13:0] b;
        reg signed [31:0] q;
        begin
            idx_lanes = 0;
            sh = 24 - rc_lz;                      // the round is written out: synthesis will not take
            for (e = 0; e < LI; e = e + 1) begin  // fx_rnd_shr's variable shift through a nested call
                idx = bt * CPB + ln * LI + e;
                if (idx < IDIM) begin
                    a = $signed(unit[idx*8 +: 8]) + $signed({2'b0, scale});
                    b = 14'sd15 * a;
                    q = $signed(b) * $signed({1'b0, rc_r});
                    if (sh > 0) q = (q + (32'sd1 <<< (sh - 1))) >>> sh;
                    if (q < 0) q = 0;
                    if (q > 15) q = 15;
                    idx_lanes[e*4 +: 4] = q[3:0];
                end
            end
        end
    endfunction

    // The block record is read from the memory the means went into, and the
    // index means from theirs, each addressed a cycle before it is wanted.
    always @* begin
        blk_raddr  = head * REC_BEATS + beat;
        ibar_raddr = nbeat[IAW-1:0];
        if (state == S_BLK_DATA && !req_valid && wdata_ready) blk_raddr = head * REC_BEATS + beat + 1;
        if (state == S_BLK) blk_raddr = head * REC_BEATS;
    end

    // The beat to write, formed the cycle before it is wanted.  Which beat
    // that is is a wire, so each packer is built once: written as a call at
    // the first beat and another at the rest, the index packer was two arrays
    // of CPB multipliers, and everything they share -- the reciprocal, the
    // shift, the scale -- was driving twice the gates it needed to.
    wire [31:0] wb_beat = (state == S_WIN) ? 32'd0 : beat + 1;
    always @(posedge clk) begin
        if (state == S_WIN || (state == S_WIN_DATA && !req_valid && wdata_ready))
            wb_q <= rec_beat(k_r, v_r, head, wb_beat);
        if (ipack) begin
            // The last beat of an index record is the scale, and is one lane's
            // work; the rest are LI codes at a time into the beat being held.
            if (beat == CB) wb_q <= {{(DW-8){1'b0}}, scale};
            else wb_q[lane*LI*4 +: LI*4] <= idx_lanes(beat, lane);
        end
    end

    integer b, e;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; head <= 0; beat <= 0; nbeat <= 0; req_valid <= 1'b0; done <= 1'b0;
            nv_in <= 1'b0; rc_start <= 1'b0; ipack <= 1'b0; lane <= 0;
        end else begin
            done <= 1'b0;
            nv_in <= 1'b0;
            rc_start <= 1'b0;
            case (state)
                S_IDLE: if (start) begin
                    k_r <= k_rows; v_r <= v_rows; idx_r <= idx_k; pos_r <= pos;
                    head <= 0;
                    state <= S_WIN;
                end
                S_WIN: begin
                    // Window record of this head, a beat at a time.
                    req_addr <= window_base + (head * W + (pos_r % W)) * REC_BYTES;
                    beats_r <= REC_BEATS;
                    req_valid <= 1'b1;
                    beat <= 0;
                    state <= S_WIN_DATA;
                end
                S_WIN_DATA: begin
                    if (req_valid && req_ready) req_valid <= 1'b0;
                    if (!req_valid && wdata_ready) begin
                        beat <= beat + 1'b1;
                        if (beat == REC_BEATS - 1) begin
                            if (head == NKV - 1) begin
                                head <= 0;
                                // The means were taken as the sums arrived.
                                if (block_end) state <= S_BLK;
                                else state <= S_DONE;
                            end else begin
                                head <= head + 1'b1;
                                state <= S_WIN;
                            end
                        end
                    end
                end
                S_BLK: begin
                    req_addr <= block_base + ((pos_r / BS) * NKV + head) * REC_BYTES;
                    beats_r <= REC_BEATS;
                    req_valid <= 1'b1;
                    beat <= 0;
                    state <= S_BLK_DATA;
                end
                S_BLK_DATA: begin
                    if (req_valid && req_ready) req_valid <= 1'b0;
                    if (!req_valid && wdata_ready) begin
                        beat <= beat + 1'b1;
                        if (beat == REC_BEATS - 1) begin
                            if (head == NKV - 1) begin head <= 0; nbeat <= 0; state <= S_NORM_IN; end
                            else begin head <= head + 1'b1; state <= S_BLK; end
                        end
                    end
                end
                S_NORM_IN: begin
                    nv_in <= 1'b1;                        // n_x is the memory's answer to nbeat
                    nbeat <= nbeat + 1'b1;
                    if (nbeat == IDIM / NL - 1) begin nbeat <= 0; state <= S_NORM_OUT; end
                end
                S_NORM_OUT: begin
                    if (nv_out) begin
                        unit[nbeat*NL*8 +: NL*8] <= n_y;
                        nbeat <= nbeat + 1'b1;
                        if (nbeat == IDIM / NL - 1) state <= S_SCALE;
                    end
                end
                S_SCALE: begin
                    // scale = max |u|, at least 1; then its reciprocal.
                    scale <= unit_absmax[7:0];
                    rc_start <= 1'b1;
                    state <= S_CODES;
                end
                S_CODES: if (rc_done) begin
                    // code = clip(round(15 (u + scale) * r >> (24 - lz)), 0, 15),
                    // LI of them at a time (see idx_lanes).  The request goes
                    // out while the first beat is still being packed: the data
                    // side waits on ipack, not on the request.
                    req_addr <= index_base + (pos_r / BS) * IREC_BYTES;
                    beats_r <= IREC_BEATS;
                    req_valid <= 1'b1;
                    beat <= 0;
                    ipack <= 1'b1; lane <= 0;
                    state <= S_IREC_DATA;
                end
                S_IREC_DATA: begin
                    if (req_valid && req_ready) req_valid <= 1'b0;
                    if (ipack) begin
                        lane <= lane + 1'b1;
                        if (lane == LSTEPS - 1 || beat == CB) begin ipack <= 1'b0; lane <= 0; end
                    end else if (!req_valid && wdata_ready) begin
                        beat <= beat + 1'b1;
                        if (beat == IREC_BEATS - 1) state <= S_DONE;
                        else begin ipack <= 1'b1; lane <= 0; end
                    end
                end
                S_DONE: begin done <= 1'b1; state <= S_IDLE; end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
