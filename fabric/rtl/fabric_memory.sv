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
    // Partial dot product of this beat's codes.
    integer c;
    reg signed [63:0] part;
    always @* begin
        part = 0;
        for (c = 0; c < CPB; c = c + 1)
            if (beat * CPB + c < IDIM)
                part = part + (2 * $signed({1'b0, q_codes[(beat*CPB + c)*4 +: 4]}) - 64'sd15)
                             * (2 * $signed({1'b0, rdata[c*4 +: 4]}) - 64'sd15);
    end
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
                        acc <= acc + part[31:0];
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
    reg [2*HD*8-1:0] recs [0:MAXR-1];                      // unpacked key then value, per record of the request
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
                // Beat b of half h of record wrec lands at element h*HD + b*EPB.
                if (beat < HALF_BEATS) recs[wrec][(beat*EPB)*8 +: EPB*8] <= unpacked;
                else                   recs[wrec][(HD + (beat - HALF_BEATS)*EPB)*8 +: EPB*8] <= unpacked;
                if (beat == REC_BEATS - 1) begin
                    beat <= 0; wrec <= wrec + 1'b1;
                    if (wrec == count - 1) inflight <= 1'b0;
                end else beat <= beat + 1'b1;
            end
            if (emitting) begin
                if (!out_valid || out_fire) begin
                    out_valid <= 1'b1;
                    out_kind  <= (ob < OUT_BEATS / 2) ? 2'd2 : 2'd3;
                    out_data  <= recs[rrec][ob*L*8 +: L*8];
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
// record.  `start` latches the token; `done` when every write has been
// issued and accepted.
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
    reg signed [SUMW-1:0] sum_k [0:NKV*HD-1];
    reg signed [SUMW-1:0] sum_v [0:NKV*HD-1];
    reg signed [SUMW-1:0] sum_i [0:IDIM-1];
    reg [LOG_BS:0]      count;

    // The record being written: packed key and value halves of one head.
    reg [REC_BEATS*DW-1:0] rec;
    reg [IREC_BEATS*DW-1:0] irec;
    reg [11:0] beats_r;
    assign req_beats = beats_r;

    // Packing of an int8 vector into a half: element e at bit KV_BITS*e.
    function automatic [DW-1:0] pack_beat(input [NKV*HD*8-1:0] rows, input integer head, input integer b);
        integer e;
        reg signed [63:0] q;
        begin
            pack_beat = 0;
            for (e = 0; e < EPB; e = e + 1)
                if (b * EPB + e < HD) begin
                    if (KV_BITS == 8) pack_beat[e*8 +: 8] = rows[(head*HD + b*EPB + e)*8 +: 8];
                    else begin
                        q = fx_sat(fx_rnd_shr($signed(rows[(head*HD + b*EPB + e)*8 +: 8]), 4), 4);
                        pack_beat[e*4 +: 4] = q[3:0];
                    end
                end
        end
    endfunction

    // Block means and index mean as int8 rows.
    reg [NKV*HD*8-1:0] kbar, vbar;
    reg [IDIM*8-1:0]   ibar;
    integer j;
    reg signed [63:0] t;

    // Index unit vector: the norm unit over the mean, then codes.
    reg             nv_in;
    reg [NL*8-1:0]  n_x;
    wire            nv_out;
    wire [NL*8-1:0] n_y;
    fabric_rmsnorm #(.D(IDIM), .XW(8), .OW(8), .L(NL), .SW(SW), .LUT_DIR(LUT_DIR)) u_norm (
        .clk(clk), .rst_n(rst_n), .in_valid(nv_in), .in_x(n_x), .in_gain({NL{16'd1}}), .mult(16'd1), .shift(6'd7),
        .eps({SW{1'b0}}), .out_valid(nv_out), .out_y(n_y));
    reg [IDIM*8-1:0] unit;
    reg [7:0]  scale;
    reg        rc_start;
    wire       rc_done;
    wire [16:0] rc_r;
    wire [5:0]  rc_lz;
    fabric_recip #(.LW(9), .LUT_DIR(LUT_DIR)) u_rc (.clk(clk), .start(rc_start), .l({scale, 1'b0}), .done(rc_done), .r(rc_r), .lz_out(rc_lz));

    localparam [3:0] S_IDLE = 0, S_WIN = 1, S_WIN_DATA = 2, S_BLK = 3, S_BLK_DATA = 4, S_NORM_IN = 5, S_NORM_OUT = 6,
                     S_SCALE = 7, S_CODES = 8, S_IREC = 9, S_IREC_DATA = 10, S_DONE = 11;
    reg [3:0]  state;
    reg [3:0]  head;
    reg [7:0]  beat;
    reg [7:0]  nbeat;
    reg        block_end;
    assign wdata_valid = (state == S_WIN_DATA) || (state == S_BLK_DATA) || (state == S_IREC_DATA);
    assign wdata       = (state == S_IREC_DATA) ? irec[beat*DW +: DW] : rec[beat*DW +: DW];

    integer b, e;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; head <= 0; beat <= 0; nbeat <= 0; count <= 0; req_valid <= 1'b0; done <= 1'b0;
            nv_in <= 1'b0; rc_start <= 1'b0; block_end <= 1'b0;
            for (j = 0; j < NKV*HD; j = j + 1) begin sum_k[j] <= 0; sum_v[j] <= 0; end
            for (j = 0; j < IDIM; j = j + 1) sum_i[j] <= 0;
        end else begin
            done <= 1'b0;
            nv_in <= 1'b0;
            rc_start <= 1'b0;
            case (state)
                S_IDLE: if (start) begin
                    k_r <= k_rows; v_r <= v_rows; idx_r <= idx_k; pos_r <= pos;
                    for (j = 0; j < NKV*HD; j = j + 1) begin
                        sum_k[j] <= sum_k[j] + $signed(k_rows[j*8 +: 8]);
                        sum_v[j] <= sum_v[j] + $signed(v_rows[j*8 +: 8]);
                    end
                    for (j = 0; j < IDIM; j = j + 1) sum_i[j] <= sum_i[j] + $signed(idx_k[j*8 +: 8]);
                    block_end <= (count == BS - 1);
                    count <= (count == BS - 1) ? 0 : count + 1'b1;
                    head <= 0;
                    state <= S_WIN;
                end
                S_WIN: begin
                    // Window record of this head.
                    for (b = 0; b < HALF_BEATS; b = b + 1) begin
                        rec[b*DW +: DW] <= pack_beat(k_r, head, b);
                        rec[(HALF_BEATS + b)*DW +: DW] <= pack_beat(v_r, head, b);
                    end
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
                                if (block_end) begin
                                    // Means of the block.
                                    for (j = 0; j < NKV*HD; j = j + 1) begin
                                        t = fx_rnd_shr(sum_k[j], LOG_BS); kbar[j*8 +: 8] <= t[7:0];
                                        t = fx_rnd_shr(sum_v[j], LOG_BS); vbar[j*8 +: 8] <= t[7:0];
                                        sum_k[j] <= 0; sum_v[j] <= 0;
                                    end
                                    for (j = 0; j < IDIM; j = j + 1) begin
                                        t = fx_rnd_shr(sum_i[j], LOG_BS); ibar[j*8 +: 8] <= t[7:0];
                                        sum_i[j] <= 0;
                                    end
                                    state <= S_BLK;
                                end else state <= S_DONE;
                            end else begin
                                head <= head + 1'b1;
                                state <= S_WIN;
                            end
                        end
                    end
                end
                S_BLK: begin
                    for (b = 0; b < HALF_BEATS; b = b + 1) begin
                        rec[b*DW +: DW] <= pack_beat(kbar, head, b);
                        rec[(HALF_BEATS + b)*DW +: DW] <= pack_beat(vbar, head, b);
                    end
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
                    nv_in <= 1'b1;
                    n_x <= ibar[nbeat*NL*8 +: NL*8];
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
                    t = 1;
                    for (j = 0; j < IDIM; j = j + 1) begin
                        if ($signed(unit[j*8 +: 8]) > t) t = $signed(unit[j*8 +: 8]);
                        if (-$signed(unit[j*8 +: 8]) > t) t = -$signed(unit[j*8 +: 8]);
                    end
                    scale <= t[7:0];
                    rc_start <= 1'b1;
                    state <= S_CODES;
                end
                S_CODES: if (rc_done) begin
                    // code = clip(round(15 (u + scale) * r >> (24 - lz)), 0, 15)
                    irec <= 0;
                    for (j = 0; j < IDIM; j = j + 1) begin
                        t = fx_rnd_shr(64'sd15 * ($signed(unit[j*8 +: 8]) + $signed({56'b0, scale})) * $signed({47'b0, rc_r}), 24 - rc_lz);
                        if (t < 0) t = 0;
                        if (t > 15) t = 15;
                        irec[j*4 +: 4] <= t[3:0];
                    end
                    irec[CB*DW +: 8] <= scale;
                    req_addr <= index_base + (pos_r / BS) * IREC_BYTES;
                    beats_r <= IREC_BEATS;
                    req_valid <= 1'b1;
                    beat <= 0;
                    state <= S_IREC_DATA;
                end
                S_IREC_DATA: begin
                    if (req_valid && req_ready) req_valid <= 1'b0;
                    if (!req_valid && wdata_ready) begin
                        beat <= beat + 1'b1;
                        if (beat == IREC_BEATS - 1) state <= S_DONE;
                    end
                end
                S_DONE: begin done <= 1'b1; state <= S_IDLE; end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
