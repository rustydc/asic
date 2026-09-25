// The clock crossing between the core (800 MHz: the arbiter and the units
// on the memory port) and the HPI controller (250 MHz, the device clock).
//
// fabric_async_fifo is a dual-clock FIFO with gray-coded pointers and
// two-flop synchronisers (the Cummings design): full is decided on the
// write clock from the synchronised read pointer, empty on the read clock
// from the synchronised write pointer, so neither flag is ever optimistic.
// The read data is first-word-fall-through.
//
// fabric_mem_bridge carries the memory port across: requests (write, address,
// beats) and write beats from the core to the controller, read beats back,
// several beats a transfer on each side (see the module).  The core has no
// back-pressure on read data, so the controller side is held off from the
// read FIFO's almost-full instead; an overflow, which would mean that failed,
// is latched into rd_overflow.

`timescale 1ns/1ps
`default_nettype none

module fabric_async_fifo #(
    parameter int W  = 128,
    parameter int AW = 4,                      // 2^AW entries
    parameter int AF = 0                       // wafull: this few entries or fewer are free
) (
    input  wire         wclk,
    input  wire         wrst_n,
    input  wire         wr_en,
    input  wire [W-1:0] wdata,
    output reg          wfull,
    output wire         wafull,
    input  wire         rclk,
    input  wire         rrst_n,
    input  wire         rd_en,
    output wire [W-1:0] rdata,
    output reg          rempty
);
    reg [W-1:0] mem [0:(1<<AW)-1];
    reg [AW:0]  wptr_bin, wptr_gray, rptr_bin, rptr_gray;
    reg [AW:0]  wq1_rptr, wq2_rptr;            // read pointer in the write domain
    reg [AW:0]  rq1_wptr, rq2_wptr;            // write pointer in the read domain

    // Write side.
    wire            w_go        = wr_en && !wfull;
    wire [AW:0]     wptr_bin_n  = wptr_bin + {{AW{1'b0}}, w_go};
    wire [AW:0]     wptr_gray_n = wptr_bin_n ^ (wptr_bin_n >> 1);
    wire            wfull_n     = (wptr_gray_n == {~wq2_rptr[AW:AW-1], wq2_rptr[AW-2:0]});
    // Almost full from the same synchronised pointer, so it is late by the
    // synchroniser as full is: never optimistic.  A writer that cannot stop at
    // once looks at this instead of wfull.
    reg  [AW:0]     wq2_rbin;
    integer         gi;
    always @* begin
        wq2_rbin[AW] = wq2_rptr[AW];
        for (gi = AW - 1; gi >= 0; gi = gi - 1) wq2_rbin[gi] = wq2_rbin[gi+1] ^ wq2_rptr[gi];
    end
    assign wafull = ((wptr_bin - wq2_rbin) >= ((1 << AW) - AF));
    always @(posedge wclk or negedge wrst_n) begin
        if (!wrst_n) begin
            wptr_bin <= 0; wptr_gray <= 0; wfull <= 1'b0; wq1_rptr <= 0; wq2_rptr <= 0;
        end else begin
            if (w_go) mem[wptr_bin[AW-1:0]] <= wdata;
            wptr_bin  <= wptr_bin_n;
            wptr_gray <= wptr_gray_n;
            wfull     <= wfull_n;
            wq1_rptr  <= rptr_gray;
            wq2_rptr  <= wq1_rptr;
        end
    end

    // Read side.
    wire            r_go        = rd_en && !rempty;
    wire [AW:0]     rptr_bin_n  = rptr_bin + {{AW{1'b0}}, r_go};
    wire [AW:0]     rptr_gray_n = rptr_bin_n ^ (rptr_bin_n >> 1);
    wire            rempty_n    = (rptr_gray_n == rq2_wptr);
    assign rdata = mem[rptr_bin[AW-1:0]];
    always @(posedge rclk or negedge rrst_n) begin
        if (!rrst_n) begin
            rptr_bin <= 0; rptr_gray <= 0; rempty <= 1'b1; rq1_wptr <= 0; rq2_wptr <= 0;
        end else begin
            rptr_bin  <= rptr_bin_n;
            rptr_gray <= rptr_gray_n;
            rempty    <= rempty_n;
            rq1_wptr  <= wptr_gray;
            rq2_wptr  <= rq1_wptr;
        end
    end
endmodule

// ---------------------------------------------------------------------------
// The memory port across the two clocks.
//
// The controller side moves MXB beats a transfer, the core side CXB for a
// wide request and one for a narrow one; both count a request in beats, and
// a request's last transfer carries what is left.  The core side packs its
// write beats into controller transfers and unpacks read transfers into its
// own, knowing each request's size and width from a queue it keeps of them.
// Read data is pushed back through the controller side's rdata_ready, from
// the read FIFO's almost-full, so the core may take reads as slowly as a beat
// a cycle: at MXB = 4 the controller can hand over more than that.
// ---------------------------------------------------------------------------
module fabric_mem_bridge #(
    parameter int DW     = 128,
    parameter int CXB    = 1,                  // beats a wide core transfer carries
    parameter int MXB    = 1,                  // beats a controller transfer carries
    parameter int REQ_AW = 2,                  // 4 requests
    parameter int WD_AW  = 4,                  // 16 write transfers
    parameter int RD_AW  = 4                   // 16 read transfers
) (
    // core side
    input  wire              c_clk,
    input  wire              c_rst_n,
    input  wire              c_req_valid,
    output wire              c_req_ready,
    input  wire              c_req_write,
    input  wire              c_req_wide,
    input  wire [31:0]       c_req_addr,
    input  wire [11:0]       c_req_beats,
    input  wire              c_wdata_valid,
    output wire              c_wdata_ready,
    input  wire [CXB*DW-1:0] c_wdata,
    output reg               c_rdata_valid,
    output reg  [CXB*DW-1:0] c_rdata,
    // controller side
    input  wire              m_clk,
    input  wire              m_rst_n,
    output wire              m_req_valid,
    input  wire              m_req_ready,
    output wire              m_req_write,
    output wire [31:0]       m_req_addr,
    output wire [11:0]       m_req_beats,
    output wire              m_wdata_valid,
    input  wire              m_wdata_ready,
    output wire [MXB*DW-1:0] m_wdata,
    input  wire              m_rdata_valid,
    output wire              m_rdata_ready,
    input  wire [MXB*DW-1:0] m_rdata,
    output reg               rd_overflow
);
    localparam int IW = 13;                     // a request's beats and its width, for the core side's queue
    wire c_req_go = c_req_valid && c_req_ready;
    wire c_wide   = (CXB > 1) && (c_req_wide === 1'b1);
    // The core side's queue of read requests, for unpacking their data.
    reg  [IW-1:0] rq [0:3];
    reg  [2:0]    rq_wr, rq_rd;
    wire          rq_full  = ((rq_wr - rq_rd) == 3'd4);
    wire          rq_empty = (rq_wr == rq_rd);

    // Requests.
    wire        req_full, req_empty;
    wire [44:0] req_out;
    fabric_async_fifo #(.W(45), .AW(REQ_AW)) u_req (
        .wclk(c_clk), .wrst_n(c_rst_n), .wr_en(c_req_go), .wdata({c_req_write, c_req_addr, c_req_beats}), .wfull(req_full), .wafull(),
        .rclk(m_clk), .rrst_n(m_rst_n), .rd_en(m_req_ready), .rdata(req_out), .rempty(req_empty));
    assign c_req_ready = !req_full && !rq_full;
    assign m_req_valid = !req_empty;
    assign {m_req_write, m_req_addr, m_req_beats} = req_out;

    // Write beats, packed MXB to a controller transfer.  Write data follows
    // its request, one request's at a time.
    reg  [11:0]     w_left;
    reg             w_wide;
    reg  [DW-1:0]   pk [0:MXB-1];
    reg  [7:0]      pk_n;
    wire [7:0]      w_in = (w_wide && w_left > 1) ? CXB : 1;
    wire            wd_full, wd_empty;
    wire            w_go  = c_wdata_valid && c_wdata_ready;
    wire            w_out = w_go && ((pk_n + w_in == MXB) || (w_left == w_in));
    reg  [MXB*DW-1:0] w_group;
    integer         pi;
    always @* begin
        for (pi = 0; pi < MXB; pi = pi + 1)
            w_group[pi*DW +: DW] = (pi < pk_n) ? pk[pi] : c_wdata[((pi - pk_n) % CXB)*DW +: DW];
    end
    fabric_async_fifo #(.W(MXB*DW), .AW(WD_AW)) u_wd (
        .wclk(c_clk), .wrst_n(c_rst_n), .wr_en(w_out), .wdata(w_group), .wfull(wd_full), .wafull(),
        .rclk(m_clk), .rrst_n(m_rst_n), .rd_en(m_wdata_ready), .rdata(m_wdata), .rempty(wd_empty));
    assign c_wdata_ready = !wd_full;
    assign m_wdata_valid = !wd_empty;
    always @(posedge c_clk or negedge c_rst_n) begin
        if (!c_rst_n) begin w_left <= 0; w_wide <= 1'b0; pk_n <= 0; end
        else begin
            if (c_req_go && c_req_write) begin w_left <= c_req_beats; w_wide <= c_wide; end
            if (w_go) begin
                w_left <= w_left - w_in;
                if (w_out) pk_n <= 0;
                else begin
                    for (pi = 0; pi < CXB; pi = pi + 1)
                        if (pi < w_in) pk[pk_n + pi] <= c_wdata[pi*DW +: DW];
                    pk_n <= pk_n + w_in;
                end
            end
        end
    end

    // Read transfers, pushed back before the FIFO can fill: the controller
    // sees the core's reads a synchroniser late, so it stops with room left.
    wire              rd_full, rd_empty, rd_afull;
    wire [MXB*DW-1:0] rd_out;
    wire              rd_pop;
    fabric_async_fifo #(.W(MXB*DW), .AW(RD_AW), .AF(6)) u_rd (
        .wclk(m_clk), .wrst_n(m_rst_n), .wr_en(m_rdata_valid), .wdata(m_rdata), .wfull(rd_full), .wafull(rd_afull),
        .rclk(c_clk), .rrst_n(c_rst_n), .rd_en(rd_pop), .rdata(rd_out), .rempty(rd_empty));
    assign m_rdata_ready = !rd_afull;
    always @(posedge m_clk or negedge m_rst_n) begin
        if (!m_rst_n) rd_overflow <= 1'b0;
        else if (m_rdata_valid && rd_full) rd_overflow <= 1'b1;
    end

    // The unpacking: a transfer is a request's next MXB beats (or its last
    // few), handed on CXB a cycle for a wide request and one for a narrow.
    reg  [11:0]   r_left;                       // beats of the current read still to hand on
    reg           r_wide;
    reg  [DW-1:0] grp [0:MXB-1];
    reg  [7:0]    g_n, g_pos;                   // beats of the held transfer left, and where they start
    wire          r_active = (r_left != 0);
    wire [7:0]    g_new    = (r_left < MXB) ? r_left[7:0] : MXB[7:0];
    assign rd_pop = r_active && (g_n == 0) && !rd_empty;
    wire [7:0]    avail    = rd_pop ? g_new : g_n;
    wire          r_emit   = r_active && (avail != 0);
    wire [7:0]    r_n      = (r_wide && avail > 1) ? CXB : 1;
    reg  [DW-1:0] src [0:MXB-1];
    integer       ri;
    always @* for (ri = 0; ri < MXB; ri = ri + 1) src[ri] = rd_pop ? rd_out[ri*DW +: DW] : grp[ri];
    wire [7:0]    base = rd_pop ? 8'd0 : g_pos;
    always @(posedge c_clk or negedge c_rst_n) begin
        if (!c_rst_n) begin
            rq_wr <= 0; rq_rd <= 0; r_left <= 0; r_wide <= 1'b0; g_n <= 0; g_pos <= 0; c_rdata_valid <= 1'b0; c_rdata <= 0;
        end else begin
            c_rdata_valid <= 1'b0;
            if (c_req_go && !c_req_write) begin rq[rq_wr[1:0]] <= {c_wide, c_req_beats}; rq_wr <= rq_wr + 1'b1; end
            if (!r_active && !rq_empty) begin
                {r_wide, r_left} <= rq[rq_rd[1:0]]; rq_rd <= rq_rd + 1'b1; g_n <= 0; g_pos <= 0;
            end else if (r_emit) begin
                c_rdata_valid <= 1'b1;
                for (ri = 0; ri < CXB; ri = ri + 1)
                    c_rdata[ri*DW +: DW] <= (ri < r_n) ? src[(base + ri) % MXB] : {DW{1'b0}};
                if (rd_pop) for (ri = 0; ri < MXB; ri = ri + 1) grp[ri] <= rd_out[ri*DW +: DW];
                g_n    <= avail - r_n;
                g_pos  <= base + r_n;
                r_left <= r_left - r_n;
            end
        end
    end
endmodule

`default_nettype wire
