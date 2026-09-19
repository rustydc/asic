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
// beats) and write beats from the core to the controller, read beats back.
// The port has no back-pressure on read data, so the read FIFO must never
// fill: the controller produces at most one beat per 4 ns and the core drains
// one per 1.25 ns, and the FIFO holds eight; an overflow, which would mean
// the core stopped its clock, is latched into rd_overflow.

`timescale 1ns/1ps
`default_nettype none

module fabric_async_fifo #(
    parameter int W  = 128,
    parameter int AW = 4                       // 2^AW entries
) (
    input  wire         wclk,
    input  wire         wrst_n,
    input  wire         wr_en,
    input  wire [W-1:0] wdata,
    output reg          wfull,
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
// ---------------------------------------------------------------------------
module fabric_mem_bridge #(
    parameter int DW     = 128,
    parameter int REQ_AW = 2,                  // 4 requests
    parameter int WD_AW  = 4,                  // 16 write beats
    parameter int RD_AW  = 3                   // 8 read beats
) (
    // core side
    input  wire          c_clk,
    input  wire          c_rst_n,
    input  wire          c_req_valid,
    output wire          c_req_ready,
    input  wire          c_req_write,
    input  wire [31:0]   c_req_addr,
    input  wire [11:0]   c_req_beats,
    input  wire          c_wdata_valid,
    output wire          c_wdata_ready,
    input  wire [DW-1:0] c_wdata,
    output reg           c_rdata_valid,
    output reg  [DW-1:0] c_rdata,
    // controller side
    input  wire          m_clk,
    input  wire          m_rst_n,
    output wire          m_req_valid,
    input  wire          m_req_ready,
    output wire          m_req_write,
    output wire [31:0]   m_req_addr,
    output wire [11:0]   m_req_beats,
    output wire          m_wdata_valid,
    input  wire          m_wdata_ready,
    output wire [DW-1:0] m_wdata,
    input  wire          m_rdata_valid,
    input  wire [DW-1:0] m_rdata,
    output reg           rd_overflow
);
    // Requests.
    wire        req_full, req_empty;
    wire [44:0] req_out;
    fabric_async_fifo #(.W(45), .AW(REQ_AW)) u_req (
        .wclk(c_clk), .wrst_n(c_rst_n), .wr_en(c_req_valid), .wdata({c_req_write, c_req_addr, c_req_beats}), .wfull(req_full),
        .rclk(m_clk), .rrst_n(m_rst_n), .rd_en(m_req_ready), .rdata(req_out), .rempty(req_empty));
    assign c_req_ready = !req_full;
    assign m_req_valid = !req_empty;
    assign {m_req_write, m_req_addr, m_req_beats} = req_out;

    // Write beats.
    wire wd_full, wd_empty;
    fabric_async_fifo #(.W(DW), .AW(WD_AW)) u_wd (
        .wclk(c_clk), .wrst_n(c_rst_n), .wr_en(c_wdata_valid), .wdata(c_wdata), .wfull(wd_full),
        .rclk(m_clk), .rrst_n(m_rst_n), .rd_en(m_wdata_ready), .rdata(m_wdata), .rempty(wd_empty));
    assign c_wdata_ready = !wd_full;
    assign m_wdata_valid = !wd_empty;

    // Read beats: no back-pressure on either side; the core drains whatever is there.
    wire          rd_full, rd_empty;
    wire [DW-1:0] rd_out;
    fabric_async_fifo #(.W(DW), .AW(RD_AW)) u_rd (
        .wclk(m_clk), .wrst_n(m_rst_n), .wr_en(m_rdata_valid), .wdata(m_rdata), .wfull(rd_full),
        .rclk(c_clk), .rrst_n(c_rst_n), .rd_en(1'b1), .rdata(rd_out), .rempty(rd_empty));
    always @(posedge m_clk or negedge m_rst_n) begin
        if (!m_rst_n) rd_overflow <= 1'b0;
        else if (m_rdata_valid && rd_full) rd_overflow <= 1'b1;
    end
    always @(posedge c_clk or negedge c_rst_n) begin
        if (!c_rst_n) begin c_rdata_valid <= 1'b0; c_rdata <= 0; end
        else begin
            c_rdata_valid <= !rd_empty;
            c_rdata <= rd_out;
        end
    end
endmodule

`default_nettype wire
