// Self-checking testbench: one head's state through fabric_row_dma, the
// delta engine and back, both DMA ports behind fabric_mem_arbiter on the
// memory model; the image afterwards against fabric.memory.emit_row_dma_vectors.

`timescale 1ns/1ps
`default_nettype none

module tb_row_dma #(
    parameter int K     = 16,
    parameter int V     = 16,
    parameter int DECAY = 60000,
    parameter int BETA  = 30000,
    parameter int YSH   = 9,
    parameter int BASE  = 4096,
    parameter int WORDS = 512
);
    localparam int DW = 128;
    localparam int ROW_BITS = V * 16;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    reg [7:0]      km [0:K-1], qm [0:K-1], vm [0:V-1];
    reg [15:0]     ey [0:V-1];
    reg [DW-1:0]   expected [0:WORDS-1];

    // Memory and arbiter.
    wire          m_req_valid, m_req_ready, m_req_write, m_wdata_valid, m_wdata_ready, m_rdata_valid;
    wire [31:0]   m_req_addr;
    wire [11:0]   m_req_beats;
    wire [DW-1:0] m_wdata, m_rdata;
    fabric_mem_model #(.DW(DW), .WORDS(WORDS), .LAT(3), .FILE("mem.hex")) mem (
        .clk(clk), .rst_n(rst_n), .req_valid(m_req_valid), .req_ready(m_req_ready), .req_write(m_req_write),
        .req_addr(m_req_addr), .req_beats(m_req_beats), .wdata_valid(m_wdata_valid), .wdata_ready(m_wdata_ready),
        .wdata(m_wdata), .rdata_valid(m_rdata_valid), .rdata(m_rdata));
    wire [1:0]    r_req_valid, r_req_ready, r_req_write, r_wdata_valid, r_wdata_ready, r_rdata_valid;
    wire [63:0]   r_req_addr;
    wire [23:0]   r_req_beats;
    wire [2*DW-1:0] r_wdata;
    wire [DW-1:0] r_rdata;
    fabric_mem_arbiter #(.N(2), .DW(DW)) arb (
        .clk(clk), .rst_n(rst_n), .r_req_valid(r_req_valid), .r_req_ready(r_req_ready), .r_req_write(r_req_write),
        .r_req_addr(r_req_addr), .r_req_beats(r_req_beats), .r_wdata_valid(r_wdata_valid), .r_wdata_ready(r_wdata_ready),
        .r_wdata(r_wdata), .r_rdata_valid(r_rdata_valid), .r_rdata(r_rdata),
        .m_req_valid(m_req_valid), .m_req_ready(m_req_ready), .m_req_write(m_req_write), .m_req_addr(m_req_addr),
        .m_req_beats(m_req_beats), .m_wdata_valid(m_wdata_valid), .m_wdata_ready(m_wdata_ready), .m_wdata(m_wdata),
        .m_rdata_valid(m_rdata_valid), .m_rdata(m_rdata));

    // DMA: port 0 reads, port 1 writes.
    reg  rd_start = 0, wr_start = 0;
    wire rd_done, wr_done, row_out_valid, row_in_valid;
    wire [ROW_BITS-1:0] row_out, row_in;
    assign r_req_write[0] = 1'b0;
    assign r_req_write[1] = 1'b1;
    assign r_wdata_valid[0] = 1'b0;
    assign r_wdata[DW-1:0] = {DW{1'b0}};
    fabric_row_dma #(.ROW_BITS(ROW_BITS), .DW(DW), .ROWS(K)) dma (
        .clk(clk), .rst_n(rst_n),
        .rd_start(rd_start), .rd_base(BASE[31:0]), .rd_done(rd_done), .row_out_valid(row_out_valid), .row_out(row_out),
        .rd_req_valid(r_req_valid[0]), .rd_req_ready(r_req_ready[0]), .rd_req_addr(r_req_addr[31:0]),
        .rd_req_beats(r_req_beats[11:0]), .rd_rdata_valid(r_rdata_valid[0]), .rd_rdata(r_rdata),
        .wr_start(wr_start), .wr_base(BASE[31:0]), .wr_done(wr_done), .row_in_valid(row_in_valid), .row_in(row_in),
        .wr_req_valid(r_req_valid[1]), .wr_req_ready(r_req_ready[1]), .wr_req_addr(r_req_addr[63:32]),
        .wr_req_beats(r_req_beats[23:12]), .wr_wdata_valid(r_wdata_valid[1]), .wr_wdata_ready(r_wdata_ready[1]),
        .wr_wdata(r_wdata[2*DW-1:DW]));

    // The engine.
    reg            start = 0;
    reg [K*8-1:0]  q = 0, k = 0;
    reg [V*8-1:0]  v = 0;
    wire           y_valid;
    wire [V*16-1:0] y;
    fabric_delta_state #(.K(K), .V(V), .YSH(YSH)) engine (
        .clk(clk), .rst_n(rst_n), .start(start), .q(q), .k(k), .v(v), .decay(DECAY[15:0]), .beta(BETA[15:0]),
        .row_in_valid(row_out_valid), .row_in(row_out), .row_out_valid(row_in_valid), .row_out(row_in),
        .y_valid(y_valid), .y(y));

    integer i, j, errors, guard;
    reg seen_wr = 0, seen_y = 0;
    reg [V*16-1:0] y_seen;
    always @(posedge clk) begin
        if (wr_done) seen_wr <= 1;
        if (y_valid) begin seen_y <= 1; y_seen <= y; end
    end

    initial begin
        $readmemh("k.hex", km);
        $readmemh("q.hex", qm);
        $readmemh("v.hex", vm);
        $readmemh("expected_y.hex", ey);
        $readmemh("expected_mem.hex", expected);
        for (i = 0; i < K; i = i + 1) begin q[i*8 +: 8] = qm[i]; k[i*8 +: 8] = km[i]; end
        for (i = 0; i < V; i = i + 1) v[i*8 +: 8] = vm[i];
        errors = 0;
        repeat (2) @(posedge clk);
        rst_n = 1;
        @(negedge clk);
        start = 1; rd_start = 1; wr_start = 1;
        @(negedge clk);
        start = 0; rd_start = 0; wr_start = 0;
        guard = 0;
        while (!(seen_wr && seen_y) && guard < 40 * K * (ROW_BITS / DW) + 200) begin @(posedge clk); #1; guard = guard + 1; end
        if (!seen_wr || !seen_y) begin $display("FAIL: write done %0d, y %0d", seen_wr, seen_y); $finish; end
        repeat (4) @(posedge clk);
        for (j = 0; j < V; j = j + 1)
            if (y_seen[j*16 +: 16] !== ey[j]) begin
                errors = errors + 1;
                if (errors <= 5) $display("y[%0d]: got %h expected %h", j, y_seen[j*16 +: 16], ey[j]);
            end
        for (i = 0; i < WORDS; i = i + 1)
            if (mem.mem[i] !== expected[i]) begin
                errors = errors + 1;
                if (errors <= 5) $display("beat %0d: got %h expected %h", i, mem.mem[i], expected[i]);
            end
        if (errors == 0) $display("PASS: %0d rows through memory", K);
        else $display("FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule

`default_nettype wire
