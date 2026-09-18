// Self-checking testbench for fabric_kv_append: TOKENS appends into the
// memory model, then the whole image against fabric.memory.emit_kv_append_vectors.

`timescale 1ns/1ps
`default_nettype none

module tb_kv_append #(
    parameter int TOKENS      = 20,
    parameter int HD          = 64,
    parameter int NKV         = 2,
    parameter int IDIM        = 32,
    parameter int BS          = 4,
    parameter int KV_BITS     = 8,
    parameter int W           = 16,
    parameter int WINDOW_BASE = 0,
    parameter int BLOCK_BASE  = 0,
    parameter int INDEX_BASE  = 0,
    parameter int WORDS       = 1024
);
    localparam int DW = 128;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    reg [NKV*HD*8-1:0] km [0:TOKENS-1], vm [0:TOKENS-1];
    reg [IDIM*8-1:0]   im [0:TOKENS-1];
    reg [DW-1:0]       expected [0:WORDS-1];

    wire          req_valid, req_ready, wdata_valid, wdata_ready;
    wire [31:0]   req_addr;
    wire [7:0]    req_beats;
    wire [DW-1:0] wdata;
    fabric_mem_model #(.DW(DW), .WORDS(WORDS), .LAT(2)) mem (
        .clk(clk), .rst_n(rst_n), .req_valid(req_valid), .req_ready(req_ready), .req_write(1'b1), .req_addr(req_addr),
        .req_beats(req_beats), .wdata_valid(wdata_valid), .wdata_ready(wdata_ready), .wdata(wdata), .rdata_valid(), .rdata());

    reg                start = 0;
    reg [31:0]         pos = 0;
    reg [NKV*HD*8-1:0] k_rows = 0, v_rows = 0;
    reg [IDIM*8-1:0]   idx_k = 0;
    wire               done;
    fabric_kv_append #(.DW(DW), .HD(HD), .NKV(NKV), .IDIM(IDIM), .BS(BS), .KV_BITS(KV_BITS), .W(W)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .pos(pos), .window_base(WINDOW_BASE[31:0]), .block_base(BLOCK_BASE[31:0]),
        .index_base(INDEX_BASE[31:0]), .k_rows(k_rows), .v_rows(v_rows), .idx_k(idx_k), .done(done),
        .req_valid(req_valid), .req_ready(req_ready), .req_addr(req_addr), .req_beats(req_beats),
        .wdata_valid(wdata_valid), .wdata_ready(wdata_ready), .wdata(wdata));

    integer t, i, errors, guard;
    reg seen_done = 0;
    always @(posedge clk) if (done) seen_done <= 1;

    initial begin
        $readmemh("k.hex", km);
        $readmemh("v.hex", vm);
        $readmemh("idx.hex", im);
        $readmemh("expected_mem.hex", expected);
        for (i = 0; i < WORDS; i = i + 1) mem.mem[i] = 0;
        errors = 0;
        repeat (2) @(posedge clk);
        rst_n = 1;
        for (t = 0; t < TOKENS; t = t + 1) begin
            @(negedge clk);
            start = 1; pos = t; k_rows = km[t]; v_rows = vm[t]; idx_k = im[t];
            seen_done = 0;
            @(negedge clk);
            start = 0;
            guard = 0;
            while (!seen_done && guard < 4000) begin @(posedge clk); #1; guard = guard + 1; end
            if (!seen_done) begin $display("FAIL: token %0d never done", t); $finish; end
        end
        repeat (4) @(posedge clk);
        for (i = 0; i < WORDS; i = i + 1)
            if (mem.mem[i] !== expected[i]) begin
                errors = errors + 1;
                if (errors <= 5) $display("beat %0d: got %h expected %h", i, mem.mem[i], expected[i]);
            end
        if (errors == 0) $display("PASS: %0d tokens, %0d beats", TOKENS, WORDS);
        else $display("FAIL: %0d mismatching beats", errors);
        $finish;
    end
endmodule

`default_nettype wire
