// Self-checking testbench for fabric_index_scan feeding fabric_topk over the
// memory model, against fabric.memory.emit_index_scan_vectors.

`timescale 1ns/1ps
`default_nettype none

module tb_index_scan #(
    parameter int IDIM      = 128,
    parameter int BLOCKS    = 40,
    parameter int K         = 8,
    parameter int BASE      = 0,
    parameter int REC_BEATS = 5,
    parameter int RPB       = 25,
    parameter int WORDS     = 256,
    parameter int EXPECTED  = 8
);
    localparam int DW = 128;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    reg [IDIM*4-1:0] qm [0:0];
    reg [15:0] eid [0:EXPECTED-1];
    reg [31:0] esc [0:EXPECTED-1];

    wire          req_valid, req_ready, rdata_valid;
    wire [31:0]   req_addr;
    wire [11:0]   req_beats;
    wire [DW-1:0] rdata;
    fabric_mem_model #(.DW(DW), .WORDS(WORDS), .LAT(3), .FILE("mem.hex")) mem (
        .clk(clk), .rst_n(rst_n), .req_valid(req_valid), .req_ready(req_ready), .req_write(1'b0), .req_addr(req_addr),
        .req_beats(req_beats), .wdata_valid(1'b0), .wdata_ready(), .wdata({DW{1'b0}}), .rdata_valid(rdata_valid), .rdata(rdata));

    reg              start = 0;
    wire             scan_done, cand_valid;
    wire [15:0]      cand_id;
    wire signed [31:0] cand_score;
    fabric_index_scan #(.DW(DW), .IDIM(IDIM), .IDW(16), .RPB(RPB)) scan (
        .clk(clk), .rst_n(rst_n), .start(start), .base(BASE[31:0]), .n_blocks(BLOCKS[15:0]), .q_codes(qm[0]),
        .done(scan_done), .cand_valid(cand_valid), .cand_id(cand_id), .cand_score(cand_score),
        .req_valid(req_valid), .req_ready(req_ready), .req_addr(req_addr), .req_beats(req_beats),
        .rdata_valid(rdata_valid), .rdata(rdata));

    reg clear = 0, finish = 0;
    wire out_valid, out_last, done;
    wire [15:0] out_id;
    wire signed [31:0] out_score;
    fabric_topk #(.K(K), .IDW(16), .SW(32)) topk (
        .clk(clk), .rst_n(rst_n), .clear(clear), .cand_valid(cand_valid), .cand_id(cand_id), .cand_score(cand_score),
        .finish(finish), .out_valid(out_valid), .out_id(out_id), .out_score(out_score), .out_last(out_last), .done(done));

    integer errors, got, guard;
    reg seen_scan = 0, seen_done = 0;
    always @(posedge clk) begin
        if (scan_done) seen_scan <= 1;
        if (done) seen_done <= 1;
        if (out_valid) begin
            if (got >= EXPECTED || out_id !== eid[got] || out_score !== $signed(esc[got])) begin
                errors = errors + 1;
                if (errors <= 5) $display("rank %0d: got %0d/%0d expected %0d/%0d", got, out_id, out_score, eid[got], $signed(esc[got]));
            end
            got = got + 1;
        end
    end

    initial begin
        $readmemh("q_codes.hex", qm);
        $readmemh("expected_id.hex", eid);
        $readmemh("expected_score.hex", esc);
        errors = 0; got = 0;
        repeat (2) @(posedge clk);
        rst_n = 1;
        @(negedge clk);
        clear = 1; start = 1;
        @(negedge clk);
        clear = 0; start = 0;
        guard = 0;
        while (!seen_scan && guard < BLOCKS * (REC_BEATS + 8) + 20) begin @(posedge clk); #1; guard = guard + 1; end
        if (!seen_scan) begin $display("FAIL: scan never done"); $finish; end
        @(negedge clk);
        finish = 1;
        @(negedge clk);
        finish = 0;
        guard = 0;
        while (!seen_done && guard < K + 10) begin @(posedge clk); #1; guard = guard + 1; end
        if (!seen_done) $display("FAIL: never done");
        else if (got != EXPECTED) $display("FAIL: %0d entries, expected %0d", got, EXPECTED);
        else if (errors == 0) $display("PASS: top %0d of %0d blocks, %0d records per request", K, BLOCKS, RPB);
        else $display("FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule

`default_nettype wire
