// Self-checking testbench for fabric_index_scan feeding fabric_topk over the
// memory model, against fabric.memory.emit_index_scan_vectors: NQ queries
// scored in one pass over the index, each against its own blocks and into
// its own top-K.  The requests are wide, two beats a transfer.

`timescale 1ns/1ps
`default_nettype none

module tb_index_scan #(
    parameter int IDIM      = 128,
    parameter int BLOCKS    = 40,
    parameter int K         = 8,
    parameter int NQ        = 1,
    parameter int BASE      = 0,
    parameter int REC_BEATS = 5,
    parameter int RPB       = 25,
    parameter int WORDS     = 256
);
    localparam int DW = 128;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    reg [IDIM*4-1:0] qm [0:NQ-1];
    reg [15:0] counts [0:NQ-1];
    reg [15:0] en [0:NQ-1];
    reg [15:0] eid [0:NQ*K-1];
    reg [31:0] esc [0:NQ*K-1];

    wire          req_valid, req_ready, req_wide, rdata_valid;
    wire [31:0]   req_addr;
    wire [11:0]   req_beats;
    wire [2*DW-1:0] rdata;
    fabric_mem_model #(.DW(DW), .XW(2), .WORDS(WORDS), .LAT(3), .FILE("mem.hex")) mem (
        .clk(clk), .rst_n(rst_n), .req_valid(req_valid), .req_ready(req_ready), .req_write(1'b0), .req_wide(req_wide), .req_addr(req_addr),
        .req_beats(req_beats), .wdata_valid(1'b0), .wdata_ready(), .wdata({2*DW{1'b0}}), .rdata_valid(rdata_valid), .rdata(rdata));

    reg              start = 0;
    wire             scan_done;
    wire [NQ-1:0]    cand_valid;
    wire [15:0]      cand_id;
    wire [NQ*32-1:0] cand_score;
    reg  [NQ*IDIM*4-1:0] q_all;
    reg  [NQ*16-1:0]     n_all;
    fabric_index_scan #(.DW(DW), .IDIM(IDIM), .IDW(16), .RPB(RPB), .NQ(NQ)) scan (
        .clk(clk), .rst_n(rst_n), .start(start), .base(BASE[31:0]), .n_blocks(BLOCKS[15:0]), .n_q(n_all), .q_codes(q_all),
        .done(scan_done), .cand_valid(cand_valid), .cand_id(cand_id), .cand_score(cand_score),
        .req_valid(req_valid), .req_ready(req_ready), .req_wide(req_wide), .req_addr(req_addr), .req_beats(req_beats),
        .rdata_valid(rdata_valid), .rdata(rdata));

    reg clear = 0, finish = 0;
    wire [NQ-1:0] out_valid, out_last, done;
    wire [NQ*16-1:0] out_id;
    wire [NQ*32-1:0] out_score;
    integer errors, guard, q;
    integer got [0:NQ-1];
    reg [NQ-1:0] seen_done = 0;
    reg seen_scan = 0;
    genvar gq;
    generate
        for (gq = 0; gq < NQ; gq = gq + 1) begin : g_q
            fabric_topk #(.K(K), .IDW(16), .SW(32)) topk (
                .clk(clk), .rst_n(rst_n), .clear(clear), .cand_valid(cand_valid[gq]), .cand_id(cand_id),
                .cand_score($signed(cand_score[gq*32 +: 32])), .finish(finish), .out_valid(out_valid[gq]), .out_id(out_id[gq*16 +: 16]),
                .out_score(out_score[gq*32 +: 32]), .out_last(out_last[gq]), .done(done[gq]));
            always @(posedge clk) begin
                if (done[gq]) seen_done[gq] <= 1'b1;
                if (out_valid[gq]) begin
                    if (got[gq] >= en[gq] || out_id[gq*16 +: 16] !== eid[gq*K + got[gq]]
                        || out_score[gq*32 +: 32] !== esc[gq*K + got[gq]]) begin
                        errors = errors + 1;
                        if (errors <= 5) $display("query %0d rank %0d: got %0d/%0d expected %0d/%0d", gq, got[gq],
                                                  out_id[gq*16 +: 16], $signed(out_score[gq*32 +: 32]),
                                                  eid[gq*K + got[gq]], $signed(esc[gq*K + got[gq]]));
                    end
                    got[gq] = got[gq] + 1;
                end
            end
        end
    endgenerate
    integer cyc = 0, t_scan = 0;
    always @(posedge clk) begin
        cyc = cyc + 1;
        if (scan_done && !seen_scan) t_scan = cyc;
        if (scan_done) seen_scan <= 1;
    end

    initial begin
        $readmemh("q_codes.hex", qm);
        $readmemh("counts.hex", counts);
        $readmemh("expected_n.hex", en);
        $readmemh("expected_id.hex", eid);
        $readmemh("expected_score.hex", esc);
        errors = 0;
        for (q = 0; q < NQ; q = q + 1) begin
            got[q] = 0;
            q_all[q*IDIM*4 +: IDIM*4] = qm[q];
            n_all[q*16 +: 16] = counts[q];
        end
        repeat (2) @(posedge clk);
        rst_n = 1;
        @(negedge clk);
        clear = 1; start = 1;
        @(negedge clk);
        clear = 0; start = 0;
        cyc = 0;
        guard = 0;
        while (!seen_scan && guard < BLOCKS * (REC_BEATS + 8) + 20) begin @(posedge clk); #1; guard = guard + 1; end
        if (!seen_scan) begin $display("FAIL: scan never done"); $finish; end
        @(negedge clk);
        finish = 1;
        @(negedge clk);
        finish = 0;
        guard = 0;
        while (seen_done != {NQ{1'b1}} && guard < K + 10) begin @(posedge clk); #1; guard = guard + 1; end
        if (seen_done != {NQ{1'b1}}) $display("FAIL: never done");
        for (q = 0; q < NQ; q = q + 1)
            if (got[q] != en[q]) begin errors = errors + 1; $display("FAIL: query %0d: %0d entries, expected %0d", q, got[q], en[q]); end
        if (errors == 0) $display("PASS: %0d queries, top %0d of %0d blocks, %0d records per request, in %0d cycles", NQ, K, BLOCKS, RPB, t_scan);
        else $display("FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule

`default_nettype wire
