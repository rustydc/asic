// Self-checking testbench for fabric_topk against fabric.memory.emit_topk_vectors.

`timescale 1ns/1ps
`default_nettype none

module tb_topk #(
    parameter int K        = 8,
    parameter int N        = 64,
    parameter int EXPECTED = 8
);
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    reg [23:0] sm [0:N-1];
    reg [15:0] eid [0:EXPECTED-1];
    reg [23:0] esc [0:EXPECTED-1];

    reg               clear = 0, cand_valid = 0, finish = 0;
    reg [15:0]        cand_id = 0;
    reg signed [31:0] cand_score = 0;
    wire              out_valid, out_last, done;
    wire [15:0]       out_id;
    wire signed [31:0] out_score;
    fabric_topk #(.K(K), .IDW(16), .SW(32)) dut (
        .clk(clk), .rst_n(rst_n), .clear(clear), .cand_valid(cand_valid), .cand_id(cand_id), .cand_score(cand_score),
        .finish(finish), .out_valid(out_valid), .out_id(out_id), .out_score(out_score), .out_last(out_last), .done(done));

    integer i, errors, got, guard;
    reg seen_done = 0;
    always @(posedge clk) begin
        if (done) seen_done <= 1;
        if (out_valid) begin
            if (got >= EXPECTED || out_id !== eid[got] || out_score !== $signed({{8{esc[got][23]}}, esc[got]})) begin
                errors = errors + 1;
                if (errors <= 5) $display("rank %0d: got %0d/%0d expected %0d/%0d", got, out_id, out_score, eid[got], $signed({{8{esc[got][23]}}, esc[got]}));
            end
            got = got + 1;
        end
    end

    initial begin
        $readmemh("scores.hex", sm);
        $readmemh("expected_id.hex", eid);
        $readmemh("expected_score.hex", esc);
        errors = 0; got = 0;
        repeat (2) @(posedge clk);
        rst_n = 1;
        @(negedge clk);
        clear = 1;
        @(negedge clk);
        clear = 0;
        for (i = 0; i < N; i = i + 1) begin
            cand_valid = 1; cand_id = i; cand_score = $signed({{8{sm[i][23]}}, sm[i]});
            @(negedge clk);
            if (i % 5 == 2) begin cand_valid = 0; @(negedge clk); end
        end
        cand_valid = 0;
        @(negedge clk);
        finish = 1;
        @(negedge clk);
        finish = 0;
        guard = 0;
        while (!seen_done && guard < K + 10) begin @(posedge clk); #1; guard = guard + 1; end
        if (!seen_done) $display("FAIL: never done");
        else if (got != EXPECTED) $display("FAIL: %0d entries, expected %0d", got, EXPECTED);
        else if (errors == 0) $display("PASS: top %0d of %0d", K, N);
        else $display("FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule

`default_nettype wire
