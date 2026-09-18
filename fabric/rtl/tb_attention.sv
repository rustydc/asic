// Self-checking testbench for fabric_attention against fabric.layer.emit_attention_vectors.

`timescale 1ns/1ps
`default_nettype none

module tb_attention #(
    parameter int HD        = 32,
    parameter int G         = 2,
    parameter int N         = 20,
    parameter int L         = 8,
    parameter int MULT_S    = 4096,
    parameter int SH_S      = 16,
    parameter int MULT_GATE = 1,
    parameter int SH_GATE   = 12,
    parameter int MULT_O    = 1,
    parameter int SH_O      = 24
);
    localparam int BEATS = HD / L;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg [HD*8-1:0] qm [0:G-1], gm [0:G-1], km [0:N-1], vm [0:N-1], em [0:G-1];

    reg            start = 0, in_valid = 0, finish = 0;
    reg [1:0]      in_kind = 0;
    reg [L*8-1:0]  in_data = 0;
    wire           in_ready, out_valid, done;
    wire [L*8-1:0] out_data;
    fabric_attention #(.HD(HD), .G(G), .L(L)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .in_valid(in_valid), .in_ready(in_ready), .in_kind(in_kind), .in_data(in_data),
        .finish(finish), .mult_s(MULT_S[15:0]), .sh_s(SH_S[5:0]), .mult_gate(MULT_GATE[15:0]), .sh_gate(SH_GATE[5:0]),
        .mult_o(MULT_O[15:0]), .sh_o(SH_O[5:0]), .out_valid(out_valid), .out_data(out_data), .done(done));

    integer g, n, b, errors, got, guard;
    reg seen_done = 0;
    always @(posedge clk) begin
        if (done) seen_done <= 1;
        if (out_valid) begin
            if (out_data !== em[got / BEATS][(got % BEATS)*L*8 +: L*8]) begin
                errors = errors + 1;
                if (errors <= 5) $display("out head %0d beat %0d: got %h expected %h", got / BEATS, got % BEATS, out_data, em[got / BEATS][(got % BEATS)*L*8 +: L*8]);
            end
            got = got + 1;
        end
    end

    // One beat, waiting for ready.
    task send(input [1:0] kind, input [HD*8-1:0] row, input integer beat);
        begin
            @(negedge clk);
            while (!in_ready) @(negedge clk);
            in_valid = 1; in_kind = kind; in_data = row[beat*L*8 +: L*8];
            @(posedge clk); #1;
            in_valid = 0;
        end
    endtask

    initial begin
        $readmemh("q.hex", qm);
        $readmemh("gate.hex", gm);
        $readmemh("k.hex", km);
        $readmemh("v.hex", vm);
        $readmemh("expected_out.hex", em);
        errors = 0; got = 0;
        repeat (2) @(posedge clk);
        rst_n = 1;
        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;
        for (g = 0; g < G; g = g + 1) for (b = 0; b < BEATS; b = b + 1) send(2'd0, qm[g], b);
        for (g = 0; g < G; g = g + 1) for (b = 0; b < BEATS; b = b + 1) send(2'd1, gm[g], b);
        for (n = 0; n < N; n = n + 1) begin
            for (b = 0; b < BEATS; b = b + 1) send(2'd2, km[n], b);
            for (b = 0; b < BEATS; b = b + 1) send(2'd3, vm[n], b);
        end
        @(negedge clk);
        while (!in_ready) @(negedge clk);
        finish = 1;
        @(negedge clk);
        finish = 0;
        guard = 0;
        while (!seen_done && guard < G * (BEATS + 20)) begin @(posedge clk); #1; guard = guard + 1; end
        if (!seen_done) $display("FAIL: never done");
        else if (got != G * BEATS) $display("FAIL: %0d output beats of %0d", got, G * BEATS);
        else if (errors == 0) $display("PASS: %0d heads over %0d rows", G, N);
        else $display("FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule

`default_nettype wire
