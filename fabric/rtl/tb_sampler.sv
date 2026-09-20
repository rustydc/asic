// Self-checking testbench for fabric_sampler against fabric.controller.emit_sampler_vectors.

`timescale 1ns/1ps
`default_nettype none

module tb_sampler #(
    parameter int K     = 32,
    parameter int CASES = 20
);
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    reg [63:0] l0 [0:CASES*(K+1)-1];
    reg [63:0] l1 [0:CASES*(K+1)-1];
    reg [63:0] pm [0:CASES-1];
    reg [31:0] rn [0:CASES-1];
    reg [39:0] ex [0:CASES-1];

    reg               clear = 0, in_valid = 0, in_list = 0, finish = 0;
    reg [31:0]        in_row = 0;
    reg signed [31:0] in_logit = 0;
    reg [15:0]        inv_t = 0, top_p = 0;
    reg [7:0]         top_k = 0;
    reg [31:0]        rnd = 0;
    wire              out_valid;
    wire [31:0]       out_row;
    wire [7:0]        out_index;
    fabric_sampler #(.K(K), .LUT_DIR("./")) dut (
        .clk(clk), .rst_n(rst_n), .clear(clear), .in_valid(in_valid), .in_list(in_list), .in_row(in_row), .in_logit(in_logit),
        .finish(finish), .inv_t(inv_t), .top_k(top_k), .top_p(top_p), .rnd(rnd),
        .out_valid(out_valid), .out_row(out_row), .out_index(out_index));

    integer c, i, n, errors, guard;
    reg seen = 0;
    reg [31:0] got_row;
    reg [7:0]  got_index;
    always @(posedge clk) if (out_valid) begin seen <= 1; got_row <= out_row; got_index <= out_index; end

    initial begin
        $readmemh("list0.hex", l0);
        $readmemh("list1.hex", l1);
        $readmemh("params.hex", pm);
        $readmemh("rnd.hex", rn);
        $readmemh("expected.hex", ex);
        errors = 0;
        repeat (2) @(posedge clk);
        rst_n = 1;
        for (c = 0; c < CASES; c = c + 1) begin
            @(negedge clk);
            clear = 1; @(negedge clk); clear = 0;
            inv_t = pm[c][63:48]; top_k = pm[c][47:40]; top_p = pm[c][39:24]; rnd = rn[c];
            n = l0[c*(K+1)];
            for (i = 0; i < n; i = i + 1) begin
                in_valid = 1; in_list = 0; in_row = l0[c*(K+1) + 1 + i][63:32]; in_logit = l0[c*(K+1) + 1 + i][31:0];
                @(negedge clk);
            end
            n = l1[c*(K+1)];
            for (i = 0; i < n; i = i + 1) begin
                in_valid = 1; in_list = 1; in_row = l1[c*(K+1) + 1 + i][63:32]; in_logit = l1[c*(K+1) + 1 + i][31:0];
                @(negedge clk);
            end
            in_valid = 0;
            seen = 0;
            finish = 1; @(negedge clk); finish = 0;
            guard = 0;
            while (!seen && guard < 2000) begin @(posedge clk); #1; guard = guard + 1; end
            if (!seen) begin $display("FAIL: case %0d never answered", c); $finish; end
            if (got_row !== ex[c][39:8] || got_index !== ex[c][7:0]) begin
                errors = errors + 1;
                if (errors <= 5) $display("case %0d: got row %0d index %0d, expected row %0d index %0d", c, got_row, got_index, ex[c][39:8], ex[c][7:0]);
            end
        end
        if (errors == 0) $display("PASS: %0d cases", CASES);
        else $display("FAIL: %0d of %0d cases", errors, CASES);
        $finish;
    end
endmodule

`default_nettype wire
