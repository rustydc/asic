// Self-checking testbench for fabric_head_gates against fabric.layer.emit_head_gate_vectors.

`timescale 1ns/1ps
`default_nettype none

module tb_head_gates #(
    parameter int H   = 32,
    parameter int ACC = 24
);
    reg clk = 0;
    always #5 clk = ~clk;

    reg [ACC-1:0] am [0:H-1], bm [0:H-1];
    reg [15:0]    mam [0:H-1], mbm [0:H-1], acm [0:H-1], dtm [0:H-1];
    reg [5:0]     sam [0:H-1], sbm [0:H-1];
    reg [15:0]    ed [0:H-1], eb [0:H-1];

    reg                  in_valid = 0;
    reg signed [ACC-1:0] a_acc = 0, b_acc = 0;
    reg [15:0]           mult_a = 0, mult_b = 0, a_coef = 0;
    reg signed [15:0]    dt_bias = 0;
    reg [5:0]            sh_a = 0, sh_b = 0;
    wire                 out_valid;
    wire [15:0]          decay, beta;
    fabric_head_gates #(.ACC(ACC)) dut (
        .clk(clk), .in_valid(in_valid), .a_acc(a_acc), .b_acc(b_acc), .mult_a(mult_a), .sh_a(sh_a), .mult_b(mult_b), .sh_b(sh_b),
        .a_coef(a_coef), .dt_bias(dt_bias), .out_valid(out_valid), .decay(decay), .beta(beta));

    integer i, errors, got;
    always @(posedge clk) begin
        if (out_valid) begin
            if (decay !== ed[got] || beta !== eb[got]) begin
                errors = errors + 1;
                if (errors <= 5) $display("head %0d: got %h/%h expected %h/%h", got, decay, beta, ed[got], eb[got]);
            end
            got = got + 1;
        end
    end

    initial begin
        $readmemh("a_acc.hex", am);
        $readmemh("b_acc.hex", bm);
        $readmemh("mult_a.hex", mam);
        $readmemh("sh_a.hex", sam);
        $readmemh("mult_b.hex", mbm);
        $readmemh("sh_b.hex", sbm);
        $readmemh("a_coef.hex", acm);
        $readmemh("dt_bias.hex", dtm);
        $readmemh("expected_decay.hex", ed);
        $readmemh("expected_beta.hex", eb);
        errors = 0; got = 0;
        repeat (2) @(posedge clk);
        for (i = 0; i < H; i = i + 1) begin
            @(negedge clk);
            in_valid = 1;
            a_acc = am[i]; b_acc = bm[i]; mult_a = mam[i]; sh_a = sam[i]; mult_b = mbm[i]; sh_b = sbm[i];
            a_coef = acm[i]; dt_bias = dtm[i];
        end
        @(negedge clk);
        in_valid = 0;
        repeat (14) @(posedge clk);
        if (got != H) $display("FAIL: %0d outputs, expected %0d", got, H);
        else if (errors == 0) $display("PASS: %0d heads", H);
        else $display("FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule

`default_nettype wire
