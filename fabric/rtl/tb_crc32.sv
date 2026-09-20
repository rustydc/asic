// Self-checking testbench for fabric_crc32 against fabric.controller.emit_crc_vectors.

`timescale 1ns/1ps
`default_nettype none

module tb_crc32 #(
    parameter int CASES = 16,
    parameter int WORDS = 256
);
    reg clk = 0;
    always #5 clk = ~clk;
    reg [31:0] wm [0:WORDS-1];
    reg [15:0] ln [0:CASES-1];
    reg [31:0] ex [0:CASES-1];

    reg         clear = 0, valid = 0;
    reg [31:0]  data = 0;
    reg [2:0]   bytes = 0;
    wire [31:0] crc;
    fabric_crc32 dut (.clk(clk), .clear(clear), .valid(valid), .data(data), .bytes(bytes), .crc(crc));

    integer c, w, left, at, errors;
    initial begin
        $readmemh("words.hex", wm);
        $readmemh("lengths.hex", ln);
        $readmemh("expected.hex", ex);
        errors = 0; at = 0;
        @(negedge clk);
        for (c = 0; c < CASES; c = c + 1) begin
            clear = 1; @(negedge clk); clear = 0;
            left = ln[c];
            while (left > 0) begin
                valid = 1; data = wm[at]; bytes = (left >= 4) ? 3'd4 : left[2:0];
                at = at + 1; left = left - 4;
                @(negedge clk);
            end
            valid = 0;
            @(negedge clk);
            if (crc !== ex[c]) begin
                errors = errors + 1;
                if (errors <= 5) $display("case %0d: got %h expected %h", c, crc, ex[c]);
            end
        end
        if (errors == 0) $display("PASS: %0d packets", CASES);
        else $display("FAIL: %0d of %0d packets", errors, CASES);
        $finish;
    end
endmodule

`default_nettype wire
