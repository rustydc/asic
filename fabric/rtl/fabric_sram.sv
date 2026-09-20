// A memory macro: one write port with a byte mask, NRD read ports, the data a
// cycle after the address.  The body is here because the testbenches have to
// run something; the blackbox attribute is for synthesis, which has to stop at
// the boundary.  A flow that maps one of these to flops does not report a
// memory at all -- it reports the mux that selects a word, and the counter
// driving it as a net with a thousand loads, which is what the attention core
// and the append were reporting before they were brought in as macros.
//
// Reads see the word as it was before a write to the same address in the same
// cycle, which is what the arrays these replaced did.

`timescale 1ns/1ps
`default_nettype none

(* blackbox *)
module fabric_sram #(
    parameter int W   = 128,                      // bits per word
    parameter int D   = 256,                      // words
    parameter int NRD = 1,                        // read ports
    parameter int NWR = 1,                        // write ports
    parameter int MB  = 8,                        // bits per write-mask bit
    parameter int AW  = (D > 1) ? $clog2(D) : 1   // derived; callers leave it alone
) (
    input  wire                 clk,
    input  wire [NRD-1:0]       rd_en,
    input  wire [NRD*AW-1:0]    rd_addr,
    output reg  [NRD*W-1:0]     rd_data,
    input  wire [NWR-1:0]       wr_en,
    input  wire [NWR*AW-1:0]    wr_addr,
    input  wire [NWR*W-1:0]     wr_data,
    input  wire [NWR*(W/MB)-1:0] wr_mask
);
    reg [W-1:0] mem [0:D-1];
    integer p, m;
    always @(posedge clk) begin
        for (p = 0; p < NRD; p = p + 1)
            rd_data[p*W +: W] <= rd_en[p] ? mem[rd_addr[p*AW +: AW]] : {W{1'bx}};
        for (p = 0; p < NWR; p = p + 1)
            if (wr_en[p])
                for (m = 0; m < W/MB; m = m + 1)   // blocking: the reads above precede it, and Verilator wants it so
                    if (wr_mask[p*(W/MB) + m]) mem[wr_addr[p*AW +: AW]][m*MB +: MB] = wr_data[p*W + m*MB +: MB];
    end
endmodule

`default_nettype wire
