// Self-checking testbench for fabric_delta_state8 against fabric.layer.emit_delta8_vectors.

`timescale 1ns/1ps
`default_nettype none

module tb_delta_state8 #(
    parameter int K     = 16,
    parameter int V     = 16,
    parameter int VL    = V,
    parameter int DECAY = 60000,
    parameter int BETA  = 30000,
    parameter int G     = 40000,
    parameter int E     = 0,
    parameter int PEAK  = 0,
    parameter int EXPECTED_G = 40000,
    parameter int EXPECTED_E = 0,
    parameter int EXPECTED_PEAK = 0,
    parameter int NSAT = 0,
    parameter int EXPECTED_NSAT = 0,
    parameter int PEAK_GROW = 47,
    parameter int SAT_SHIFT = 6,
    parameter int YSH   = 9
);
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg [V*8-1:0]  smem [0:K-1], esmem [0:K-1];
    reg [7:0]      km [0:K-1], qm [0:K-1], vm [0:V-1];
    reg [15:0]     ey [0:V-1];

    reg            start = 0;
    reg [K*8-1:0]  q = 0, k = 0;
    reg [V*8-1:0]  v = 0;
    localparam int SL = V / VL;
    reg            row_in_valid = 0;
    reg [V*8-1:0]  row_in = 0;
    wire           row_out_valid, y_valid;
    wire [V*8-1:0] row_out;
    wire [V*16-1:0] y;
    wire [15:0]    g_out;
    wire [7:0]     e_out, peak_out;
    wire [15:0]    nsat_out;
    fabric_delta_state8 #(.K(K), .V(V), .VL(VL), .YSH(YSH), .PEAK_GROW(PEAK_GROW), .SAT_SHIFT(SAT_SHIFT)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .q(q), .k(k), .v(v), .decay(DECAY[15:0]), .beta(BETA[15:0]),
        .g_in(G[15:0]), .e_in(E[7:0]), .peak_in(PEAK[7:0]), .nsat_in(NSAT[15:0]), .g_out(g_out), .e_out(e_out), .peak_out(peak_out), .nsat_out(nsat_out),
        .row_in_valid(row_in_valid), .row_in(row_in), .row_out_valid(row_out_valid),
        .row_out(row_out), .y_valid(y_valid), .y(y));

    integer i, j, errors, rows, seen_y;
    always @(posedge clk) begin
        if (row_out_valid) begin
            if (row_out !== esmem[rows]) begin
                errors = errors + 1;
                if (errors <= 5) $display("row %0d: got %h expected %h", rows, row_out, esmem[rows]);
            end
            rows = rows + 1;
        end
        if (y_valid) begin
            seen_y = seen_y + 1;
            for (j = 0; j < V; j = j + 1)
                if (y[j*16 +: 16] !== ey[j]) begin
                    errors = errors + 1;
                    if (errors <= 5) $display("y[%0d]: got %h expected %h", j, y[j*16 +: 16], ey[j]);
                end
        end
    end

    initial begin
        $readmemh("s.hex", smem);
        $readmemh("expected_s.hex", esmem);
        $readmemh("k.hex", km);
        $readmemh("q.hex", qm);
        $readmemh("v.hex", vm);
        $readmemh("expected_y.hex", ey);
        for (i = 0; i < K; i = i + 1) begin q[i*8 +: 8] = qm[i]; k[i*8 +: 8] = km[i]; end
        for (i = 0; i < V; i = i + 1) v[i*8 +: 8] = vm[i];
        errors = 0; rows = 0; seen_y = 0;
        repeat (2) @(posedge clk);
        rst_n = 1;
        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;
        for (i = 0; i < K; i = i + 1) begin
            row_in_valid = 1; row_in = smem[i];
            @(negedge clk);
            // A row of V bytes reaches the engine over V/16 beats, which is
            // the rate its V/VL slices consume one at; idle between rows the
            // way the adapter does, plus the odd extra gap.
            row_in_valid = 0;
            if (SL > 1) repeat (SL - 1) @(negedge clk);
            else if (i % 4 == 1) @(negedge clk);
        end
        row_in_valid = 0;
        repeat ((K + 80) * SL) @(posedge clk);
        if (g_out !== EXPECTED_G[15:0]) begin errors = errors + 1; $display("scale: got %0d expected %0d", g_out, EXPECTED_G); end
        if (e_out !== EXPECTED_E[7:0]) begin errors = errors + 1; $display("exponent: got %0d expected %0d", $signed(e_out), $signed(EXPECTED_E[7:0])); end
        if (peak_out !== EXPECTED_PEAK[7:0]) begin errors = errors + 1; $display("peak: got %0d expected %0d", peak_out, EXPECTED_PEAK); end
        if (nsat_out !== EXPECTED_NSAT[15:0]) begin errors = errors + 1; $display("saturated: got %0d expected %0d", nsat_out, EXPECTED_NSAT); end
        if (rows != K || seen_y != 1) $display("FAIL: %0d rows and %0d y outputs", rows, seen_y);
        else if (errors == 0) $display("PASS: %0dx%0d int8 state, scale %0d -> %0d, exponent %0d -> %0d, peak %0d, %0d saturated", K, V, G, g_out, $signed(E[7:0]), $signed(e_out), peak_out, nsat_out);
        else $display("FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule

`default_nettype wire
