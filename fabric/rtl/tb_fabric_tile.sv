// Self-checking testbench for fabric_tile.  Run from a directory containing
// the vector files written by fabric.tile.emit_vectors:
//   rom.hex x.hex psum_in.hex mult.hex shift.hex expected_psum.hex expected_q.hex
// Prints PASS or FAIL and the mismatch count.

`timescale 1ns/1ps
`default_nettype none

module tb_fabric_tile #(
    parameter int ROWS = 4096,
    parameter int COLS = 64,
    parameter int WB   = 4,
    parameter int AB   = 8,
    parameter int P    = 2,
    parameter int ACC  = 24,
    parameter int SB   = 16,
    parameter int SHB  = 5
);
    localparam int CYCLES = ROWS / P;

    reg                 clk = 0;
    reg                 rst_n = 0;
    reg                 start = 0;
    reg  [COLS*ACC-1:0] psum_in = '0;
    reg                 x_valid = 0;
    reg  [P*AB-1:0]     x_data = '0;
    reg  [COLS*SB-1:0]  mult = '0;
    reg  [COLS*SHB-1:0] shift = '0;
    wire                x_ready;
    wire                done;
    wire                q_valid;
    wire [COLS*ACC-1:0] psum_out;
    wire [COLS*AB-1:0]  q_out;

    fabric_tile #(.ROWS(ROWS), .COLS(COLS), .WB(WB), .AB(AB), .P(P), .ACC(ACC), .SB(SB), .SHB(SHB),
                  .ROM_FILE("rom.hex")) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .psum_in(psum_in), .x_valid(x_valid), .x_data(x_data),
        .mult(mult), .shift(shift), .x_ready(x_ready), .done(done), .psum_out(psum_out), .q_out(q_out),
        .q_valid(q_valid));

    always #5 clk = ~clk;

    reg [AB-1:0]  xmem   [0:ROWS-1];
    reg [ACC-1:0] pmem   [0:COLS-1];
    reg [SB-1:0]  mmem   [0:COLS-1];
    reg [SHB-1:0] smem   [0:COLS-1];
    reg [ACC-1:0] epsum  [0:COLS-1];
    reg [AB-1:0]  eq     [0:COLS-1];

    integer i, b, errors, cyc, guard;

    // Sticky flags so one-cycle pulses cannot race the checks below.
    reg seen_done = 0, seen_q_valid = 0;
    always @(posedge clk) begin
        if (done)    seen_done <= 1;
        if (q_valid) seen_q_valid <= 1;
    end

    initial begin
        $readmemh("x.hex", xmem);
        $readmemh("psum_in.hex", pmem);
        $readmemh("mult.hex", mmem);
        $readmemh("shift.hex", smem);
        $readmemh("expected_psum.hex", epsum);
        $readmemh("expected_q.hex", eq);
        for (i = 0; i < COLS; i = i + 1) begin
            psum_in[i*ACC +: ACC] = pmem[i];
            mult[i*SB +: SB]      = mmem[i];
            shift[i*SHB +: SHB]   = smem[i];
        end

        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;

        // Stream activations with an occasional bubble to exercise x_valid.
        cyc = 0;
        while (cyc < CYCLES) begin
            @(negedge clk);
            if (cyc % 7 == 3 && !x_valid) begin
                x_valid = 0;               // one-cycle bubble
                @(negedge clk);
            end
            for (b = 0; b < P; b = b + 1) x_data[b*AB +: AB] = xmem[cyc*P + b];
            x_valid = 1;
            @(posedge clk);
            #1;
            cyc = cyc + 1;
        end
        @(negedge clk);
        x_valid = 0;

        guard = 0;
        while (!seen_done && guard < 10) begin
            @(posedge clk);
            #1;
            guard = guard + 1;
        end
        if (!seen_done) begin
            $display("FAIL: done never asserted");
            $finish;
        end

        errors = 0;
        for (i = 0; i < COLS; i = i + 1) begin
            if (psum_out[i*ACC +: ACC] !== epsum[i]) begin
                errors = errors + 1;
                if (errors <= 5)
                    $display("psum mismatch col %0d: got %h expected %h", i, psum_out[i*ACC +: ACC], epsum[i]);
            end
        end

        // The shared requantizer walks the columns after done.
        guard = 0;
        while (!seen_q_valid && guard < COLS + 4) begin
            @(posedge clk);
            #1;
            guard = guard + 1;
        end
        if (!seen_q_valid) begin
            $display("FAIL: q_valid never asserted");
            $finish;
        end
        for (i = 0; i < COLS; i = i + 1) begin
            if (q_out[i*AB +: AB] !== eq[i]) begin
                errors = errors + 1;
                if (errors <= 5)
                    $display("q mismatch col %0d: got %h expected %h", i, q_out[i*AB +: AB], eq[i]);
            end
        end
        if (errors == 0) $display("PASS: %0d columns, %0d cycles", COLS, CYCLES);
        else             $display("FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule

`default_nettype wire
