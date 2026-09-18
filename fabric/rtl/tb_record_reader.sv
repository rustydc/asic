// Self-checking testbench: fabric_record_reader reads the records a
// retrieval named from the memory model and feeds fabric_attention; the
// heads' outputs are checked against fabric.memory.emit_record_reader_vectors.

`timescale 1ns/1ps
`default_nettype none

module tb_record_reader #(
    parameter int HD        = 64,
    parameter int KV_BITS   = 8,
    parameter int G         = 2,
    parameter int L         = 16,
    parameter int N         = 18,
    parameter int REC_BEATS = 8,
    parameter int WORDS     = 1024,
    parameter int MULT_S    = 4096,
    parameter int SH_S      = 18,
    parameter int MULT_GATE = 1,
    parameter int SH_GATE   = 12,
    parameter int MULT_O    = 1,
    parameter int SH_O      = 24
);
    localparam int DW = 128;
    localparam int BEATS = HD / L;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    reg [31:0]     am [0:N-1];
    reg [HD*8-1:0] qm [0:G-1], gm [0:G-1], em [0:G-1];

    wire          req_valid, req_ready, rdata_valid;
    wire [31:0]   req_addr;
    wire [7:0]    req_beats;
    wire [DW-1:0] rdata;
    fabric_mem_model #(.DW(DW), .WORDS(WORDS), .LAT(3), .FILE("mem.hex")) mem (
        .clk(clk), .rst_n(rst_n), .req_valid(req_valid), .req_ready(req_ready), .req_write(1'b0), .req_addr(req_addr),
        .req_beats(req_beats), .wdata_valid(1'b0), .wdata_ready(), .wdata({DW{1'b0}}), .rdata_valid(rdata_valid), .rdata(rdata));

    reg           addr_valid = 0;
    reg [31:0]    addr = 0;
    wire          addr_ready, rd_valid, rd_ready, rec_done;
    wire [1:0]    rd_kind;
    wire [L*8-1:0] rd_data;
    fabric_record_reader #(.DW(DW), .HD(HD), .KV_BITS(KV_BITS), .L(L)) reader (
        .clk(clk), .rst_n(rst_n), .addr_valid(addr_valid), .addr_ready(addr_ready), .addr(addr),
        .req_valid(req_valid), .req_ready(req_ready), .req_addr(req_addr), .req_beats(req_beats),
        .rdata_valid(rdata_valid), .rdata(rdata), .out_valid(rd_valid), .out_ready(rd_ready), .out_kind(rd_kind),
        .out_data(rd_data), .rec_done(rec_done));

    // The testbench loads q and gate itself, then hands the beat stream to the reader.
    reg           tb_valid = 0, loading = 1;
    reg [1:0]     tb_kind = 0;
    reg [L*8-1:0] tb_data = 0;
    wire          in_ready, out_valid, done;
    wire [L*8-1:0] out_data;
    reg           start = 0, finish = 0;
    wire          in_valid = loading ? tb_valid : rd_valid;
    wire [1:0]    in_kind  = loading ? tb_kind : rd_kind;
    wire [L*8-1:0] in_data = loading ? tb_data : rd_data;
    assign rd_ready = !loading && in_ready;
    fabric_attention #(.HD(HD), .G(G), .L(L)) att (
        .clk(clk), .rst_n(rst_n), .start(start), .in_valid(in_valid), .in_ready(in_ready), .in_kind(in_kind), .in_data(in_data),
        .finish(finish), .mult_s(MULT_S[15:0]), .sh_s(SH_S[5:0]), .mult_gate(MULT_GATE[15:0]), .sh_gate(SH_GATE[5:0]),
        .mult_o(MULT_O[15:0]), .sh_o(SH_O[5:0]), .out_valid(out_valid), .out_data(out_data), .done(done));

    integer g, b, n, errors, got, guard, recs;
    reg seen_done = 0;
    always @(posedge clk) begin
        if (done) seen_done <= 1;
        if (rec_done) recs = recs + 1;
        if (out_valid) begin
            if (out_data !== em[got / BEATS][(got % BEATS)*L*8 +: L*8]) begin
                errors = errors + 1;
                if (errors <= 5) $display("out head %0d beat %0d: got %h expected %h", got / BEATS, got % BEATS, out_data, em[got / BEATS][(got % BEATS)*L*8 +: L*8]);
            end
            got = got + 1;
        end
    end

    task send(input [1:0] kind, input [HD*8-1:0] row, input integer beat);
        begin
            @(negedge clk);
            while (!in_ready) @(negedge clk);
            tb_valid = 1; tb_kind = kind; tb_data = row[beat*L*8 +: L*8];
            @(posedge clk); #1;
            tb_valid = 0;
        end
    endtask

    initial begin
        $readmemh("addrs.hex", am);
        $readmemh("q.hex", qm);
        $readmemh("gate.hex", gm);
        $readmemh("expected_out.hex", em);
        errors = 0; got = 0; recs = 0;
        repeat (2) @(posedge clk);
        rst_n = 1;
        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;
        for (g = 0; g < G; g = g + 1) for (b = 0; b < BEATS; b = b + 1) send(2'd0, qm[g], b);
        for (g = 0; g < G; g = g + 1) for (b = 0; b < BEATS; b = b + 1) send(2'd1, gm[g], b);
        @(negedge clk);
        loading = 0;
        for (n = 0; n < N; n = n + 1) begin
            @(negedge clk);
            while (!addr_ready) @(negedge clk);
            addr_valid = 1; addr = am[n];
            @(posedge clk); #1;
            addr_valid = 0;
        end
        guard = 0;
        while (recs < N && guard < N * (REC_BEATS + 2 * BEATS + 20)) begin @(posedge clk); #1; guard = guard + 1; end
        if (recs != N) begin $display("FAIL: %0d records read of %0d", recs, N); $finish; end
        @(negedge clk);
        while (!in_ready) @(negedge clk);
        finish = 1;
        @(negedge clk);
        finish = 0;
        guard = 0;
        while (!seen_done && guard < G * (BEATS + 20)) begin @(posedge clk); #1; guard = guard + 1; end
        if (!seen_done) $display("FAIL: attention never done");
        else if (got != G * BEATS) $display("FAIL: %0d output beats of %0d", got, G * BEATS);
        else if (errors == 0) $display("PASS: %0d records into %0d heads", N, G);
        else $display("FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule

`default_nettype wire
