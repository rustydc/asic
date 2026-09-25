// Self-checking testbench for fabric_die_link against
// fabric.die.emit_die_link_vectors: groups of packets go in, a stand-in
// engine runs each batch, and the batches it is started with and the packets
// that leave are the model's.  The stand-in adds to each lane's vectors a
// constant naming the program and the lane, reading and writing the vector
// buffer at the table's addresses; the link must stay off the buffer while
// it runs.

`timescale 1ns/1ps
`default_nettype none

module tb_die_link #(
    parameter int D          = 16,
    parameter int CHUNK      = 3,
    parameter int LANES      = 4,
    parameter int SLOT_PAGES = 37,
    parameter int PAGE_BASE  = 5,
    parameter int PACKETS    = 8,
    parameter int GROUPS     = 2,
    parameter int IWORDS     = 64,
    parameter int OWORDS     = 64,
    parameter int BATCHES    = 2,
    parameter int CRC_ERRORS = 0,
    parameter int MALFORMED  = 0
);
    localparam int AW = 24;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg [31:0]  im [0:IWORDS-1];
    reg [19:0]  pk [0:PACKETS-1];
    reg [31:0]  gout [0:GROUPS-1];
    reg [31:0]  om [0:OWORDS-1];
    reg [127:0] sm [0:BATCHES-1];
    reg [255:0] tm [0:7];

    reg         u_valid = 0, u_sop = 0, d_ready = 1;
    reg [31:0]  u_data = 0;
    wire        u_ready, d_valid, d_sop;
    wire [31:0] d_data;
    wire        e_start;
    wire [15:0] e_pc, e_steps;
    wire [3:0]  e_first;
    wire [4*21-1:0] e_slot_page;
    reg         e_done = 0;
    wire        v_sel, v_wr_en, v_rd_en;
    wire [AW-1:0] v_wr_addr, v_rd_addr;
    wire [127:0] v_wr_data;
    reg  [127:0] v_rd_data;
    wire [15:0] crc_errors, malformed;

    fabric_die_link #(.D(D), .CHUNK(CHUNK), .LANES(LANES), .SLOT_PAGES(SLOT_PAGES), .PAGE_BASE(PAGE_BASE),
                      .GATHER_WAIT(16), .AW(AW), .TABLE_FILE("die_table.hex")) dut (
        .clk(clk), .rst_n(rst_n),
        .u_valid(u_valid), .u_data(u_data), .u_sop(u_sop), .u_ready(u_ready),
        .d_valid(d_valid), .d_data(d_data), .d_sop(d_sop), .d_ready(d_ready),
        .e_start(e_start), .e_pc(e_pc), .e_steps(e_steps), .e_first(e_first), .e_slot_page(e_slot_page), .e_done(e_done),
        .v_sel(v_sel), .v_wr_en(v_wr_en), .v_wr_addr(v_wr_addr), .v_wr_data(v_wr_data),
        .v_rd_en(v_rd_en), .v_rd_addr(v_rd_addr), .v_rd_data(v_rd_data),
        .crc_errors(crc_errors), .malformed(malformed));

    // The vector buffer: beats, answering the cycle after the address.
    reg [127:0] vb [0:4095];
    integer errors = 0, ow = 0, nb = 0, running = 0;
    always @(posedge clk) begin
        if (v_rd_en) v_rd_data <= vb[v_rd_addr >> 4];
        if (v_wr_en) vb[v_wr_addr >> 4] <= v_wr_data;
        if ((v_wr_en || v_rd_en) && (!v_sel || running)) begin
            errors = errors + 1;
            if (errors <= 5) $display("the link used the buffer while the engine ran");
        end
    end

    // The stand-in engine.
    integer e, k, t, q, lanes_n, toks;
    reg [255:0] ent;
    reg [127:0] w;
    reg [15:0]  x;
    always @(posedge clk) if (e_start) begin
        if (nb >= BATCHES || {e_slot_page, e_first, e_steps, e_pc} !== sm[nb][119:0]) begin
            errors = errors + 1;
            if (errors <= 5) $display("batch %0d: started with %h expected %h", nb, {e_slot_page, e_first, e_steps, e_pc}, sm[nb][119:0]);
        end
        nb = nb + 1;
        running = 1;
        fork begin
            e = -1;
            for (q = 0; q < 8; q = q + 1) if (tm[q][15:0] == e_pc) e = q;
            repeat (40) @(posedge clk);
            if (e >= 0) begin
                ent = tm[e];
                lanes_n = (e & 3) + 1;
                toks = (e >> 2) ? CHUNK : 1;
                for (k = 0; k < lanes_n; k = k + 1)
                    for (t = 0; t < toks * D / 8; t = t + 1) begin
                        w = vb[(ent[32 + 24*k +: 24] >> 4) + t];
                        for (q = 0; q < 8; q = q + 1) begin
                            x = w[16*q +: 16] + 16'd1 + 16'(3 * k) + 16'(7 * e);
                            w[16*q +: 16] = x;
                        end
                        vb[(ent[128 + 24*k +: 24] >> 4) + t] = w;
                    end
            end
            @(posedge clk); #1 e_done = 1; running = 0;
            @(posedge clk); #1 e_done = 0;
        end join_none
    end

    // What leaves, against the model's words.
    always @(posedge clk) if (d_valid && d_ready) begin
        if (ow >= OWORDS || d_data !== om[ow]) begin
            errors = errors + 1;
            if (errors <= 5) $display("out word %0d: got %h expected %h", ow, d_data, ow < OWORDS ? om[ow] : 0);
        end
        ow = ow + 1;
    end
    // The next die stalls now and then.
    always @(posedge clk) d_ready <= ($random & 3) != 0;

    initial begin
        #2000000;
        $display("FAIL: stuck: %0d out words of %0d, %0d batches of %0d", ow, OWORDS, nb, BATCHES);
        $finish;
    end

    integer p, g, iw, guard, j;
    initial begin
        $readmemh("in.hex", im);
        $readmemh("pk.hex", pk);
        $readmemh("gout.hex", gout);
        $readmemh("out.hex", om);
        $readmemh("starts.hex", sm);
        $readmemh("die_table.hex", tm);
        repeat (2) @(posedge clk);
        #1 rst_n = 1;
        iw = 0; g = 0;
        for (p = 0; p < PACKETS; p = p + 1) begin
            for (j = 0; j < pk[p][15:0]; j = j + 1) begin
                u_valid = 1; u_sop = (j == 0); u_data = im[iw + j];
                @(posedge clk);
                while (!u_ready) @(posedge clk);
                #1;
                if (j % 7 == 6) begin u_valid = 0; @(posedge clk); #1; end     // a gap the link must wait out
            end
            u_valid = 0; u_sop = 0;
            iw = iw + pk[p][15:0];
            if (pk[p][16]) begin
                // The end of a group: quiet until its batches are out.
                guard = 0;
                while (ow < gout[g] && guard < 100000) begin @(posedge clk); guard = guard + 1; end
                if (ow < gout[g]) begin
                    $display("FAIL: group %0d: %0d out words of %0d; link r%0d x%0d lanes %0d, %0d batches",
                             g, ow, gout[g], dut.rstate, dut.xstate, dut.lanes, nb);
                    $finish;
                end
                repeat (40) @(posedge clk);
                #1 g = g + 1;
            end
        end
        if (ow != OWORDS) begin errors = errors + 1; $display("%0d out words, expected %0d", ow, OWORDS); end
        if (nb != BATCHES) begin errors = errors + 1; $display("%0d batches, expected %0d", nb, BATCHES); end
        if (crc_errors != CRC_ERRORS || malformed != MALFORMED) begin
            errors = errors + 1;
            $display("dropped %0d for the CRC and %0d malformed, expected %0d and %0d", crc_errors, malformed, CRC_ERRORS, MALFORMED);
        end
        if (errors == 0) $display("PASS: %0d packets, %0d batches, %0d words out", PACKETS, BATCHES, OWORDS);
        else $display("FAIL: %0d errors", errors);
        $finish;
    end
endmodule

`default_nettype wire
