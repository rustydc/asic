// The vector buffer's two read paths side by side: the gather the engine is
// built with (GATHER = 1, what synthesis gets) and the direct read the
// simulations take (GATHER = 0).  One random stream drives both -- reads at
// any byte offset, ports sharing an address, a wide read and a wide write,
// byte masks, the ports folded onto the crossbar or not -- and every port's
// answer must be the same, bit for bit and X for X, every cycle.  The
// stream keeps to what the engine may ask: a bank no more reads than it
// has, one write a bank, and one of a folded crossbar port's sharers at once.

`timescale 1ns/1ps
`default_nettype none

module tb_vb_paths #(
    parameter int NR     = 10,
    parameter int NW     = 6,
    parameter int NB     = 8,
    parameter int BSH    = 9,
    parameter int FOLD   = 0,
    parameter int CYCLES = 4000,
    parameter int SEED   = 1
);
    localparam int AW = BSH + $clog2(NB) + 1;
    localparam int BYTES = NB << BSH;
    // Folded, logical port i shares crossbar port i / 2; a map of four bits a port.
    function automatic [63:0] pair_map(input integer half);
        integer i;
        begin
            pair_map = 0;
            for (i = 0; i < 16; i = i + 1) pair_map[i*4 +: 4] = 4'((half * 16 + i) / 2);
        end
    endfunction
    localparam [63:0] MAP0 = pair_map(0), MAP1 = pair_map(1);
    localparam int NPR = FOLD ? (NR + 1) / 2 : NR;
    localparam int NPW = FOLD ? (NW + 1) / 2 : NW;
    localparam [63:0] ALL = {64{1'b1}};

    reg clk = 0;
    always #5 clk = ~clk;

    reg  [NR-1:0]     rd_en = 0;
    reg  [NR*AW-1:0]  rd_addr = 0;
    reg  [NW-1:0]     wr_en = 0;
    reg  [NW*AW-1:0]  wr_addr = 0;
    reg  [NW*128-1:0] wr_data = 0;
    reg  [NW*16-1:0]  wr_be = 0;
    reg  [127:0]      wr_hi = 0;
    reg  [15:0]       wr_hi_be = 0;
    wire [NR*128-1:0] rd_a, rd_b;
    wire [127:0]      hi_a, hi_b;

    fabric_vb #(.BYTES(BYTES), .NR(NR), .NW(NW), .AW(AW), .NB(NB), .BSH(BSH), .RCAP2(ALL), .RCAP3(0), .WCAP2(0),
                .FOLD(FOLD), .NPR(NPR), .NPW(NPW), .RMAP0(MAP0), .RMAP1(MAP1), .WMAP0(MAP0), .WMAP1(MAP1),
                .WIDE_R(0), .WIDE_W(0), .GATHER(1), .INIT_FILE("vb.hex")) u_gather (
        .clk(clk), .rd_en(rd_en), .rd_addr(rd_addr), .rd_data(rd_a), .rd_hi(hi_a), .wr_en(wr_en), .wr_addr(wr_addr),
        .wr_data(wr_data), .wr_be(wr_be), .wr_hi(wr_hi), .wr_hi_be(wr_hi_be));
    fabric_vb #(.BYTES(BYTES), .NR(NR), .NW(NW), .AW(AW), .NB(NB), .BSH(BSH), .RCAP2(ALL), .RCAP3(0), .WCAP2(0),
                .FOLD(FOLD), .NPR(NPR), .NPW(NPW), .RMAP0(MAP0), .RMAP1(MAP1), .WMAP0(MAP0), .WMAP1(MAP1),
                .WIDE_R(0), .WIDE_W(0), .GATHER(0), .INIT_FILE("vb.hex")) u_direct (
        .clk(clk), .rd_en(rd_en), .rd_addr(rd_addr), .rd_data(rd_b), .rd_hi(hi_b), .wr_en(wr_en), .wr_addr(wr_addr),
        .wr_data(wr_data), .wr_be(wr_be), .wr_hi(wr_hi), .wr_hi_be(wr_hi_be));

    // The answers, compared half a cycle after the edge that makes them.
    integer diffs = 0, known = 0, j;
    always @(negedge clk) begin
        for (j = 0; j < NR; j = j + 1) begin
            if (rd_a[j*128 +: 128] !== rd_b[j*128 +: 128]) begin
                diffs = diffs + 1;
                if (diffs <= 5) $display("port %0d: gather %h direct %h", j, rd_a[j*128 +: 128], rd_b[j*128 +: 128]);
            end
            if (^rd_a[j*128 +: 128] !== 1'bx) known = known + 1;
        end
        if (hi_a !== hi_b) begin
            diffs = diffs + 1;
            if (diffs <= 5) $display("wide: gather %h direct %h", hi_a, hi_b);
        end
    end

    // A random address in a bank, anywhere a sixteen-byte access fits whole
    // (a read reaches into the next word), or on a word boundary.
    function automatic [AW-1:0] addr_in(input integer bank, input bit aligned);
        reg [31:0] off;
        begin
            off = $urandom % ((1 << BSH) - 32);
            if (aligned) off = off & ~32'd15;
            addr_in = AW'((bank << BSH) + off);
        end
    endfunction

    integer c, i, r0, w0, b, seed;
    initial begin
        seed = SEED;
        r0 = $urandom(seed);
        repeat (3) @(posedge clk);
        for (c = 0; c < CYCLES; c = c + 1) begin
            #1;
            r0 = $urandom % NB; w0 = $urandom % NB;
            rd_en = 0; wr_en = 0;
            for (i = 0; i < NR; i = i + 1)
                // Two ports a bank at most (every bank has two), and folded
                // only one of a pair asking.
                if (!(FOLD && (i % 2) && rd_en[i - 1]) && ($urandom % 4 != 0)) begin
                    rd_en[i] = 1'b1;
                    b = (i + r0) % NB;
                    if (i > 0 && rd_en[i - 1] && $urandom % 6 == 0 && !FOLD)
                        rd_addr[i*AW +: AW] = rd_addr[(i-1)*AW +: AW];      // one address, one bank port, two readers
                    else
                        rd_addr[i*AW +: AW] = addr_in(b, i == 0 && $urandom % 2);
                end
            for (i = 0; i < NW; i = i + 1)
                if (!(FOLD && (i % 2) && wr_en[i - 1]) && ($urandom % 3 != 0)) begin
                    wr_en[i] = 1'b1;
                    wr_addr[i*AW +: AW] = addr_in((i + w0) % NB, i == 0);
                    wr_data[i*128 +: 128] = {$urandom, $urandom, $urandom, $urandom};
                    wr_be[i*16 +: 16] = ($urandom % 3 == 0) ? 16'hFFFF : 16'($urandom);
                end
            wr_hi = {$urandom, $urandom, $urandom, $urandom};
            wr_hi_be = ($urandom % 2) ? 16'hFFFF : 16'($urandom);
            @(posedge clk);
        end
        #1 rd_en = 0; wr_en = 0;
        repeat (4) @(posedge clk);
        if (diffs == 0 && known > CYCLES) $display("PASS: %0d cycles, %0d answers compared", CYCLES, known);
        else $display("FAIL: %0d differences, %0d known answers", diffs, known);
        $finish;
    end
endmodule

`default_nettype wire
