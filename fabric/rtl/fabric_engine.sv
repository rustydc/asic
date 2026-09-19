// The layer engine: the token sequencer (fabric_sequencer.sv) driving the
// real units through one byte-addressed vector buffer.
//
// Every unit sits behind an adapter with the same face to the controller:
// a command (length, source and destination byte addresses, two 32-bit
// arguments, a tag) accepted when the engine is idle, and a done pulse
// carrying the tag one cycle after the step's last result was written to
// the vector buffer.  The adapters read the buffer 16 bytes per port per
// cycle (one cycle of latency) and write 16 bytes with byte enables.
//
//   fabric_vb              the vector buffer (a multi-port SRAM stand-in)
//   fabric_norm_adapter    one norm engine: int8 or int16 input, a gain of
//                          one or silu of a requantized int8 vector
//   fabric_pass_adapter    the tile array: a pass is a set of tiles run
//                          on one broadcast activation stream, row blocks
//                          chained through their partial sums
//   fabric_conv_adapter    the causal conv and its history
//   fabric_gates_adapter   the per-head gates from the raw accumulators
//   fabric_delta_adapter   one int8 state engine over a state slot
//   fabric_swiglu_adapter, fabric_residual_adapter
//   fabric_mem_adapter     beats between the memory port and the buffer
//   fabric_layer_engine    all of it behind start / done
//
// The program and every table come from fabric/engine.py, whose run of
// the same steps on the integer model the engine must reproduce bit for bit.

`timescale 1ns/1ps
`default_nettype none
`include "fabric_fx.svh"

// ---------------------------------------------------------------------------
// The vector buffer.
// ---------------------------------------------------------------------------
module fabric_vb #(
    parameter int BYTES = 4096,
    parameter int NR    = 1,
    parameter int NW    = 1,
    parameter int AW    = 16,
    parameter     INIT_FILE = ""
) (
    input  wire              clk,
    input  wire [NR*AW-1:0]  rd_addr,
    output reg  [NR*128-1:0] rd_data,
    input  wire [NW-1:0]     wr_en,
    input  wire [NW*AW-1:0]  wr_addr,
    input  wire [NW*128-1:0] wr_data,
    input  wire [NW*16-1:0]  wr_be
);
    reg [7:0] mem [0:BYTES-1];
    integer i, p, b;
    initial begin
        if (INIT_FILE != "") $readmemh(INIT_FILE, mem);
        else for (i = 0; i < BYTES; i = i + 1) mem[i] = 8'd0;
    end
    function automatic [127:0] rd16(input [AW-1:0] a);
        integer k;
        for (k = 0; k < 16; k = k + 1) rd16[k*8 +: 8] = (a + k < BYTES) ? mem[a + k] : 8'd0;
    endfunction
    always @(posedge clk) begin
        for (p = 0; p < NR; p = p + 1) rd_data[p*128 +: 128] <= rd16(rd_addr[p*AW +: AW]);
        for (p = 0; p < NW; p = p + 1)
            if (wr_en[p])
                for (b = 0; b < 16; b = b + 1)
                    if (wr_be[p*16 + b] && (wr_addr[p*AW +: AW] + b < BYTES))
                        mem[wr_addr[p*AW +: AW] + b] <= wr_data[p*128 + b*8 +: 8];
    end
endmodule

// ---------------------------------------------------------------------------
// Norm engine.  arg[7:0] selects the constants (mult, shift, eps and the
// gain's requantizer), arg[8] int16 input, arg[9] the gain is silu of the
// int8 vector at byte address arg[31:16]; len is the beat count.
// ---------------------------------------------------------------------------
module fabric_norm_adapter #(
    parameter int NL     = 8,
    parameter int DMAX   = 4096,
    parameter int SW     = 44,
    parameter int NCONST = 4,
    parameter int AW     = 16,
    parameter     LUT_DIR = "./"
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          cmd_valid,
    input  wire [15:0]   cmd_len,
    input  wire [15:0]   cmd_src,
    input  wire [15:0]   cmd_dst,
    input  wire [31:0]   cmd_arg,
    input  wire [7:0]    cmd_tag,
    output wire          cmd_ready,
    output reg           done_valid,
    output reg  [7:0]    done_tag,
    output wire [AW-1:0] rd_addr_x,
    output wire [AW-1:0] rd_addr_g,
    input  wire [127:0]  rd_data_x,
    input  wire [127:0]  rd_data_g,
    output reg           wr_en,
    output reg  [AW-1:0] wr_addr,
    output reg  [127:0]  wr_data,
    output reg  [15:0]   wr_be
);
    localparam int BW = $clog2(DMAX / NL) + 1;
    reg [44+SW-1:0] consts [0:NCONST-1];
    initial $readmemh("norm_consts.hex", consts);

    reg          busy, issuing, wrote;
    reg [BW-1:0] n, i, o;
    reg [AW-1:0] src, dst, gaddr;
    reg          int16, gated;
    reg [7:0]    tag;
    reg [44+SW-1:0] k;
    assign cmd_ready = !busy;
    assign rd_addr_x = src + (int16 ? i * 2 * NL : i * NL);
    assign rd_addr_g = gaddr + i * NL;

    // Front pipeline: address (1), data (2), gain requantized (2), silu (3..6); x waits alongside.
    reg             v1, v2, v3, v4, v5, v6;
    reg [NL*16-1:0] x1, x2, x3, x4, x5, gq2;
    wire [NL*16-1:0] silu_y;
    wire [NL-1:0]    silu_v;
    integer l;
    reg signed [63:0] gq;
    always @(posedge clk) begin
        v1 <= issuing; v2 <= v1; v3 <= v2; v4 <= v3; v5 <= v4; v6 <= v5;
        for (l = 0; l < NL; l = l + 1) begin
            x1[l*16 +: 16] <= int16 ? rd_data_x[l*16 +: 16] : {{8{rd_data_x[l*8+7]}}, rd_data_x[l*8 +: 8]};
            gq = fx_requant($signed({{56{rd_data_g[l*8+7]}}, rd_data_g[l*8 +: 8]}), k[37:22], k[43:38], 16);
            gq2[l*16 +: 16] <= gq[15:0];
        end
        x2 <= x1; x3 <= x2; x4 <= x3; x5 <= x4;
    end
    genvar g;
    generate
        for (g = 0; g < NL; g = g + 1) begin : g_silu
            fabric_silu #(.LUT_DIR(LUT_DIR)) u_silu (.clk(clk), .valid_in(v2), .t(gq2[g*16 +: 16]), .valid_out(silu_v[g]), .y(silu_y[g*16 +: 16]));
        end
    endgenerate
    wire [NL*16-1:0] gain6 = gated ? silu_y : {NL{16'd1}};

    wire            out_valid;
    wire [NL*8-1:0] out_y;
    fabric_rmsnorm #(.D(DMAX), .XW(16), .OW(8), .L(NL), .SW(SW), .LUT_DIR(LUT_DIR)) u_norm (
        .clk(clk), .rst_n(rst_n), .in_valid(v6), .n_beats(n), .in_x(x5), .in_gain(gain6), .mult(k[15:0]), .shift(k[21:16]),
        .eps(k[44 +: SW]), .out_valid(out_valid), .out_y(out_y));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0; issuing <= 1'b0; wrote <= 1'b0; done_valid <= 1'b0; wr_en <= 1'b0; i <= 0; o <= 0; n <= 0;
        end else begin
            done_valid <= wrote; wrote <= 1'b0; wr_en <= 1'b0;
            if (cmd_valid && cmd_ready) begin
                busy <= 1'b1; issuing <= 1'b1; n <= cmd_len[BW-1:0]; src <= cmd_src; dst <= cmd_dst;
                gaddr <= cmd_arg[31:16]; int16 <= cmd_arg[8]; gated <= cmd_arg[9]; k <= consts[cmd_arg[7:0]];
                tag <= cmd_tag; i <= 0; o <= 0;
            end
            if (issuing) begin
                i <= i + 1'b1;
                if (i == n - 1) issuing <= 1'b0;
            end
            if (out_valid) begin
                wr_en <= 1'b1; wr_addr <= dst + o * NL; wr_data <= {{(128-NL*8){1'b0}}, out_y}; wr_be <= (16'd1 << NL) - 1'b1;
                o <= o + 1'b1;
                if (o == n - 1) begin wrote <= 1'b1; done_tag <= tag; busy <= 1'b0; end
            end
        end
    end
endmodule

// ---------------------------------------------------------------------------
// The tile array as one pass engine.  passes.hex describes each tile:
//   [3:0] pass  [7:4] row block  [15:8] the tile whose partial sums it
//   continues (FF: none)  [16] last row block  [17] raw: write the
//   accumulators as 32-bit words  [25:18] bytes to write  [41:26] byte
//   offset of its output inside the destination.
// A command runs pass arg[7:0] over arg[15:8] row blocks: each row block's
// tiles start together and consume ROWS activations from src + rb * ROWS,
// P per cycle; when the last block's requantizer walk ends every last-block
// tile's output goes to dst + offset.
// ---------------------------------------------------------------------------
module fabric_pass_adapter #(
    parameter int NT   = 4,
    parameter int ROWS = 96,
    parameter int COLS = 16,
    parameter int WB   = 4,
    parameter int AB   = 8,
    parameter int P    = 2,
    parameter int ACC  = 24,
    parameter int SB   = 16,
    parameter int SHB  = 5,
    parameter int AW   = 16
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          cmd_valid,
    input  wire [15:0]   cmd_len,
    input  wire [15:0]   cmd_src,
    input  wire [15:0]   cmd_dst,
    input  wire [31:0]   cmd_arg,
    input  wire [7:0]    cmd_tag,
    output wire          cmd_ready,
    output reg           done_valid,
    output reg  [7:0]    done_tag,
    output wire [AW-1:0] rd_addr,
    input  wire [127:0]  rd_data,
    output reg           wr_en,
    output reg  [AW-1:0] wr_addr,
    output reg  [127:0]  wr_data,
    output reg  [15:0]   wr_be
);
    localparam int CYC = ROWS / P;
    localparam int PW  = COLS * ACC;
    localparam int QW  = COLS * AB;
    localparam int VW  = COLS * 32;          // the output vector as raw words
    reg [47:0]   tab [0:NT-1];
    reg [SB-1:0] mult_all [0:NT*COLS-1];
    reg [SHB-1:0] shift_all [0:NT*COLS-1];
    initial begin
        $readmemh("passes.hex", tab);
        $readmemh("tiles_mult.hex", mult_all);
        $readmemh("tiles_shift.hex", shift_all);
    end

    localparam [2:0] S_IDLE = 0, S_START = 1, S_STREAM = 2, S_WAIT = 3, S_WRITE = 4, S_DONE = 5;
    reg [2:0]    state;
    reg [7:0]    pass, nrb, rb, t, tag;
    reg [AW-1:0] src, dst;
    reg [15:0]   i;
    reg [3:0]    j;
    reg [NT-1:0] sel;
    reg          xv;
    assign cmd_ready = (state == S_IDLE);
    assign rd_addr   = src + rb * ROWS + i * P;

    wire [NT-1:0]    q_valid_t, start_t;
    wire [NT*PW-1:0] psum_out_flat;
    wire [NT*QW-1:0] q_out_flat;
    genvar gt, gc;
    generate
        for (gt = 0; gt < NT; gt = gt + 1) begin : g_tile
            wire [COLS*SB-1:0]  mult_w;
            wire [COLS*SHB-1:0] shift_w;
            for (gc = 0; gc < COLS; gc = gc + 1) begin : g_k
                assign mult_w[gc*SB +: SB]   = mult_all[gt*COLS + gc];
                assign shift_w[gc*SHB +: SHB] = shift_all[gt*COLS + gc];
            end
            wire [7:0]    chain   = tab[gt][15:8];
            wire [PW-1:0] psum_in = (chain == 8'hFF) ? {PW{1'b0}} : psum_out_flat[chain*PW +: PW];
            assign start_t[gt] = (state == S_START) && (tab[gt][3:0] == pass[3:0]) && (tab[gt][7:4] == rb[3:0]);
            fabric_tile #(.ROWS(ROWS), .COLS(COLS), .WB(WB), .AB(AB), .P(P), .ACC(ACC), .SB(SB), .SHB(SHB), .ROM_FILE("")) u_tile (
                .clk(clk), .rst_n(rst_n), .start(start_t[gt]), .psum_in(psum_in), .x_valid(xv), .x_data(rd_data[P*AB-1:0]),
                .mult(mult_w), .shift(shift_w), .x_ready(), .done(), .psum_out(psum_out_flat[gt*PW +: PW]),
                .q_out(q_out_flat[gt*QW +: QW]), .q_valid(q_valid_t[gt]));
            initial $readmemh($sformatf("tile_%0d.hex", gt), u_tile.rom.rom);
        end
    endgenerate

    // The output vector of tile t: its requantized bytes, or its accumulators as words.
    wire [47:0]   cur     = tab[t];
    wire          cur_hit = (cur[3:0] == pass[3:0]) && cur[16];
    wire [7:0]    nbytes  = cur[25:18];
    wire [4:0]    beats   = (nbytes + 15) / 16;
    reg  [VW-1:0] vec;
    integer c;
    always @* begin
        vec = {VW{1'b0}};
        if (cur[17]) begin
            for (c = 0; c < COLS; c = c + 1)
                vec[c*32 +: 32] = {{(32-ACC){psum_out_flat[t*PW + c*ACC + ACC - 1]}}, psum_out_flat[t*PW + c*ACC +: ACC]};
        end else vec[QW-1:0] = q_out_flat[t*QW +: QW];
    end
    wire [8:0]  remaining = nbytes - j * 16;
    wire [15:0] be_w      = (remaining >= 16) ? 16'hFFFF : ((16'd1 << remaining[3:0]) - 1'b1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; done_valid <= 1'b0; wr_en <= 1'b0; xv <= 1'b0; sel <= 0; i <= 0; j <= 0; t <= 0; rb <= 0;
        end else begin
            done_valid <= 1'b0; wr_en <= 1'b0; xv <= 1'b0;
            case (state)
                S_IDLE: if (cmd_valid) begin
                    pass <= cmd_arg[7:0]; nrb <= cmd_arg[15:8]; src <= cmd_src; dst <= cmd_dst; tag <= cmd_tag;
                    rb <= 0; state <= S_START;
                end
                S_START: begin
                    sel <= start_t; i <= 0; state <= S_STREAM;
                end
                S_STREAM: begin
                    xv <= 1'b1; i <= i + 1'b1;
                    if (i == CYC - 1) state <= S_WAIT;
                end
                S_WAIT: if (|(q_valid_t & sel)) begin
                    if (rb == nrb - 1) begin t <= 0; j <= 0; state <= S_WRITE; end
                    else begin rb <= rb + 1'b1; state <= S_START; end
                end
                S_WRITE: begin
                    if (cur_hit) begin
                        wr_en <= 1'b1; wr_addr <= dst + cur[41:26] + j * 16; wr_data <= vec[j*128 +: 128]; wr_be <= be_w;
                        if (j == beats - 1) begin j <= 0; t <= t + 1'b1; if (t == NT - 1) state <= S_DONE; end
                        else j <= j + 1'b1;
                    end else begin
                        t <= t + 1'b1;
                        if (t == NT - 1) state <= S_DONE;
                    end
                end
                default: begin done_valid <= 1'b1; done_tag <= tag; state <= S_IDLE; end
            endcase
        end
    end
endmodule

// ---------------------------------------------------------------------------
// The causal convolution.  src: the new int8 samples; arg[15:0]: the
// history (4 bytes per channel, oldest first); dst: the int8 outputs;
// arg2[15:0]: the shifted history; len: beats of CL channels.
// ---------------------------------------------------------------------------
module fabric_conv_adapter #(
    parameter int CL = 4,
    parameter int KK = 4,
    parameter int C  = 128,
    parameter int AW = 16,
    parameter     LUT_DIR = "./"
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          cmd_valid,
    input  wire [15:0]   cmd_len,
    input  wire [15:0]   cmd_src,
    input  wire [15:0]   cmd_dst,
    input  wire [31:0]   cmd_arg,
    input  wire [31:0]   cmd_arg2,
    input  wire [7:0]    cmd_tag,
    output wire          cmd_ready,
    output reg           done_valid,
    output reg  [7:0]    done_tag,
    output wire [AW-1:0] rd_addr_x,
    output wire [AW-1:0] rd_addr_h,
    input  wire [127:0]  rd_data_x,
    input  wire [127:0]  rd_data_h,
    output reg           wr_en_y,
    output reg  [AW-1:0] wr_addr_y,
    output reg  [127:0]  wr_data_y,
    output reg  [15:0]   wr_be_y,
    output reg           wr_en_h,
    output reg  [AW-1:0] wr_addr_h,
    output reg  [127:0]  wr_data_h,
    output reg  [15:0]   wr_be_h
);
    localparam int HW = (KK - 1) * 8;
    reg [KK*8-1:0] taps [0:C-1];
    reg [43:0]     consts [0:C-1];
    initial begin
        $readmemh("conv_taps.hex", taps);
        $readmemh("conv_consts.hex", consts);
    end
    reg          busy, issuing, wrote, v1;
    reg [15:0]   n, i, o;
    reg [AW-1:0] src, dst, hist, hnext;
    reg [7:0]    tag;
    reg [CL*KK*8-1:0] w1;
    reg [CL*16-1:0]   mi1, mo1;
    reg [CL*6-1:0]    si1, so1;
    assign cmd_ready = !busy;
    assign rd_addr_x = src + i * CL;
    assign rd_addr_h = hist + i * 16;
    integer c;
    always @(posedge clk) begin
        v1 <= issuing;
        for (c = 0; c < CL; c = c + 1) begin
            w1[c*KK*8 +: KK*8] <= taps[i*CL + c];
            mi1[c*16 +: 16] <= consts[i*CL + c][15:0];  si1[c*6 +: 6] <= consts[i*CL + c][21:16];
            mo1[c*16 +: 16] <= consts[i*CL + c][37:22]; so1[c*6 +: 6] <= consts[i*CL + c][43:38];
        end
    end
    reg [CL*HW-1:0] in_hist;
    always @* for (c = 0; c < CL; c = c + 1) in_hist[c*HW +: HW] = rd_data_h[c*32 +: HW];
    wire             out_valid;
    wire [CL*8-1:0]  out_y;
    wire [CL*HW-1:0] out_hist;
    fabric_conv_silu #(.K(KK), .L(CL), .LUT_DIR(LUT_DIR)) u_conv (
        .clk(clk), .in_valid(v1), .in_x(rd_data_x[CL*8-1:0]), .in_hist(in_hist), .in_w(w1), .mult_in(mi1), .sh_in(si1),
        .mult_out(mo1), .sh_out(so1), .out_valid(out_valid), .out_y(out_y), .out_hist(out_hist));
    reg [127:0] hist_beat;
    always @* begin
        hist_beat = 128'd0;
        for (c = 0; c < CL; c = c + 1) hist_beat[c*32 +: HW] = out_hist[c*HW +: HW];
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0; issuing <= 1'b0; wrote <= 1'b0; done_valid <= 1'b0; wr_en_y <= 1'b0; wr_en_h <= 1'b0; i <= 0; o <= 0; n <= 0;
        end else begin
            done_valid <= wrote; wrote <= 1'b0; wr_en_y <= 1'b0; wr_en_h <= 1'b0;
            if (cmd_valid && cmd_ready) begin
                busy <= 1'b1; issuing <= 1'b1; n <= cmd_len; src <= cmd_src; dst <= cmd_dst; hist <= cmd_arg[15:0];
                hnext <= cmd_arg2[15:0]; tag <= cmd_tag; i <= 0; o <= 0;
            end
            if (issuing) begin
                i <= i + 1'b1;
                if (i == n - 1) issuing <= 1'b0;
            end
            if (out_valid) begin
                wr_en_y <= 1'b1; wr_addr_y <= dst + o * CL; wr_data_y <= {{(128-CL*8){1'b0}}, out_y}; wr_be_y <= (16'd1 << CL) - 1'b1;
                wr_en_h <= 1'b1; wr_addr_h <= hnext + o * 16; wr_data_h <= hist_beat; wr_be_h <= 16'hFFFF;
                o <= o + 1'b1;
                if (o == n - 1) begin wrote <= 1'b1; done_tag <= tag; busy <= 1'b0; end
            end
        end
    end
endmodule

// ---------------------------------------------------------------------------
// The head gates.  src: the b accumulators as 32-bit words, arg[15:0]: the
// a accumulators, dst: 4 bytes per head (decay, beta), len: heads.
// ---------------------------------------------------------------------------
module fabric_gates_adapter #(
    parameter int NVMAX = 64,
    parameter int ACC   = 24,
    parameter int AW    = 16,
    parameter     LUT_DIR = "./"
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          cmd_valid,
    input  wire [15:0]   cmd_len,
    input  wire [15:0]   cmd_src,
    input  wire [15:0]   cmd_dst,
    input  wire [31:0]   cmd_arg,
    input  wire [7:0]    cmd_tag,
    output wire          cmd_ready,
    output reg           done_valid,
    output reg  [7:0]    done_tag,
    output wire [AW-1:0] rd_addr_b,
    output wire [AW-1:0] rd_addr_a,
    input  wire [127:0]  rd_data_b,
    input  wire [127:0]  rd_data_a,
    output reg           wr_en,
    output reg  [AW-1:0] wr_addr,
    output reg  [127:0]  wr_data,
    output reg  [15:0]   wr_be
);
    reg [75:0] consts [0:NVMAX-1];
    initial $readmemh("gates_consts.hex", consts);
    reg          busy, issuing, wrote, v1;
    reg [15:0]   n, i, o;
    reg [AW-1:0] src, dst, a_addr;
    reg [7:0]    tag;
    reg [75:0]   k1;
    assign cmd_ready = !busy;
    assign rd_addr_b = src + i * 4;
    assign rd_addr_a = a_addr + i * 4;
    always @(posedge clk) begin v1 <= issuing; k1 <= consts[i]; end
    wire        out_valid;
    wire [15:0] decay, beta;
    fabric_head_gates #(.ACC(ACC), .LUT_DIR(LUT_DIR)) u_gates (
        .clk(clk), .in_valid(v1), .a_acc(rd_data_a[ACC-1:0]), .b_acc(rd_data_b[ACC-1:0]), .mult_a(k1[15:0]), .sh_a(k1[21:16]),
        .mult_b(k1[37:22]), .sh_b(k1[43:38]), .a_coef(k1[59:44]), .dt_bias(k1[75:60]), .out_valid(out_valid), .decay(decay), .beta(beta));
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0; issuing <= 1'b0; wrote <= 1'b0; done_valid <= 1'b0; wr_en <= 1'b0; i <= 0; o <= 0; n <= 0;
        end else begin
            done_valid <= wrote; wrote <= 1'b0; wr_en <= 1'b0;
            if (cmd_valid && cmd_ready) begin
                busy <= 1'b1; issuing <= 1'b1; n <= cmd_len; src <= cmd_src; dst <= cmd_dst; a_addr <= cmd_arg[15:0];
                tag <= cmd_tag; i <= 0; o <= 0;
            end
            if (issuing) begin
                i <= i + 1'b1;
                if (i == n - 1) issuing <= 1'b0;
            end
            if (out_valid) begin
                wr_en <= 1'b1; wr_addr <= dst + o * 4; wr_data <= {96'd0, beta, decay}; wr_be <= 16'h000F;
                o <= o + 1'b1;
                if (o == n - 1) begin wrote <= 1'b1; done_tag <= tag; busy <= 1'b0; end
            end
        end
    end
endmodule

// ---------------------------------------------------------------------------
// One int8 state engine.  src: k (K bytes), arg[15:0]: q, arg[31:16]: v (V
// bytes), arg2[15:0]: the head's gates word, arg2[31:16]: the state slot
// (a header beat of g, e, peak, nsat then K rows of V bytes), dst: y (V
// int16).  The rows stream through fabric_delta_state8 and back into the
// slot; y and the new header are written last.
// ---------------------------------------------------------------------------
module fabric_delta_adapter #(
    parameter int K   = 16,
    parameter int V   = 16,
    parameter int YSH = 9,
    parameter int AW  = 16,
    parameter int E_MIN = -4,
    parameter int E_MAX = 6,
    parameter int PEAK_GROW = 47,
    parameter int SAT_SHIFT = 6
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          cmd_valid,
    input  wire [15:0]   cmd_src,
    input  wire [15:0]   cmd_dst,
    input  wire [31:0]   cmd_arg,
    input  wire [31:0]   cmd_arg2,
    input  wire [7:0]    cmd_tag,
    output wire          cmd_ready,
    output reg           done_valid,
    output reg  [7:0]    done_tag,
    output reg  [AW-1:0] rd_addr,
    input  wire [127:0]  rd_data,
    output reg           wr_en,
    output reg  [AW-1:0] wr_addr,
    output reg  [127:0]  wr_data,
    output reg  [15:0]   wr_be
);
    localparam int KB = K / 16, VB = V / 16, NLD = 2 * KB + VB + 2;
    localparam int RW = $clog2(K * VB + 1);
    localparam [3:0] S_IDLE = 0, S_LOAD = 1, S_START = 2, S_ROWS = 3, S_RUN = 4, S_Y = 5, S_HDR = 6, S_DONE = 7;
    reg [3:0]    state;
    reg [AW-1:0] k_addr, q_addr, v_addr, g_addr, slot, y_addr;
    reg [7:0]    tag;
    reg [K*8-1:0] q_r, k_r;
    reg [V*8-1:0] v_r;
    reg [15:0]    decay, beta, g_in, nsat_in;
    reg [7:0]     e_in, peak_in;
    reg [7:0]     ld, ld_d;
    reg           ldv;
    reg [RW-1:0]  r, r_d, rows_written;
    reg           rv;
    reg [V*8-1:0] row_buf, row_in;
    reg           row_in_valid, start;
    reg [3:0]     j;
    assign cmd_ready = (state == S_IDLE);

    wire            row_out_valid, y_valid;
    wire [V*8-1:0]  row_out;
    wire [V*16-1:0] y;
    wire [15:0]     g_out, nsat_out;
    wire [7:0]      e_out, peak_out;
    fabric_delta_state8 #(.K(K), .V(V), .YSH(YSH), .E_MIN(E_MIN), .E_MAX(E_MAX), .PEAK_GROW(PEAK_GROW), .SAT_SHIFT(SAT_SHIFT)) u_delta (
        .clk(clk), .rst_n(rst_n), .start(start), .q(q_r), .k(k_r), .v(v_r), .decay(decay), .beta(beta), .g_in(g_in), .e_in(e_in),
        .peak_in(peak_in), .nsat_in(nsat_in), .g_out(g_out), .e_out(e_out), .peak_out(peak_out), .nsat_out(nsat_out),
        .row_in_valid(row_in_valid), .row_in(row_in), .row_out_valid(row_out_valid), .row_out(row_out), .y_valid(y_valid), .y(y));

    // Rows out are queued (they leave one per cycle, they are written VB beats each).
    reg [V*8-1:0] fifo [0:K-1];
    reg [RW-1:0]  fw, fr;
    reg [3:0]     fj;
    reg           y_seen;
    reg [V*16-1:0] y_r;
    wire [7:0]    ld_n = ld;
    wire [V*8-1:0] row_asm;                  // the row as its last beat lands
    generate
        if (VB == 1) begin : g_one assign row_asm = rd_data; end
        else begin : g_many assign row_asm = {rd_data, row_buf[(VB-1)*128-1:0]}; end
    endgenerate
    always @* begin
        if (ld_n < KB)               rd_addr = q_addr + ld_n * 16;
        else if (ld_n < 2 * KB)      rd_addr = k_addr + (ld_n - KB) * 16;
        else if (ld_n < 2 * KB + VB) rd_addr = v_addr + (ld_n - 2 * KB) * 16;
        else if (ld_n == 2 * KB + VB) rd_addr = g_addr;
        else                         rd_addr = slot;
        if (state == S_ROWS) rd_addr = slot + 16 + r * 16;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; done_valid <= 1'b0; wr_en <= 1'b0; ldv <= 1'b0; rv <= 1'b0; row_in_valid <= 1'b0; start <= 1'b0;
            ld <= 0; r <= 0; fw <= 0; fr <= 0; fj <= 0; y_seen <= 1'b0; rows_written <= 0; j <= 0;
        end else begin
            done_valid <= 1'b0; wr_en <= 1'b0; row_in_valid <= 1'b0; start <= 1'b0;
            ldv <= (state == S_LOAD); ld_d <= ld;
            rv <= (state == S_ROWS); r_d <= r;
            // Operand loads land one cycle after their issue.
            if (ldv) begin
                if (ld_d < KB)                q_r[ld_d*128 +: 128] <= rd_data;
                else if (ld_d < 2 * KB)       k_r[(ld_d - KB)*128 +: 128] <= rd_data;
                else if (ld_d < 2 * KB + VB)  v_r[(ld_d - 2 * KB)*128 +: 128] <= rd_data;
                else if (ld_d == 2 * KB + VB) begin decay <= rd_data[15:0]; beta <= rd_data[31:16]; end
                else begin g_in <= rd_data[15:0]; e_in <= rd_data[23:16]; peak_in <= rd_data[31:24]; nsat_in <= rd_data[47:32]; end
            end
            // Row beats assemble into rows.
            if (rv) begin
                if (VB == 1 || r_d % VB == VB - 1) begin
                    row_in_valid <= 1'b1;
                    row_in <= row_asm;
                end else row_buf[(r_d % VB)*128 +: 128] <= rd_data;
            end
            if (row_out_valid) begin fifo[fw] <= row_out; fw <= fw + 1'b1; end
            if (y_valid) begin y_seen <= 1'b1; y_r <= y; end
            case (state)
                S_IDLE: if (cmd_valid) begin
                    k_addr <= cmd_src; y_addr <= cmd_dst; q_addr <= cmd_arg[15:0]; v_addr <= cmd_arg[31:16];
                    g_addr <= cmd_arg2[15:0]; slot <= cmd_arg2[31:16]; tag <= cmd_tag;
                    ld <= 0; r <= 0; fw <= 0; fr <= 0; fj <= 0; y_seen <= 1'b0; rows_written <= 0; j <= 0;
                    state <= S_LOAD;
                end
                S_LOAD: begin
                    ld <= ld + 1'b1;
                    if (ld == NLD - 1) state <= S_START;
                end
                S_START: if (ldv && ld_d == NLD - 1) begin   // the header lands this edge; start next cycle
                    start <= 1'b1; state <= S_ROWS;
                end
                S_ROWS: begin
                    r <= r + 1'b1;
                    if (r == K * VB - 1) state <= S_RUN;
                end
                S_RUN: begin
                    if (fr != fw) begin
                        wr_en <= 1'b1; wr_addr <= slot + 16 + (fr * VB + fj) * 16; wr_data <= fifo[fr][fj*128 +: 128]; wr_be <= 16'hFFFF;
                        if (fj == VB - 1) begin fj <= 0; fr <= fr + 1'b1; rows_written <= rows_written + 1'b1; end
                        else fj <= fj + 1'b1;
                    end else if (rows_written == K && y_seen) begin
                        j <= 0; state <= S_Y;
                    end
                end
                S_Y: begin
                    wr_en <= 1'b1; wr_addr <= y_addr + j * 16; wr_data <= y_r[j*128 +: 128]; wr_be <= 16'hFFFF;
                    j <= j + 1'b1;
                    if (j == 2 * VB - 1) state <= S_HDR;
                end
                S_HDR: begin
                    wr_en <= 1'b1; wr_addr <= slot; wr_data <= {80'd0, nsat_out, peak_out, e_out, g_out}; wr_be <= 16'hFFFF;
                    state <= S_DONE;
                end
                default: begin done_valid <= 1'b1; done_tag <= tag; state <= S_IDLE; end
            endcase
        end
    end
endmodule

// ---------------------------------------------------------------------------
// SwiGLU.  src: gate (int8), arg[15:0]: up, arg[23:16]: constants, dst: act; len: beats of NL.
// ---------------------------------------------------------------------------
module fabric_swiglu_adapter #(
    parameter int NL = 8,
    parameter int AW = 16,
    parameter     LUT_DIR = "./"
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          cmd_valid,
    input  wire [15:0]   cmd_len,
    input  wire [15:0]   cmd_src,
    input  wire [15:0]   cmd_dst,
    input  wire [31:0]   cmd_arg,
    input  wire [7:0]    cmd_tag,
    output wire          cmd_ready,
    output reg           done_valid,
    output reg  [7:0]    done_tag,
    output wire [AW-1:0] rd_addr_g,
    output wire [AW-1:0] rd_addr_u,
    input  wire [127:0]  rd_data_g,
    input  wire [127:0]  rd_data_u,
    output reg           wr_en,
    output reg  [AW-1:0] wr_addr,
    output reg  [127:0]  wr_data,
    output reg  [15:0]   wr_be
);
    reg [43:0] consts [0:15];
    initial $readmemh("swiglu_consts.hex", consts);
    reg          busy, issuing, wrote, v1;
    reg [15:0]   n, i, o;
    reg [AW-1:0] src, dst, u_addr;
    reg [7:0]    tag;
    reg [43:0]   k;
    assign cmd_ready = !busy;
    assign rd_addr_g = src + i * NL;
    assign rd_addr_u = u_addr + i * NL;
    always @(posedge clk) v1 <= issuing;
    wire            out_valid;
    wire [NL*8-1:0] out_y;
    fabric_swiglu #(.L(NL), .LUT_DIR(LUT_DIR)) u_sw (
        .clk(clk), .in_valid(v1), .in_g(rd_data_g[NL*8-1:0]), .in_u(rd_data_u[NL*8-1:0]), .mult_g(k[15:0]), .sh_g(k[21:16]),
        .mult_o(k[37:22]), .sh_o(k[43:38]), .out_valid(out_valid), .out_y(out_y));
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0; issuing <= 1'b0; wrote <= 1'b0; done_valid <= 1'b0; wr_en <= 1'b0; i <= 0; o <= 0; n <= 0;
        end else begin
            done_valid <= wrote; wrote <= 1'b0; wr_en <= 1'b0;
            if (cmd_valid && cmd_ready) begin
                busy <= 1'b1; issuing <= 1'b1; n <= cmd_len; src <= cmd_src; dst <= cmd_dst; u_addr <= cmd_arg[15:0];
                k <= consts[cmd_arg[19:16]]; tag <= cmd_tag; i <= 0; o <= 0;
            end
            if (issuing) begin
                i <= i + 1'b1;
                if (i == n - 1) issuing <= 1'b0;
            end
            if (out_valid) begin
                wr_en <= 1'b1; wr_addr <= dst + o * NL; wr_data <= {{(128-NL*8){1'b0}}, out_y}; wr_be <= (16'd1 << NL) - 1'b1;
                o <= o + 1'b1;
                if (o == n - 1) begin wrote <= 1'b1; done_tag <= tag; busy <= 1'b0; end
            end
        end
    end
endmodule

// ---------------------------------------------------------------------------
// The residual add.  src: h (int16), arg[15:0]: y (int8), arg[23:16]: constants, dst: h'; len: beats of NL.
// ---------------------------------------------------------------------------
module fabric_residual_adapter #(
    parameter int NL = 8,
    parameter int AW = 16
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          cmd_valid,
    input  wire [15:0]   cmd_len,
    input  wire [15:0]   cmd_src,
    input  wire [15:0]   cmd_dst,
    input  wire [31:0]   cmd_arg,
    input  wire [7:0]    cmd_tag,
    output wire          cmd_ready,
    output reg           done_valid,
    output reg  [7:0]    done_tag,
    output wire [AW-1:0] rd_addr_h,
    output wire [AW-1:0] rd_addr_y,
    input  wire [127:0]  rd_data_h,
    input  wire [127:0]  rd_data_y,
    output reg           wr_en,
    output reg  [AW-1:0] wr_addr,
    output reg  [127:0]  wr_data,
    output reg  [15:0]   wr_be
);
    reg [21:0] consts [0:15];
    initial $readmemh("residual_consts.hex", consts);
    reg          busy, issuing, wrote, v1;
    reg [15:0]   n, i, o;
    reg [AW-1:0] src, dst, y_addr;
    reg [7:0]    tag;
    reg [21:0]   k;
    assign cmd_ready = !busy;
    assign rd_addr_h = src + i * 2 * NL;
    assign rd_addr_y = y_addr + i * NL;
    always @(posedge clk) v1 <= issuing;
    wire             out_valid;
    wire [NL*16-1:0] out_h;
    fabric_residual #(.L(NL)) u_res (
        .clk(clk), .in_valid(v1), .in_h(rd_data_h[NL*16-1:0]), .in_y(rd_data_y[NL*8-1:0]), .mult(k[15:0]), .shift(k[21:16]),
        .out_valid(out_valid), .out_h(out_h));
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0; issuing <= 1'b0; wrote <= 1'b0; done_valid <= 1'b0; wr_en <= 1'b0; i <= 0; o <= 0; n <= 0;
        end else begin
            done_valid <= wrote; wrote <= 1'b0; wr_en <= 1'b0;
            if (cmd_valid && cmd_ready) begin
                busy <= 1'b1; issuing <= 1'b1; n <= cmd_len; src <= cmd_src; dst <= cmd_dst; y_addr <= cmd_arg[15:0];
                k <= consts[cmd_arg[19:16]]; tag <= cmd_tag; i <= 0; o <= 0;
            end
            if (issuing) begin
                i <= i + 1'b1;
                if (i == n - 1) issuing <= 1'b0;
            end
            if (out_valid) begin
                wr_en <= 1'b1; wr_addr <= dst + o * 2 * NL; wr_data <= {{(128-NL*16){1'b0}}, out_h}; wr_be <= (16'd1 << (2 * NL)) - 1'b1;
                o <= o + 1'b1;
                if (o == n - 1) begin wrote <= 1'b1; done_tag <= tag; busy <= 1'b0; end
            end
        end
    end
endmodule

// ---------------------------------------------------------------------------
// The memory unit: len beats between the memory port (beat addresses) and
// the vector buffer.  arg[3:0] = 0: memory src to buffer dst; 1: buffer
// src to memory dst.
// ---------------------------------------------------------------------------
module fabric_mem_adapter #(
    parameter int AW = 16
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          cmd_valid,
    input  wire [15:0]   cmd_len,
    input  wire [15:0]   cmd_src,
    input  wire [15:0]   cmd_dst,
    input  wire [31:0]   cmd_arg,
    input  wire [7:0]    cmd_tag,
    output wire          cmd_ready,
    output reg           done_valid,
    output reg  [7:0]    done_tag,
    output wire [AW-1:0] rd_addr,
    input  wire [127:0]  rd_data,
    output reg           wr_en,
    output reg  [AW-1:0] wr_addr,
    output reg  [127:0]  wr_data,
    output reg  [15:0]   wr_be,
    output wire [31:0]   mem_rd_addr,
    input  wire [127:0]  mem_rd_data,
    output reg           mem_wr_en,
    output reg  [31:0]   mem_wr_addr,
    output reg  [127:0]  mem_wr_data
);
    reg          busy, issuing, wrote, v1, to_mem;
    reg [15:0]   n, i, o, src, dst;
    reg [7:0]    tag;
    assign cmd_ready   = !busy;
    assign rd_addr     = src + i * 16;
    assign mem_rd_addr = {16'd0, src} + i;
    always @(posedge clk) v1 <= issuing;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0; issuing <= 1'b0; wrote <= 1'b0; done_valid <= 1'b0; wr_en <= 1'b0; mem_wr_en <= 1'b0; i <= 0; o <= 0; n <= 0;
        end else begin
            done_valid <= wrote; wrote <= 1'b0; wr_en <= 1'b0; mem_wr_en <= 1'b0;
            if (cmd_valid && cmd_ready) begin
                busy <= 1'b1; issuing <= 1'b1; n <= cmd_len; src <= cmd_src; dst <= cmd_dst; to_mem <= cmd_arg[0];
                tag <= cmd_tag; i <= 0; o <= 0;
            end
            if (issuing) begin
                i <= i + 1'b1;
                if (i == n - 1) issuing <= 1'b0;
            end
            if (v1) begin
                if (to_mem) begin mem_wr_en <= 1'b1; mem_wr_addr <= {16'd0, dst} + o; mem_wr_data <= rd_data; end
                else begin wr_en <= 1'b1; wr_addr <= dst + o * 16; wr_data <= mem_rd_data; wr_be <= 16'hFFFF; end
                o <= o + 1'b1;
                if (o == n - 1) begin wrote <= 1'b1; done_tag <= tag; busy <= 1'b0; end
            end
        end
    end
endmodule

// ---------------------------------------------------------------------------
// The engine.
// ---------------------------------------------------------------------------
module fabric_layer_engine #(
    parameter int D    = 96,
    parameter int NK   = 2,
    parameter int NV   = 4,
    parameter int HK   = 16,
    parameter int HV   = 16,
    parameter int KK   = 4,
    parameter int CONV = 128,
    parameter int ROWS = 96,
    parameter int COLS = 16,
    parameter int P    = 2,
    parameter int NT   = 56,
    parameter int WB   = 4,
    parameter int ACC  = 24,
    parameter int SB   = 16,
    parameter int SHB  = 5,
    parameter int SW   = 38,
    parameter int YSH  = 9,
    parameter int VB_BYTES = 4096,
    parameter     VB_FILE   = "vb_init.hex",
    parameter     PROG_FILE = "program.hex",
    parameter     LUT_DIR   = "./"
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire [15:0]  n_steps,
    output wire         running,
    output wire         done,
    output wire [31:0]  mem_rd_addr,
    input  wire [127:0] mem_rd_data,
    output wire         mem_wr_en,
    output wire [31:0]  mem_wr_addr,
    output wire [127:0] mem_wr_data
);
    localparam int NU = 10, NE = 4, AW = 16, NL = 8, CL = 4;
    localparam int U_TILES = 0, U_NORM = 1, U_CONV = 2, U_GATES = 3, U_DELTA = 4, U_SWIGLU = 5, U_RESIDUAL = 6, U_ROTARY = 7, U_ATTN = 8, U_MEM = 9;
    // Vector-buffer ports.
    localparam int R_NORM = 0, R_TILES = 4, R_CONV = 5, R_GATES = 7, R_DELTA = 9, R_SWIGLU = 13, R_RESIDUAL = 15, R_MEM = 17, NR = 18;
    localparam int W_NORM = 0, W_TILES = 2, W_CONV = 3, W_GATES = 5, W_DELTA = 6, W_SWIGLU = 10, W_RESIDUAL = 11, W_MEM = 12, NW = 13;

    wire [NU-1:0]      cmd_valid, cmd_ready;
    wire [3:0]         cmd_engine;
    wire [15:0]        cmd_len, cmd_src, cmd_dst;
    wire [31:0]        cmd_arg, cmd_arg2;
    wire [7:0]         cmd_tag;
    wire [NU*NE-1:0]   done_valid;
    wire [NU*NE*8-1:0] done_tag;
    fabric_sequencer #(.NU(NU), .NE(NE), .PROG_FILE(PROG_FILE)) u_seq (
        .clk(clk), .rst_n(rst_n), .start(start), .n_steps(n_steps), .running(running), .done(done),
        .cmd_valid(cmd_valid), .cmd_engine(cmd_engine), .cmd_len(cmd_len), .cmd_src(cmd_src), .cmd_dst(cmd_dst),
        .cmd_arg(cmd_arg), .cmd_arg2(cmd_arg2), .cmd_tag(cmd_tag), .cmd_ready(cmd_ready), .done_valid(done_valid), .done_tag(done_tag));

    wire [NR*AW-1:0]  rd_addr;
    wire [NR*128-1:0] rd_data;
    wire [NW-1:0]     wr_en;
    wire [NW*AW-1:0]  wr_addr;
    wire [NW*128-1:0] wr_data;
    wire [NW*16-1:0]  wr_be;
    fabric_vb #(.BYTES(VB_BYTES), .NR(NR), .NW(NW), .AW(AW), .INIT_FILE(VB_FILE)) u_vb (
        .clk(clk), .rd_addr(rd_addr), .rd_data(rd_data), .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data), .wr_be(wr_be));

    // Units without engines here: the rotary and attention cores of the global layer.
    wire [NE-1:0] ready_norm, ready_delta;
    assign cmd_ready[U_TILES]    = (cmd_engine == 0) && ready_tiles;
    assign cmd_ready[U_NORM]     = (cmd_engine < 2) && ready_norm[cmd_engine];
    assign cmd_ready[U_CONV]     = (cmd_engine == 0) && ready_conv;
    assign cmd_ready[U_GATES]    = (cmd_engine == 0) && ready_gates;
    assign cmd_ready[U_DELTA]    = ready_delta[cmd_engine];
    assign cmd_ready[U_SWIGLU]   = (cmd_engine == 0) && ready_swiglu;
    assign cmd_ready[U_RESIDUAL] = (cmd_engine == 0) && ready_residual;
    assign cmd_ready[U_ROTARY]   = 1'b0;
    assign cmd_ready[U_ATTN]     = 1'b0;
    assign cmd_ready[U_MEM]      = (cmd_engine == 0) && ready_mem;
    assign done_valid[U_ROTARY*NE +: NE] = 0;
    assign done_valid[U_ATTN*NE +: NE]   = 0;
    assign done_tag[U_ROTARY*NE*8 +: NE*8] = 0;
    assign done_tag[U_ATTN*NE*8 +: NE*8]   = 0;
    assign done_valid[U_TILES*NE + 1 +: 3] = 0;    assign done_tag[(U_TILES*NE + 1)*8 +: 24] = 0;
    assign done_valid[U_NORM*NE + 2 +: 2] = 0;     assign done_tag[(U_NORM*NE + 2)*8 +: 16] = 0;
    assign done_valid[U_CONV*NE + 1 +: 3] = 0;     assign done_tag[(U_CONV*NE + 1)*8 +: 24] = 0;
    assign done_valid[U_GATES*NE + 1 +: 3] = 0;    assign done_tag[(U_GATES*NE + 1)*8 +: 24] = 0;
    assign done_valid[U_SWIGLU*NE + 1 +: 3] = 0;   assign done_tag[(U_SWIGLU*NE + 1)*8 +: 24] = 0;
    assign done_valid[U_RESIDUAL*NE + 1 +: 3] = 0; assign done_tag[(U_RESIDUAL*NE + 1)*8 +: 24] = 0;
    assign done_valid[U_MEM*NE + 1 +: 3] = 0;      assign done_tag[(U_MEM*NE + 1)*8 +: 24] = 0;
    assign ready_norm[3:2] = 0;

    wire ready_tiles, ready_conv, ready_gates, ready_swiglu, ready_residual, ready_mem;
    genvar e;
    generate
        for (e = 0; e < 2; e = e + 1) begin : g_norm
            fabric_norm_adapter #(.NL(NL), .DMAX(D), .SW(SW), .NCONST(4), .AW(AW), .LUT_DIR(LUT_DIR)) u (
                .clk(clk), .rst_n(rst_n), .cmd_valid(cmd_valid[U_NORM] && cmd_engine == e), .cmd_len(cmd_len), .cmd_src(cmd_src),
                .cmd_dst(cmd_dst), .cmd_arg(cmd_arg), .cmd_tag(cmd_tag), .cmd_ready(ready_norm[e]),
                .done_valid(done_valid[U_NORM*NE + e]), .done_tag(done_tag[(U_NORM*NE + e)*8 +: 8]),
                .rd_addr_x(rd_addr[(R_NORM + 2*e)*AW +: AW]), .rd_addr_g(rd_addr[(R_NORM + 2*e + 1)*AW +: AW]),
                .rd_data_x(rd_data[(R_NORM + 2*e)*128 +: 128]), .rd_data_g(rd_data[(R_NORM + 2*e + 1)*128 +: 128]),
                .wr_en(wr_en[W_NORM + e]), .wr_addr(wr_addr[(W_NORM + e)*AW +: AW]), .wr_data(wr_data[(W_NORM + e)*128 +: 128]),
                .wr_be(wr_be[(W_NORM + e)*16 +: 16]));
        end
        for (e = 0; e < NE; e = e + 1) begin : g_delta
            fabric_delta_adapter #(.K(HK), .V(HV), .YSH(YSH), .AW(AW)) u (
                .clk(clk), .rst_n(rst_n), .cmd_valid(cmd_valid[U_DELTA] && cmd_engine == e), .cmd_src(cmd_src), .cmd_dst(cmd_dst),
                .cmd_arg(cmd_arg), .cmd_arg2(cmd_arg2), .cmd_tag(cmd_tag), .cmd_ready(ready_delta[e]),
                .done_valid(done_valid[U_DELTA*NE + e]), .done_tag(done_tag[(U_DELTA*NE + e)*8 +: 8]),
                .rd_addr(rd_addr[(R_DELTA + e)*AW +: AW]), .rd_data(rd_data[(R_DELTA + e)*128 +: 128]),
                .wr_en(wr_en[W_DELTA + e]), .wr_addr(wr_addr[(W_DELTA + e)*AW +: AW]), .wr_data(wr_data[(W_DELTA + e)*128 +: 128]),
                .wr_be(wr_be[(W_DELTA + e)*16 +: 16]));
        end
    endgenerate

    fabric_pass_adapter #(.NT(NT), .ROWS(ROWS), .COLS(COLS), .WB(WB), .AB(8), .P(P), .ACC(ACC), .SB(SB), .SHB(SHB), .AW(AW)) u_tiles (
        .clk(clk), .rst_n(rst_n), .cmd_valid(cmd_valid[U_TILES] && cmd_engine == 0), .cmd_len(cmd_len), .cmd_src(cmd_src), .cmd_dst(cmd_dst),
        .cmd_arg(cmd_arg), .cmd_tag(cmd_tag), .cmd_ready(ready_tiles), .done_valid(done_valid[U_TILES*NE]), .done_tag(done_tag[U_TILES*NE*8 +: 8]),
        .rd_addr(rd_addr[R_TILES*AW +: AW]), .rd_data(rd_data[R_TILES*128 +: 128]),
        .wr_en(wr_en[W_TILES]), .wr_addr(wr_addr[W_TILES*AW +: AW]), .wr_data(wr_data[W_TILES*128 +: 128]), .wr_be(wr_be[W_TILES*16 +: 16]));

    fabric_conv_adapter #(.CL(CL), .KK(KK), .C(CONV), .AW(AW), .LUT_DIR(LUT_DIR)) u_conv (
        .clk(clk), .rst_n(rst_n), .cmd_valid(cmd_valid[U_CONV] && cmd_engine == 0), .cmd_len(cmd_len), .cmd_src(cmd_src), .cmd_dst(cmd_dst),
        .cmd_arg(cmd_arg), .cmd_arg2(cmd_arg2), .cmd_tag(cmd_tag), .cmd_ready(ready_conv), .done_valid(done_valid[U_CONV*NE]),
        .done_tag(done_tag[U_CONV*NE*8 +: 8]),
        .rd_addr_x(rd_addr[R_CONV*AW +: AW]), .rd_addr_h(rd_addr[(R_CONV+1)*AW +: AW]),
        .rd_data_x(rd_data[R_CONV*128 +: 128]), .rd_data_h(rd_data[(R_CONV+1)*128 +: 128]),
        .wr_en_y(wr_en[W_CONV]), .wr_addr_y(wr_addr[W_CONV*AW +: AW]), .wr_data_y(wr_data[W_CONV*128 +: 128]), .wr_be_y(wr_be[W_CONV*16 +: 16]),
        .wr_en_h(wr_en[W_CONV+1]), .wr_addr_h(wr_addr[(W_CONV+1)*AW +: AW]), .wr_data_h(wr_data[(W_CONV+1)*128 +: 128]), .wr_be_h(wr_be[(W_CONV+1)*16 +: 16]));

    fabric_gates_adapter #(.NVMAX(NV), .ACC(ACC), .AW(AW), .LUT_DIR(LUT_DIR)) u_gates (
        .clk(clk), .rst_n(rst_n), .cmd_valid(cmd_valid[U_GATES] && cmd_engine == 0), .cmd_len(cmd_len), .cmd_src(cmd_src), .cmd_dst(cmd_dst),
        .cmd_arg(cmd_arg), .cmd_tag(cmd_tag), .cmd_ready(ready_gates), .done_valid(done_valid[U_GATES*NE]), .done_tag(done_tag[U_GATES*NE*8 +: 8]),
        .rd_addr_b(rd_addr[R_GATES*AW +: AW]), .rd_addr_a(rd_addr[(R_GATES+1)*AW +: AW]),
        .rd_data_b(rd_data[R_GATES*128 +: 128]), .rd_data_a(rd_data[(R_GATES+1)*128 +: 128]),
        .wr_en(wr_en[W_GATES]), .wr_addr(wr_addr[W_GATES*AW +: AW]), .wr_data(wr_data[W_GATES*128 +: 128]), .wr_be(wr_be[W_GATES*16 +: 16]));

    fabric_swiglu_adapter #(.NL(NL), .AW(AW), .LUT_DIR(LUT_DIR)) u_swiglu (
        .clk(clk), .rst_n(rst_n), .cmd_valid(cmd_valid[U_SWIGLU] && cmd_engine == 0), .cmd_len(cmd_len), .cmd_src(cmd_src), .cmd_dst(cmd_dst),
        .cmd_arg(cmd_arg), .cmd_tag(cmd_tag), .cmd_ready(ready_swiglu), .done_valid(done_valid[U_SWIGLU*NE]), .done_tag(done_tag[U_SWIGLU*NE*8 +: 8]),
        .rd_addr_g(rd_addr[R_SWIGLU*AW +: AW]), .rd_addr_u(rd_addr[(R_SWIGLU+1)*AW +: AW]),
        .rd_data_g(rd_data[R_SWIGLU*128 +: 128]), .rd_data_u(rd_data[(R_SWIGLU+1)*128 +: 128]),
        .wr_en(wr_en[W_SWIGLU]), .wr_addr(wr_addr[W_SWIGLU*AW +: AW]), .wr_data(wr_data[W_SWIGLU*128 +: 128]), .wr_be(wr_be[W_SWIGLU*16 +: 16]));

    fabric_residual_adapter #(.NL(NL), .AW(AW)) u_residual (
        .clk(clk), .rst_n(rst_n), .cmd_valid(cmd_valid[U_RESIDUAL] && cmd_engine == 0), .cmd_len(cmd_len), .cmd_src(cmd_src), .cmd_dst(cmd_dst),
        .cmd_arg(cmd_arg), .cmd_tag(cmd_tag), .cmd_ready(ready_residual), .done_valid(done_valid[U_RESIDUAL*NE]), .done_tag(done_tag[U_RESIDUAL*NE*8 +: 8]),
        .rd_addr_h(rd_addr[R_RESIDUAL*AW +: AW]), .rd_addr_y(rd_addr[(R_RESIDUAL+1)*AW +: AW]),
        .rd_data_h(rd_data[R_RESIDUAL*128 +: 128]), .rd_data_y(rd_data[(R_RESIDUAL+1)*128 +: 128]),
        .wr_en(wr_en[W_RESIDUAL]), .wr_addr(wr_addr[W_RESIDUAL*AW +: AW]), .wr_data(wr_data[W_RESIDUAL*128 +: 128]), .wr_be(wr_be[W_RESIDUAL*16 +: 16]));

    fabric_mem_adapter #(.AW(AW)) u_mem (
        .clk(clk), .rst_n(rst_n), .cmd_valid(cmd_valid[U_MEM] && cmd_engine == 0), .cmd_len(cmd_len), .cmd_src(cmd_src), .cmd_dst(cmd_dst),
        .cmd_arg(cmd_arg), .cmd_tag(cmd_tag), .cmd_ready(ready_mem), .done_valid(done_valid[U_MEM*NE]), .done_tag(done_tag[U_MEM*NE*8 +: 8]),
        .rd_addr(rd_addr[R_MEM*AW +: AW]), .rd_data(rd_data[R_MEM*128 +: 128]),
        .wr_en(wr_en[W_MEM]), .wr_addr(wr_addr[W_MEM*AW +: AW]), .wr_data(wr_data[W_MEM*128 +: 128]), .wr_be(wr_be[W_MEM*16 +: 16]),
        .mem_rd_addr(mem_rd_addr), .mem_rd_data(mem_rd_data), .mem_wr_en(mem_wr_en), .mem_wr_addr(mem_wr_addr), .mem_wr_data(mem_wr_data));
endmodule

`default_nettype wire
