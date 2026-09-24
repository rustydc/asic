// Self-checking harness for fabric_layer_engine: runs a program written by
// fabric.engine.EngineRun over the real units and dumps the vector buffer
// and the memory for the Python side to compare bit for bit.  The memory
// behind the port is fabric_memory.sv's behavioural model, or with USE_HPI
// the die's own path: fabric_mem_bridge across the clock crossing, the
// stripe unit, NDEV channel controllers and NDEV device models on the
// 250 MHz controller clock, their images loaded from and dumped to
// devs.hex / devs_out.hex through the stripe map.

`timescale 1ns/1ps
`default_nettype none

module tb_layer_engine #(
    parameter int N    = 33,
    parameter int D    = 96,
    parameter int NK   = 2,
    parameter int NV   = 4,
    parameter int HK   = 16,
    parameter int HV   = 16,
    parameter int KK   = 4,
    parameter int CONV = 128,
    parameter int FFN  = 192,
    parameter int NH   = 4,
    parameter int NKV  = 2,
    parameter int HD   = 24,
    parameter int RD   = 6,
    parameter int IDIM = 32,
    parameter int W    = 16,
    parameter int BS   = 4,
    parameter int TOP  = 2,
    parameter int KV_BITS = 4,
    parameter int REC_BYTES = 32,
    parameter int RPB  = 64,
    parameter int MAXR = 64,
    parameter int WINDOW_OFF = 0,
    parameter int BLOCK_OFF  = 2048,
    parameter int INDEX_OFF  = 6144,
    parameter int SUMS_OFF   = 8192,
    parameter int ATT_L = 8,
    parameter int SW_L  = 16,
    parameter int ROWS = 96,
    parameter int COLS = 16,
    parameter int P    = 2,
    parameter int NT   = 56,
    parameter int TMAX = 1,
    parameter int MODEL_TILES = 0,
    parameter int WB   = 4,
    parameter int ACC  = 24,
    parameter int SB   = 16,
    parameter int SHB  = 5,
    parameter int SW   = 38,
    parameter int YSH  = 9,
    parameter int VB_BYTES  = 4096,
    parameter int VB_BANKS  = 1,
    parameter int VB_BANK_SHIFT = 12,
    parameter int VB_NPR = 24,
    parameter int VB_NPW = 19,
    parameter [63:0] VB_RMAP0 = 64'hFEDCBA9876543210,
    parameter [63:0] VB_RMAP1 = 64'hFEDCBA9876543210,
    parameter [63:0] VB_WMAP0 = 64'hFEDCBA9876543210,
    parameter [63:0] VB_WMAP1 = 64'hFEDCBA9876543210,
    parameter [63:0] VB_RCAP2 = 0,
    parameter [63:0] VB_RCAP3 = 0,
    parameter [63:0] VB_WCAP2 = 0,
    parameter int AW        = 24,
    parameter int MEM_BEATS = 128,
    parameter int SCHEDULE_CYCLES = 0,
    parameter int USE_HPI   = 0,
    parameter int NDEV      = 4,
    parameter int DEV_WORDS = 4096,
    parameter int MR0       = 8'h18,
    parameter int MR4       = 8'h60,
    parameter int MR8       = 8'h43
);
    reg clk = 0, rst_n = 0;
    always #0.625 clk = ~clk;
    integer cycle = 0;
    always @(posedge clk) cycle <= cycle + 1;

    reg          start = 0;
    wire         running, done;
    wire         req_valid, req_ready, req_write, req_wide, wdata_valid, wdata_ready, rdata_valid;
    wire [31:0]  req_addr;
    wire [11:0]  req_beats;
    wire [255:0] wdata, rdata;
    fabric_layer_engine #(.D(D), .NK(NK), .NV(NV), .HK(HK), .HV(HV), .KK(KK), .CONV(CONV), .NH(NH), .NKV(NKV), .HD(HD), .RD(RD), .IDIM(IDIM),
                          .W(W), .BS(BS), .TOP(TOP), .KV_BITS(KV_BITS), .REC_BYTES(REC_BYTES), .RPB(RPB), .MAXR(MAXR),
                          .WINDOW_OFF(WINDOW_OFF), .BLOCK_OFF(BLOCK_OFF), .INDEX_OFF(INDEX_OFF), .SUMS_OFF(SUMS_OFF), .ATT_L(ATT_L), .SW_L(SW_L),
                          .ROWS(ROWS), .COLS(COLS), .P(P), .NT(NT), .TMAX(TMAX), .MODEL_TILES(MODEL_TILES), .WB(WB), .ACC(ACC), .SB(SB), .SHB(SHB), .SW(SW), .YSH(YSH), .VB_BYTES(VB_BYTES), .AW(AW),
                          .VB_BANKS(VB_BANKS), .VB_BANK_SHIFT(VB_BANK_SHIFT), .VB_NPR(VB_NPR), .VB_NPW(VB_NPW), .VB_RMAP0(VB_RMAP0), .VB_RMAP1(VB_RMAP1),
        .VB_WMAP0(VB_WMAP0), .VB_WMAP1(VB_WMAP1), .VB_RCAP2(VB_RCAP2), .VB_RCAP3(VB_RCAP3), .VB_WCAP2(VB_WCAP2)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .n_steps(N[15:0]), .running(running), .done(done),
        .m_req_valid(req_valid), .m_req_ready(req_ready), .m_req_write(req_write), .m_req_wide(req_wide), .m_req_addr(req_addr),
        .m_req_beats(req_beats), .m_wdata_valid(wdata_valid), .m_wdata_ready(wdata_ready), .m_wdata(wdata), .m_rdata_valid(rdata_valid),
        .m_rdata(rdata));
    wire mem_ready;                                       // the memory can take requests
    reg  dump = 0;                                        // the memory images to their files
    reg [15:0] dimg [0:NDEV*DEV_WORDS-1];                 // every device's words in turn
    generate
        if (USE_HPI) begin : g_hpi
            reg mclk = 0;
            always #2.0 mclk = ~mclk;                     // the controller clock, 250 MHz
            wire          m_req_valid, m_req_ready, m_req_write, m_wdata_valid, m_wdata_ready, m_rdata_valid, rd_overflow;
            wire [31:0]   m_req_addr;
            wire [11:0]   m_req_beats;
            wire [127:0]  m_wdata, m_rdata;
            // The bridge moves a beat a transfer; the engine's moves are two.
            wire          n_req_valid, n_req_ready, n_req_write, n_wdata_valid, n_wdata_ready, n_rdata_valid;
            wire [31:0]   n_req_addr;
            wire [11:0]   n_req_beats;
            wire [127:0]  n_wdata, n_rdata;
            fabric_mem_narrow #(.DW(128)) narrow (
                .clk(clk), .rst_n(rst_n), .w_req_valid(req_valid), .w_req_ready(req_ready), .w_req_write(req_write), .w_req_wide(req_wide),
                .w_req_addr(req_addr), .w_req_beats(req_beats), .w_wdata_valid(wdata_valid), .w_wdata_ready(wdata_ready), .w_wdata(wdata),
                .w_rdata_valid(rdata_valid), .w_rdata(rdata),
                .n_req_valid(n_req_valid), .n_req_ready(n_req_ready), .n_req_write(n_req_write), .n_req_addr(n_req_addr),
                .n_req_beats(n_req_beats), .n_wdata_valid(n_wdata_valid), .n_wdata_ready(n_wdata_ready), .n_wdata(n_wdata),
                .n_rdata_valid(n_rdata_valid), .n_rdata(n_rdata));
            fabric_mem_bridge #(.DW(128)) bridge (
                .c_clk(clk), .c_rst_n(rst_n), .c_req_valid(n_req_valid), .c_req_ready(n_req_ready), .c_req_write(n_req_write),
                .c_req_addr(n_req_addr), .c_req_beats(n_req_beats), .c_wdata_valid(n_wdata_valid), .c_wdata_ready(n_wdata_ready),
                .c_wdata(n_wdata), .c_rdata_valid(n_rdata_valid), .c_rdata(n_rdata),
                .m_clk(mclk), .m_rst_n(rst_n), .m_req_valid(m_req_valid), .m_req_ready(m_req_ready), .m_req_write(m_req_write),
                .m_req_addr(m_req_addr), .m_req_beats(m_req_beats), .m_wdata_valid(m_wdata_valid), .m_wdata_ready(m_wdata_ready),
                .m_wdata(m_wdata), .m_rdata_valid(m_rdata_valid), .m_rdata(m_rdata), .rd_overflow(rd_overflow));
            wire [NDEV-1:0]     x_valid, x_ready, x_wdata_valid, x_wdata_ready, x_rdata_valid, x_rdata_ready, x_done;
            wire                x_write;
            wire [24:0]         x_addr;
            wire [7:0]          x_beats;
            wire [127:0]        x_wdata;
            wire [NDEV*128-1:0] x_rdata;
            fabric_hpi_stripe #(.NDEV(NDEV), .DW(128)) stripe (
                .clk(mclk), .rst_n(rst_n), .req_valid(m_req_valid), .req_ready(m_req_ready), .req_write(m_req_write), .req_addr(m_req_addr),
                .req_beats(m_req_beats), .wdata_valid(m_wdata_valid), .wdata_ready(m_wdata_ready), .wdata(m_wdata), .rdata_valid(m_rdata_valid),
                .rdata(m_rdata), .x_valid(x_valid), .x_ready(x_ready), .x_write(x_write), .x_addr(x_addr), .x_beats(x_beats),
                .x_wdata_valid(x_wdata_valid), .x_wdata_ready(x_wdata_ready), .x_wdata(x_wdata), .x_rdata_valid(x_rdata_valid),
                .x_rdata_ready(x_rdata_ready), .x_rdata(x_rdata), .x_done(x_done));
            wire [NDEV-1:0]    clk_en, init_done, device_ok, ce_n, dq_oe, dqs_oe, phy_quiet;
            wire [NDEV*16-1:0] dq_c, dq_d;
            wire [NDEV*2-1:0]  dm_c, dqs_dev, dqs_d;
            wire [NDEV-1:0]    clk_dev;
            assign mem_ready = &init_done && &device_ok;
            genvar g;
            for (g = 0; g < NDEV; g = g + 1) begin : g_dev
                wire [1:0] dqs_bus = dqs_oe[g] ? dqs_dev[g*2 +: 2] : 2'b00;
                wire       clk_gated = mclk & clk_en[g];
                assign #1.0 clk_dev[g] = clk_gated;                          // ideal quarter-period delays
                assign #1.0 dqs_d[g*2 +: 2] = dqs_bus;
                wire [15:0] dq_bus_c = dq_oe[g] ? dq_c[g*16 +: 16] : 16'hzzzz;
                wire [15:0] dq_bus_d = dqs_oe[g] ? dq_d[g*16 +: 16] : 16'hzzzz;
                fabric_hpi_channel #(.MR0(MR0), .MR4(MR4), .MR8(MR8), .TPU_CYCLES(60), .TRST_CYCLES(20)) ch (
                    .clk(mclk), .rst_n(rst_n), .phy_ready(1'b1), .phy_quiet(phy_quiet[g]),
                    .clk_en(clk_en[g]), .init_done(init_done[g]), .device_ok(device_ok[g]),
                    .xact_valid(x_valid[g]), .xact_ready(x_ready[g]), .xact_write(x_write), .xact_addr(x_addr), .xact_beats(x_beats),
                    .wdata_valid(x_wdata_valid[g]), .wdata_ready(x_wdata_ready[g]), .wdata(x_wdata),
                    .rdata_valid(x_rdata_valid[g]), .rdata_ready(x_rdata_ready[g]), .rdata(x_rdata[g*128 +: 128]), .xact_done(x_done[g]),
                    .ce_n(ce_n[g]), .dq_o(dq_c[g*16 +: 16]), .dq_oe(dq_oe[g]), .dq_i(dq_bus_d), .dm_o(dm_c[g*2 +: 2]), .dm_oe(),
                    .dqs_d(dqs_d[g*2 +: 2]));
                fabric_hpi_device #(.WORDS(DEV_WORDS), .T_DQSCK_NS(2.5 + 0.5 * (g % 4)), .PUSHOUT_SEED(7 + g)) dev (
                    .clk(clk_dev[g]), .ce_n(ce_n[g]), .dq_i(dq_bus_c), .dq_o(dq_d[g*16 +: 16]), .dq_oe(),
                    .dm_i(dm_c[g*2 +: 2]), .dqs_o(dqs_dev[g*2 +: 2]), .dqs_oe(dqs_oe[g]));
                // The image in after every time-zero initialiser, and out at the dump.
                integer w;
                initial begin #1; for (w = 0; w < DEV_WORDS; w = w + 1) dev.mem[w] = dimg[g*DEV_WORDS + w]; end
                always @(posedge dump) for (w = 0; w < DEV_WORDS; w = w + 1) dimg[g*DEV_WORDS + w] = dev.mem[w];
            end
            initial $readmemh("devs.hex", dimg);
            always @(posedge dump) begin #1; $writememh("devs_out.hex", dimg); end
            always @(posedge clk) if (rd_overflow) $display("FAIL: the bridge's read fifo overflowed");
        end else begin : g_model
            fabric_mem_model #(.DW(128), .XW(2), .WORDS(MEM_BEATS), .LAT(2), .FILE("mem_init.hex")) u_mem (
                .clk(clk), .rst_n(rst_n), .req_valid(req_valid), .req_ready(req_ready), .req_write(req_write), .req_wide(req_wide),
                .req_addr(req_addr),
                .req_beats(req_beats), .wdata_valid(wdata_valid), .wdata_ready(wdata_ready), .wdata(wdata), .rdata_valid(rdata_valid), .rdata(rdata));
            assign mem_ready = 1'b1;
            always @(posedge dump) $writememh("mem_out.hex", u_mem.mem);
        end
    endgenerate

    // The issue trace, for the Python side's dependency check.
    integer trace, issues = 0, t0 = 0, guard, took = 0;
    reg finished = 0;
    always @(posedge clk) if (|(dut.cmd_valid & dut.cmd_ready)) begin
        if (issues == 0) t0 = cycle;
        $fdisplay(trace, "%0d %0d %0d %0d", issues, dut.cmd_tag, cycle, dut.u_seq.cur_unit);
        $fflush(trace);                                    // a full-size run takes hours: the trace is its progress
        issues = issues + 1;
    end

    // Vector-buffer port demand, for the banking question.  fabric_vb has no
    // read enable, so a unit's read ports are counted as live from the cycle
    // it takes a command to the cycle it reports done; the writes are counted
    // exactly, from wr_en.  cooc is the units' co-occupancy: how many cycles
    // each pair was live together, which is the interference graph a bank
    // colouring has to satisfy.
    //
    // A port's commands are a queue, not one at a time: the units drop busy on
    // the cycle they write their last beat and report done the cycle after, so
    // the controller can hand a port its next command before the last one's
    // completion arrives.
    localparam int PNU = 10, PNE = 4, PNW = 19, PQ = 4;
    function automatic integer rd_w(input integer u);
        case (u)
            0: rd_w = TMAX;                               // tiles
            1: rd_w = 2;                                  // norm: x and the gain
            2: rd_w = 2;                                  // conv: x and the history
            3: rd_w = 2;                                  // gates
            5: rd_w = 2;                                  // swiglu
            6: rd_w = 2;                                  // residual
            default: rd_w = 1;                            // delta, rotary, attn, mem
        endcase
    endfunction
    function automatic integer wr_w(input integer u);
        wr_w = (u == 2) ? 2 : 1;                          // the conv writes y and the history
    endfunction

    reg [PNU-1:0]     ulive;
    integer rd_hist [0:63], own_hist [0:63], wr_hist [0:63];
    integer cooc [0:PNU*PNU-1];
    integer pstep [0:PNU*PNE*PQ-1], pfrom [0:PNU*PNE*PQ-1];   // each pending command's step and issue cycle
    integer phead [0:PNU*PNE-1], ptail [0:PNU*PNE-1];
    integer pcyc = 0, pf, span, nissue = 0, p_i, p_j, p_e, p_q, nrd, nown, nwr;
    initial begin
        for (p_i = 0; p_i < 64; p_i = p_i + 1) begin rd_hist[p_i] = 0; own_hist[p_i] = 0; wr_hist[p_i] = 0; end
        for (p_i = 0; p_i < PNU*PNU; p_i = p_i + 1) cooc[p_i] = 0;
        for (p_i = 0; p_i < PNU*PNE; p_i = p_i + 1) begin phead[p_i] = 0; ptail[p_i] = 0; end
    end
    always @(posedge clk) if (rst_n && running) begin
        nrd = 0; nown = 0; ulive = 0;
        for (p_i = 0; p_i < PNU; p_i = p_i + 1)
            for (p_e = 0; p_e < PNE; p_e = p_e + 1)
                if (phead[p_i*PNE + p_e] != ptail[p_i*PNE + p_e]) begin
                    nrd = nrd + rd_w(p_i); nown = nown + wr_w(p_i); ulive[p_i] = 1'b1;
                end
        nwr = 0;
        for (p_i = 0; p_i < PNW; p_i = p_i + 1) if (dut.wr_en[p_i]) nwr = nwr + 1;
        rd_hist[nrd] = rd_hist[nrd] + 1;
        own_hist[nown] = own_hist[nown] + 1;
        wr_hist[nwr] = wr_hist[nwr] + 1;
        for (p_i = 0; p_i < PNU; p_i = p_i + 1)
            if (ulive[p_i])
                for (p_j = 0; p_j < PNU; p_j = p_j + 1)
                    if (ulive[p_j]) cooc[p_i*PNU + p_j] = cooc[p_i*PNU + p_j] + 1;
        pcyc = pcyc + 1;
        for (p_i = 0; p_i < PNU*PNE; p_i = p_i + 1)
            if (dut.done_valid[p_i] && phead[p_i] != ptail[p_i]) begin
                p_q = p_i * PQ + phead[p_i] % PQ;
                $fdisplay(span, "%0d %0d %0d %0d %0d", pstep[p_q], p_i / PNE, p_i % PNE, pfrom[p_q], cycle);
                phead[p_i] = phead[p_i] + 1;
            end
        for (p_i = 0; p_i < PNU; p_i = p_i + 1)
            if (dut.cmd_valid[p_i] && dut.cmd_ready[p_i]) begin
                p_e = p_i * PNE + dut.cmd_engine;
                p_q = p_e * PQ + ptail[p_e] % PQ;
                pstep[p_q] = nissue;
                pfrom[p_q] = cycle;
                ptail[p_e] = ptail[p_e] + 1;
                if (ptail[p_e] - phead[p_e] > PQ) $display("FAIL: the port queue overflowed");
                nissue = nissue + 1;
            end
    end

    task dump_ports;
        begin
            for (p_i = 0; p_i < PNU*PNE; p_i = p_i + 1)     // the last completions land after running falls
                while (phead[p_i] != ptail[p_i]) begin
                    p_q = p_i * PQ + phead[p_i] % PQ;
                    $fdisplay(span, "%0d %0d %0d %0d %0d", pstep[p_q], p_i / PNE, p_i % PNE, pfrom[p_q], cycle);
                    phead[p_i] = phead[p_i] + 1;
                end
            pf = $fopen("ports.txt", "w");
            $fdisplay(pf, "cycles %0d", pcyc);
            for (p_i = 0; p_i < VB_BANKS; p_i = p_i + 1)   // the banking the run actually asked for
                $fdisplay(pf, "bank %0d %0d %0d", p_i, dut.u_vb.max_rd[p_i], dut.u_vb.max_wr[p_i]);
            for (p_i = 0; p_i < 64; p_i = p_i + 1)
                if (rd_hist[p_i] || own_hist[p_i] || wr_hist[p_i])
                    $fdisplay(pf, "hist %0d %0d %0d %0d", p_i, rd_hist[p_i], own_hist[p_i], wr_hist[p_i]);
            for (p_i = 0; p_i < PNU; p_i = p_i + 1) begin
                $fwrite(pf, "cooc %0d", p_i);
                for (p_j = 0; p_j < PNU; p_j = p_j + 1) $fwrite(pf, " %0d", cooc[p_i*PNU + p_j]);
                $fdisplay(pf, "");
            end
            $fclose(pf);
        end
    endtask

    initial begin
        span = $fopen("span.txt", "w");
        trace = $fopen("issue.txt", "w");
        repeat (4) @(posedge clk);
        rst_n = 1;
        guard = 0;
        while (!mem_ready && guard < 200000) begin @(posedge clk); guard = guard + 1; end
        if (!mem_ready) begin $display("FAIL: the memory never came up"); $finish; end
        @(posedge clk); #0.1;
        start = 1; @(posedge clk); #0.1; start = 0;
        guard = 0;
        while (!done && guard < 4000000) begin @(posedge clk); guard = guard + 1; end
        finished = done;
        took = cycle - t0;                        // here, not after the dumps: those take clocks of their own
        $fclose(trace);
        dump_ports;
        $fclose(span);
        dut.u_vb.dumping = 1'b1; #0.1;            // the banks back into one image
        $writememh("vb_out.hex", dut.u_vb.mem);
        dump = 1; #10;
        if (!finished) $display("FAIL: never finished, %0d of %0d issued", issues, N);
        else if (issues != N) $display("FAIL: %0d issued of %0d", issues, N);
        else $display("PASS: %0d steps in %0d cycles (the timing model said %0d)", N, took, SCHEDULE_CYCLES);
        $finish;
    end
endmodule

`default_nettype wire
