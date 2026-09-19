// Self-checking harness for fabric_layer_engine: runs a program written by
// fabric.engine.EngineRun over the real units and dumps the vector buffer
// and the memory for the Python side to compare bit for bit.  The memory
// behind the port is fabric_memory.sv's behavioural model; the HPI bridge
// that replaces it on the die is checked by tb_mem_bridge.

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
    parameter int VB_BYTES  = 4096,
    parameter int MEM_BEATS = 128,
    parameter int SCHEDULE_CYCLES = 0
);
    reg clk = 0, rst_n = 0;
    always #0.625 clk = ~clk;
    integer cycle = 0;
    always @(posedge clk) cycle <= cycle + 1;

    reg          start = 0;
    wire         running, done;
    wire         req_valid, req_ready, req_write, wdata_valid, wdata_ready, rdata_valid;
    wire [31:0]  req_addr;
    wire [11:0]  req_beats;
    wire [127:0] wdata, rdata;
    fabric_layer_engine #(.D(D), .NK(NK), .NV(NV), .HK(HK), .HV(HV), .KK(KK), .CONV(CONV), .NH(NH), .NKV(NKV), .HD(HD), .RD(RD), .IDIM(IDIM),
                          .W(W), .BS(BS), .TOP(TOP), .KV_BITS(KV_BITS), .REC_BYTES(REC_BYTES), .RPB(RPB), .MAXR(MAXR),
                          .WINDOW_OFF(WINDOW_OFF), .BLOCK_OFF(BLOCK_OFF), .INDEX_OFF(INDEX_OFF), .SUMS_OFF(SUMS_OFF), .ATT_L(ATT_L),
                          .ROWS(ROWS), .COLS(COLS), .P(P), .NT(NT), .WB(WB), .ACC(ACC), .SB(SB), .SHB(SHB), .SW(SW), .YSH(YSH), .VB_BYTES(VB_BYTES)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .n_steps(N[15:0]), .running(running), .done(done),
        .m_req_valid(req_valid), .m_req_ready(req_ready), .m_req_write(req_write), .m_req_addr(req_addr), .m_req_beats(req_beats),
        .m_wdata_valid(wdata_valid), .m_wdata_ready(wdata_ready), .m_wdata(wdata), .m_rdata_valid(rdata_valid), .m_rdata(rdata));
    fabric_mem_model #(.DW(128), .WORDS(MEM_BEATS), .LAT(2), .FILE("mem_init.hex")) u_mem (
        .clk(clk), .rst_n(rst_n), .req_valid(req_valid), .req_ready(req_ready), .req_write(req_write), .req_addr(req_addr),
        .req_beats(req_beats), .wdata_valid(wdata_valid), .wdata_ready(wdata_ready), .wdata(wdata), .rdata_valid(rdata_valid), .rdata(rdata));

    // The issue trace, for the Python side's dependency check.
    integer trace, issues = 0, t0 = 0, guard;
    always @(posedge clk) if (|(dut.cmd_valid & dut.cmd_ready)) begin
        if (issues == 0) t0 = cycle;
        $fdisplay(trace, "%0d %0d %0d %0d", issues, dut.cmd_tag, cycle, dut.u_seq.cur_unit);
        issues = issues + 1;
    end

    initial begin
        trace = $fopen("issue.txt", "w");
        repeat (2) @(posedge clk);
        rst_n = 1;
        @(posedge clk); #0.1;
        start = 1; @(posedge clk); #0.1; start = 0;
        guard = 0;
        while (!done && guard < 2000000) begin @(posedge clk); guard = guard + 1; end
        $fclose(trace);
        $writememh("vb_out.hex", dut.u_vb.mem);
        $writememh("mem_out.hex", u_mem.mem);
        if (!done) $display("FAIL: never finished, %0d of %0d issued", issues, N);
        else if (issues != N) $display("FAIL: %0d issued of %0d", issues, N);
        else $display("PASS: %0d steps in %0d cycles (the timing model said %0d)", N, cycle - t0, SCHEDULE_CYCLES);
        $finish;
    end
endmodule

`default_nettype wire
