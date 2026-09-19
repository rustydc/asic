// Self-checking harness for fabric_layer_engine: runs a program written by
// fabric.engine.EngineRun over the real units and dumps the vector buffer
// and the memory for the Python side to compare bit for bit.  The memory
// behind the port is a behavioural beat memory here (fabric_beat_memory);
// the HPI bridge and its DMAs are checked by tb_mem_bridge.

`timescale 1ns/1ps
`default_nettype none

module fabric_beat_memory #(
    parameter int BEATS = 1024,
    parameter     INIT_FILE = ""
) (
    input  wire         clk,
    input  wire [31:0]  rd_addr,
    output reg  [127:0] rd_data,
    input  wire         wr_en,
    input  wire [31:0]  wr_addr,
    input  wire [127:0] wr_data
);
    reg [127:0] mem [0:BEATS-1];
    integer i;
    initial begin
        if (INIT_FILE != "") $readmemh(INIT_FILE, mem);
        else for (i = 0; i < BEATS; i = i + 1) mem[i] = 128'd0;
    end
    always @(posedge clk) begin
        rd_data <= (rd_addr < BEATS) ? mem[rd_addr] : 128'd0;
        if (wr_en && wr_addr < BEATS) mem[wr_addr] <= wr_data;
    end
endmodule

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
    wire [31:0]  mem_rd_addr, mem_wr_addr;
    wire [127:0] mem_rd_data, mem_wr_data;
    wire         mem_wr_en;
    fabric_layer_engine #(.D(D), .NK(NK), .NV(NV), .HK(HK), .HV(HV), .KK(KK), .CONV(CONV), .ROWS(ROWS), .COLS(COLS), .P(P), .NT(NT),
                          .WB(WB), .ACC(ACC), .SB(SB), .SHB(SHB), .SW(SW), .YSH(YSH), .VB_BYTES(VB_BYTES)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .n_steps(N[15:0]), .running(running), .done(done),
        .mem_rd_addr(mem_rd_addr), .mem_rd_data(mem_rd_data), .mem_wr_en(mem_wr_en), .mem_wr_addr(mem_wr_addr), .mem_wr_data(mem_wr_data));
    fabric_beat_memory #(.BEATS(MEM_BEATS), .INIT_FILE("mem_init.hex")) u_mem (
        .clk(clk), .rd_addr(mem_rd_addr), .rd_data(mem_rd_data), .wr_en(mem_wr_en), .wr_addr(mem_wr_addr), .wr_data(mem_wr_data));

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
