// Self-checking testbench: the stripe unit, NDEV channel controllers and
// NDEV device models; random bursts from fabric.hpi.emit_hpi_vectors, the
// read data checked beat by beat and the device images at the end.

`timescale 1ns/1ps
`default_nettype none

module tb_hpi #(
    parameter int NDEV      = 4,
    parameter int DEV_WORDS = 8192,
    parameter int N         = 24,
    parameter int NW        = 1,
    parameter int NR        = 1,
    parameter int MR0       = 8'h18,
    parameter int MR4       = 8'h60,
    parameter int MR8       = 8'h43,
    parameter int  USE_DLL  = 0,                      // the PHY's delay lines and DLL instead of ideal delays
    parameter real TAP_PS   = 60.0,
    parameter int  TAPS     = 256
);
    localparam int DW = 128;
    localparam real T = 4.0;                          // 250 MHz
    reg clk = 0, rst_n = 0;
    always #(T / 2) clk = ~clk;

    reg [47:0]  reqs [0:N-1];
    reg [127:0] wdm [0:NW-1];
    reg [127:0] erd [0:NR-1];

    // Port.
    reg           req_valid = 0, req_write = 0, wdata_valid = 0;
    reg [31:0]    req_addr = 0;
    reg [11:0]    req_beats = 0;
    reg [DW-1:0]  wdata = 0;
    wire          req_ready, wdata_ready, rdata_valid;
    wire [DW-1:0] rdata;
    // Channels.
    wire [NDEV-1:0]    x_valid, x_ready, x_wdata_valid, x_wdata_ready, x_rdata_valid, x_rdata_ready, x_done;
    wire               x_write;
    wire [24:0]        x_addr;
    wire [7:0]         x_beats;
    wire [DW-1:0]      x_wdata;
    wire [NDEV*DW-1:0] x_rdata;
    fabric_hpi_stripe #(.NDEV(NDEV), .DW(DW)) stripe (
        .clk(clk), .rst_n(rst_n), .req_valid(req_valid), .req_ready(req_ready), .req_write(req_write), .req_addr(req_addr),
        .req_beats(req_beats), .wdata_valid(wdata_valid), .wdata_ready(wdata_ready), .wdata(wdata), .rdata_valid(rdata_valid),
        .rdata_ready(1'b1), .rdata(rdata), .x_valid(x_valid), .x_ready(x_ready), .x_write(x_write), .x_addr(x_addr), .x_beats(x_beats),
        .x_wdata_valid(x_wdata_valid), .x_wdata_ready(x_wdata_ready), .x_wdata(x_wdata), .x_rdata_valid(x_rdata_valid),
        .x_rdata_ready(x_rdata_ready), .x_rdata(x_rdata), .x_done(x_done));

    wire [NDEV-1:0] clk_en, init_done, device_ok, ce_n, dq_oe, dqs_oe, phy_quiet;
    // The PHY: one master DLL on the controller clock; the slaves take its
    // quarter code when every channel has CE# high.
    localparam int CW = $clog2(TAPS);
    wire          dll_locked, dll_range_err;
    wire [CW-1:0] dll_period, quarter;
    fabric_dll #(.TAPS(TAPS), .TAP_PS(TAP_PS)) dll (.clk(clk), .rst_n(rst_n), .update_ok(&phy_quiet), .locked(dll_locked),
                                                    .range_err(dll_range_err), .period_code(dll_period), .quarter(quarter));
    wire phy_ready = USE_DLL ? dll_locked : 1'b1;
    wire [NDEV*16-1:0] dq_c, dq_d;
    wire [NDEV*2-1:0]  dm_c, dqs_dev;
    wire [NDEV*2-1:0]  dqs_d;
    wire [NDEV-1:0]    clk_dev;
    genvar g;
    generate
        for (g = 0; g < NDEV; g = g + 1) begin : g_dev
            // The device clock: gated and a quarter period late.  DQS to the
            // controller: a quarter period late, low when the device releases it.
            // Either ideal delays or the PHY's slave delay lines on the DLL's code.
            wire [1:0] dqs_bus = dqs_oe[g] ? dqs_dev[g*2 +: 2] : 2'b00;
            wire       clk_gated = clk & clk_en[g];
            if (USE_DLL) begin : g_phy
                fabric_delay_line #(.TAPS(TAPS), .TAP_PS(TAP_PS)) l_clk  (.in(clk_gated), .code(quarter), .out(clk_dev[g]));
                fabric_delay_line #(.TAPS(TAPS), .TAP_PS(TAP_PS)) l_dqs0 (.in(dqs_bus[0]), .code(quarter), .out(dqs_d[g*2]));
                fabric_delay_line #(.TAPS(TAPS), .TAP_PS(TAP_PS)) l_dqs1 (.in(dqs_bus[1]), .code(quarter), .out(dqs_d[g*2+1]));
            end else begin : g_ideal
                assign #(T / 4) clk_dev[g] = clk_gated;
                assign #(T / 4) dqs_d[g*2 +: 2] = dqs_bus;
            end
            wire [15:0] dq_bus_c = dq_oe[g] ? dq_c[g*16 +: 16] : 16'hzzzz;   // controller drive
            wire [15:0] dq_bus_d = dqs_oe[g] ? dq_d[g*16 +: 16] : 16'hzzzz;  // device drive (DQ with DQS)
            fabric_hpi_channel #(.MR0(MR0), .MR4(MR4), .MR8(MR8), .TPU_CYCLES(60), .TRST_CYCLES(20)) ch (
                .clk(clk), .rst_n(rst_n), .phy_ready(phy_ready), .phy_quiet(phy_quiet[g]),
                .clk_en(clk_en[g]), .init_done(init_done[g]), .device_ok(device_ok[g]),
                .xact_valid(x_valid[g]), .xact_ready(x_ready[g]), .xact_write(x_write), .xact_addr(x_addr), .xact_beats(x_beats),
                .wdata_valid(x_wdata_valid[g]), .wdata_ready(x_wdata_ready[g]), .wdata(x_wdata),
                .rdata_valid(x_rdata_valid[g]), .rdata_ready(x_rdata_ready[g]), .rdata(x_rdata[g*DW +: DW]), .xact_done(x_done[g]),
                .ce_n(ce_n[g]), .dq_o(dq_c[g*16 +: 16]), .dq_oe(dq_oe[g]), .dq_i(dq_bus_d), .dm_o(dm_c[g*2 +: 2]), .dm_oe(),
                .dqs_d(dqs_d[g*2 +: 2]));
            // The image check, one per device.
            always @(posedge check) begin : chk
                integer w;
                for (w = 0; w < DEV_WORDS; w = w + 1)
                    if (dev.mem[w] !== edev[g*DEV_WORDS + w]) begin
                        errors = errors + 1;
                        if (errors <= 5) $display("device %0d word %0d: got %h expected %h", g, w, dev.mem[w], edev[g*DEV_WORDS + w]);
                    end
            end
            fabric_hpi_device #(.WORDS(DEV_WORDS), .T_DQSCK_NS(2.5 + 0.5 * g), .PUSHOUT_SEED(7 + g)) dev (
                .clk(clk_dev[g]), .ce_n(ce_n[g]), .dq_i(dq_bus_c), .dq_o(dq_d[g*16 +: 16]), .dq_oe(),
                .dm_i(dm_c[g*2 +: 2]), .dqs_o(dqs_dev[g*2 +: 2]), .dqs_oe(dqs_oe[g]));
        end
    endgenerate

    integer n, b, errors, got, wi, guard;
    reg [15:0] edev [0:NDEV*DEV_WORDS-1];
    reg check = 0;
    always @(posedge clk) begin
        if (rdata_valid) begin
            if (got < NR && rdata !== erd[got]) begin
                errors = errors + 1;
                if (errors <= 5) $display("read beat %0d: got %h expected %h", got, rdata, erd[got]);
            end
            got = got + 1;
        end
    end

    initial begin
        $readmemh("reqs.hex", reqs);
        $readmemh("wdata.hex", wdm);
        $readmemh("expected_rdata.hex", erd);
        errors = 0; got = 0; wi = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        guard = 0;
        while (!(&init_done) && guard < 20000) begin @(posedge clk); guard = guard + 1; end
        if (!(&init_done)) begin $display("FAIL: initialisation never completed"); $finish; end
        if (!(&device_ok)) begin $display("FAIL: device identification %b", device_ok); $finish; end
        for (n = 0; n < N; n = n + 1) begin
            @(negedge clk);
            while (!req_ready) @(negedge clk);
            req_valid = 1; req_write = reqs[n][44]; req_beats = reqs[n][43:32]; req_addr = reqs[n][31:0];
            @(posedge clk); #0.1;
            req_valid = 0;
            if (reqs[n][44]) begin
                for (b = 0; b < reqs[n][43:32]; b = b + 1) begin
                    @(negedge clk);
                    wdata_valid = 1; wdata = wdm[wi];
                    while (!wdata_ready) @(negedge clk);
                    @(posedge clk); #0.1;
                    wdata_valid = 0; wi = wi + 1;
                    if (b % 7 == 3) @(negedge clk);              // a bubble now and then
                end
            end
            guard = 0;
            while (!req_ready && guard < 200000) begin @(posedge clk); guard = guard + 1; end
            if (!req_ready) begin $display("FAIL: request %0d never completed", n); $finish; end
        end
        // Requests overlap: the last may still be running when it is taken.
        guard = 0;
        while (!stripe.idle && guard < 200000) begin @(posedge clk); guard = guard + 1; end
        repeat (50) @(posedge clk);
        if (got != NR) $display("FAIL: %0d read beats of %0d", got, NR);
        $readmemh("expected_devs.hex", edev);
        check = 1;
        #1;
        if (USE_DLL && !dll_locked) begin errors = errors + 1; $display("DLL not locked"); end
        if (errors == 0 && USE_DLL) $display("PASS: %0d requests over %0d devices, DLL quarter %0d taps of %0.0f ps", N, NDEV, quarter, TAP_PS);
        else if (errors == 0) $display("PASS: %0d requests over %0d devices", N, NDEV);
        else $display("FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule

`default_nettype wire
