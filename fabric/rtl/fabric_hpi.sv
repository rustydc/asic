// The HPI controller for the AP Memory APS512XXN-OB9-BG (x16 Xccela DDR
// PSRAM) and the stripe unit that puts the die's memory port over sixteen
// of them.  Golden model: fabric/hpi.py.
//
// Clocking.  The controller runs at the device clock (250 MHz).  It launches
// rising-edge data at its posedge and falling-edge data at its negedge, and
// the device receives a copy of the clock delayed by a quarter period, so
// every launch sits in the middle of the device's sampling window (tSP and
// tHD of 0.5 ns against a 2 ns half period).  Reads are captured by the
// device's DQS delayed by a quarter period; the delay lines are the PHY's
// and are modelled here by `#` delays in the testbench.  The crossing to
// the 800 MHz core is an asynchronous FIFO outside these modules.
//
// Frame.  CE# low; the instruction on A/DQ[7:0] at the first rising edge;
// A3, A2, A1, A0 at the second rising, second falling, third rising and
// third falling edges (A3 = RA[14:13], A2 = RA[12:5], A1 = RA[4:0] CA[10:8],
// A0 = CA[7:0]).  The device drives DQS low from the third clock and the
// first read word `latency` clocks later, plus up to `latency` more of
// refresh push-out under variable latency: the controller does not count,
// it captures on DQS.  Writes present their first word WLC clocks after the
// third clock, DM high on the byte lanes not written; register writes have
// latency 1 and use A/DQ[7:0] only.  Linear bursts wrap inside the 1024-word
// page, so the stripe unit never crosses one; CE# stays low for at most
// tCEM (4 us) and rises for at least tCPH (28 ns) between bursts.

`timescale 1ns/1ps
`default_nettype none

// ---------------------------------------------------------------------------
// Behavioural device, x16 array access after MR8[6] is set, register access
// on the low byte.  Not for synthesis.  Checks the protocol it can see:
// even addresses, x16 mode for array access, tCEM, tCPH and the write
// latency, and reports with $error.
// ---------------------------------------------------------------------------
module fabric_hpi_device #(
    parameter int  WORDS      = 8192,          // words in this model (the part has 32M)
    parameter      FILE       = "",
    parameter real T_CLK_NS   = 4.0,
    parameter real T_DQSCK_NS = 3.0,           // DQS/DQ output delay from the clock edge (2.0 to 6.5)
    parameter int  PUSHOUT_SEED = 1,           // refresh push-out pattern
    parameter int  F_MHZ      = 250
) (
    input  wire        clk,
    input  wire        ce_n,
    input  wire [15:0] dq_i,
    output reg  [15:0] dq_o,
    output reg         dq_oe,
    input  wire [1:0]  dm_i,
    output reg  [1:0]  dqs_o,
    output reg         dqs_oe
);
    localparam int AW = $clog2(WORDS);
    reg [15:0] mem [0:WORDS-1];
    integer init_w;
    initial begin
        for (init_w = 0; init_w < WORDS; init_w = init_w + 1) mem[init_w] = 16'h0000;
        if (FILE != "") $readmemh(FILE, mem);
    end
    reg [7:0] mr0, mr4, mr8;
    wire      x16 = mr8[6];
    // Latency from the register codes.
    function automatic integer read_latency(input [2:0] code);
        case (code) 3'b000: read_latency = 3; 3'b001: read_latency = 4; 3'b010: read_latency = 5; 3'b011: read_latency = 6;
                    3'b100: read_latency = 7; 3'b101: read_latency = 9; default: read_latency = 10; endcase
    endfunction
    function automatic integer write_latency(input [2:0] code);
        case (code) 3'b000: write_latency = 3; 3'b100: write_latency = 4; 3'b010: write_latency = 5; 3'b110: write_latency = 6;
                    3'b001: write_latency = 7; 3'b101: write_latency = 8; default: write_latency = 9; endcase
    endfunction

    // Frame state.
    integer  edge_n;                           // clock edges since CE# fell (0 = first rising)
    reg [7:0] instr, a3, a2, a1, a0;
    reg [24:0] waddr;                          // current word address
    integer  data_edge;                        // edge at which data starts
    integer  pushout;
    reg      is_read, is_write, is_mrr, is_mrw, active;
    real     ce_fall_t, ce_rise_t;
    integer  lfsr;

    initial begin
        mr0 = 8'h08; mr4 = 8'h40; mr8 = 8'h01;    // x8, latency 5/5, hybrid wrap 32 (the defaults)
        dq_o = 0; dq_oe = 0; dqs_o = 0; dqs_oe = 0; active = 0; edge_n = 0; lfsr = PUSHOUT_SEED;
        ce_rise_t = -1.0e9; ce_fall_t = 0.0;
    end

    // CE# edges.
    always @(negedge ce_n) begin
        if ($realtime - ce_rise_t < 28.0 - 0.01 && ce_rise_t > 0)
            $error("device: CE# low only %0.2f ns after rising, tCPH is 28 ns", $realtime - ce_rise_t);
        ce_fall_t = $realtime;
        edge_n = 0; active = 1; is_read = 0; is_write = 0; is_mrr = 0; is_mrw = 0;
        // Push-out of 0..latency clocks, from a small LFSR.
        lfsr = (lfsr * 1103515245 + 12345) & 32'h7fffffff;
    end
    always @(posedge ce_n) begin
        if (active && $realtime - ce_fall_t > 4000.0) $error("device: CE# low for %0.1f ns, tCEM is 4 us", $realtime - ce_fall_t);
        ce_rise_t = $realtime;
        active = 0;
        dq_oe <= #6 0; dqs_oe <= #6 0;         // tHZ
    end

    // The clock edges while selected.
    task automatic latch_edge(input rising);
        integer lat;
        begin
            case (edge_n)
                0: if (rising) instr = dq_i[7:0];
                2: a3 = dq_i[7:0];
                3: a2 = dq_i[7:0];
                4: a1 = dq_i[7:0];
                5: begin
                    a0 = dq_i[7:0];
                    waddr = {a3[1:0], a2, a1[7:3], a1[1:0], a0};   // RA[14:0] . CA[9:0]; CA[10] (a1[2]) is ignored at x16
                    case (instr)
                        8'h00, 8'h20: begin
                            is_read = 1;
                            if (!x16) $error("device: array read in x8 mode");
                            if (a0[0]) $error("device: odd start address");
                            lat = read_latency(mr0[4:2]);
                            pushout = mr0[5] ? lat : (lfsr % (lat + 1));
                            data_edge = 4 + 2 * (lat + pushout);          // rising edge `lat` clocks after the third
                        end
                        8'h80, 8'hA0: begin
                            is_write = 1;
                            if (!x16) $error("device: array write in x8 mode");
                            data_edge = 4 + 2 * write_latency(mr4[7:5]);
                        end
                        8'h40: begin
                            is_mrr = 1;
                            lat = read_latency(mr0[4:2]) - (F_MHZ > 200 ? 1 : 0);
                            pushout = 0;
                            data_edge = 4 + 2 * lat;
                        end
                        8'hC0: begin is_mrw = 1; data_edge = 4 + 2; end
                        8'hFF: begin mr0 = 8'h08; mr4 = 8'h40; mr8 = 8'h01; end
                        default: $error("device: unknown instruction %h", instr);
                    endcase
                end
                default: ;
            endcase
            // Writes capture from data_edge on.
            if (is_write && edge_n >= data_edge) begin
                if (!dm_i[0]) mem[waddr[AW-1:0]][7:0]  = dq_i[7:0];
                if (!dm_i[1]) mem[waddr[AW-1:0]][15:8] = dq_i[15:8];
                waddr = {waddr[24:10], waddr[9:0] + 10'd1};
            end
            if (is_mrw && edge_n == data_edge) begin
                case (a0)
                    8'h00: mr0 = dq_i[7:0];
                    8'h04: mr4 = dq_i[7:0];
                    8'h08: mr8 = dq_i[7:0];
                    8'h06: ;                                  // power modes: not modelled
                    default: $error("device: write to register %h", a0);
                endcase
            end
            edge_n = edge_n + 1;
        end
    endtask

    // Read data and strobe, launched tDQSCK after the edge.
    task automatic drive_edge(input rising);
        reg [15:0] word;
        begin
            // The strobe preamble: DQS driven low from the third clock of any read frame.
            if (edge_n == 4 && (instr == 8'h00 || instr == 8'h20 || instr == 8'h40)) begin
                dqs_oe <= #T_DQSCK_NS 1; dqs_o <= #T_DQSCK_NS 2'b00;
            end
            if (is_read) begin
                if (edge_n >= data_edge) begin
                    word = mem[waddr[AW-1:0]];
                    dq_oe <= #T_DQSCK_NS 1;
                    dq_o  <= #T_DQSCK_NS word;
                    dqs_o <= #T_DQSCK_NS (rising ? 2'b11 : 2'b00);
                    waddr = {waddr[24:10], waddr[9:0] + 10'd1};
                end
            end else if (is_mrr) begin
                if (edge_n == data_edge) begin
                    case (a0)
                        8'h00: word = mr0;
                        8'h01: word = 8'h0D | (1 << 7);                   // ULP, vendor APM
                        8'h02: word = (3'b110 << 5) | (2'b11 << 3) | 3'b110;   // good die, generation 4, 512Mb
                        8'h03: word = 8'h00;
                        8'h04: word = mr4;
                        8'h08: word = mr8;
                        default: word = 8'h00;
                    endcase
                    dq_oe <= #T_DQSCK_NS 1;
                    dq_o  <= #T_DQSCK_NS {8'h00, word[7:0]};
                    dqs_o <= #T_DQSCK_NS 2'b01;
                end else if (edge_n == data_edge + 1) begin
                    dqs_o <= #T_DQSCK_NS 2'b00;
                end
            end
        end
    endtask

    always @(posedge clk) if (!ce_n) begin drive_edge(1); latch_edge(1); end
    always @(negedge clk) if (!ce_n) begin drive_edge(0); latch_edge(0); end
endmodule

// ---------------------------------------------------------------------------
// One device's controller.  Transactions of up to 128 beats (a page) within
// one page: the write data is collected first into a page buffer, a read
// fills one and is then drained by beats.  Initialisation: the power-up
// wait with the clock stopped, a global reset, the three mode registers,
// then a check of vendor and density.
// ---------------------------------------------------------------------------
module fabric_hpi_channel #(
    parameter int MR0 = 8'h18,                 // variable latency 10, full drive
    parameter int MR4 = 8'h60,                 // write latency 9
    parameter int MR8 = 8'h43,                 // x16, 1K-word wrap
    parameter int WLC = 9,
    parameter int TPU_CYCLES  = 37500,         // 150 us at 250 MHz
    parameter int TRST_CYCLES = 500,           // 2 us
    parameter int TCPH_CYCLES = 7,             // 28 ns
    parameter int TCEM_CYCLES = 900,           // under 4 us
    parameter int F_MHZ = 250
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         phy_ready,             // the PHY's delay lines are set (the DLL is locked)
    output wire         phy_quiet,             // the slave delay lines may take a new code (CE# high)
    output reg          clk_en,                // the device clock runs only after the power-up wait
    output reg          init_done,
    output reg          device_ok,
    // transaction
    input  wire         xact_valid,
    output wire         xact_ready,
    input  wire         xact_write,
    input  wire [24:0]  xact_addr,             // word address, even
    input  wire [7:0]   xact_beats,            // 1..128
    input  wire         wdata_valid,
    output wire         wdata_ready,
    input  wire [127:0] wdata,
    output reg          rdata_valid,
    input  wire         rdata_ready,
    output wire [127:0] rdata,
    output reg          xact_done,
    // device
    output reg          ce_n,
    output wire [15:0]  dq_o,
    output reg          dq_oe,
    input  wire [15:0]  dq_i,
    output wire [1:0]   dm_o,
    output reg          dm_oe,
    input  wire [1:0]   dqs_d                  // DQS delayed a quarter period, the capture strobe
);
    // Page buffer: 128 beats of 128 bits, addressed by word for the DDR side.
    reg [127:0] buffer [0:127];
    reg [7:0]   nbeats;
    reg [7:0]   bcount;                        // beats collected or drained
    reg [10:0]  wcount;                        // words moved on the device side
    reg         write_r;
    reg [24:0]  addr_r;

    // DDR launch: rising-edge and falling-edge values.
    reg [15:0] d_rise, d_fall;
    reg [1:0]  dm_rise, dm_fall;
    assign dq_o = clk ? d_rise : d_fall;
    assign dm_o = clk ? dm_rise : dm_fall;

    // DQS-domain capture into a small ring, gray pointers to the clock domain.
    reg [15:0] cap [0:15];
    reg [4:0]  wp_r, wp_f;                     // words captured on rising and falling strobes
    always @(posedge dqs_d[0]) begin cap[{wp_r[3:0]}] <= dq_i; wp_r <= wp_r + 5'd1; end
    // Falling-edge words go to the odd slots: the two writers keep separate counters.
    reg [15:0] capf [0:15];
    always @(negedge dqs_d[0]) begin capf[{wp_f[3:0]}] <= dq_i; wp_f <= wp_f + 5'd1; end
    wire [4:0] wp_r_gray = wp_r ^ (wp_r >> 1);
    wire [4:0] wp_f_gray = wp_f ^ (wp_f >> 1);
    reg  [4:0] gr1, gr2, gf1, gf2;
    always @(posedge clk) begin gr1 <= wp_r_gray; gr2 <= gr1; gf1 <= wp_f_gray; gf2 <= gf1; end
    function automatic [4:0] ungray(input [4:0] g);
        integer i;
        begin ungray[4] = g[4]; for (i = 3; i >= 0; i = i - 1) ungray[i] = ungray[i+1] ^ g[i]; end
    endfunction
    wire [4:0] wp_r_sync = ungray(gr2);
    wire [4:0] wp_f_sync = ungray(gf2);
    reg  [4:0] rp;                             // pairs consumed
    wire       pair_ready = (wp_r_sync != rp) && (wp_f_sync != rp);
    reg        cap_clear;                      // pointers reset between transactions (in the clock domain, while DQS is quiet)

    localparam [3:0] S_PU = 0, S_RESET = 1, S_TRST = 2, S_MRW = 3, S_MRR = 4, S_IDLE = 5, S_COLLECT = 6, S_CMD = 7,
                     S_WLAT = 8, S_WDATA = 9, S_WEND = 10, S_RDATA = 11, S_CPH = 12, S_DRAIN = 13, S_DONE = 14;
    reg [3:0]  state;
    reg [15:0] wait_r;
    reg [2:0]  step;                           // init step: MR0, MR4, MR8, MRR1, MRR2
    reg [7:0]  frame [0:4];                    // instruction, A3..A0
    reg [3:0]  fedge;                          // rising edges of the frame so far
    reg [7:0]  reg_data;
    reg [7:0]  mr1_r, mr2_r;
    reg        wdata_phase;                    // odd words launched at falling edges

    assign xact_ready  = (state == S_IDLE) && init_done;
    assign wdata_ready = (state == S_COLLECT);
    assign rdata       = buffer[bcount];
    assign phy_quiet   = ce_n;                 // no frame or data on the wires while CE# is high

    task automatic load_frame(input [7:0] cmd, input [24:0] wa, input [7:0] mr);
        begin
            frame[0] = cmd;
            if (cmd == 8'hC0 || cmd == 8'h40) begin frame[1] = 0; frame[2] = 0; frame[3] = 0; frame[4] = mr; end
            else begin
                frame[1] = {6'b0, wa[24:23]};                       // RA[14:13]
                frame[2] = wa[22:15];                               // RA[12:5]
                frame[3] = {wa[14:10], 1'b0, wa[9:8]};              // RA[4:0], CA[10] = 0, CA[9:8]
                frame[4] = wa[7:0];                                 // CA[7:0]
            end
        end
    endtask

    // Rising edges: the sequencer.  Frame byte k goes out at the posedge
    // where fedge == k (k = 0 instruction, 1 A3, 2 A1); the device, clocked a
    // quarter period later, samples it on its rising edge k.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_PU; wait_r <= 0; clk_en <= 1'b0; init_done <= 1'b0; device_ok <= 1'b0; ce_n <= 1'b1; dq_oe <= 1'b0;
            d_rise <= 0; dm_rise <= 2'b11; dm_oe <= 1'b0; rdata_valid <= 1'b0; xact_done <= 1'b0; step <= 0; fedge <= 0; rp <= 0;
            cap_clear <= 1'b0; bcount <= 0; wcount <= 0; nbeats <= 0; write_r <= 1'b0; addr_r <= 0; wdata_phase <= 1'b0;
        end else begin
            xact_done <= 1'b0;
            if (state != S_WDATA && state != S_WEND) dm_rise <= 2'b11;
            case (state)
                S_PU: begin                                   // power-up: clock stopped, CE# high; the PHY locks meanwhile
                    if (wait_r != TPU_CYCLES - 1) wait_r <= wait_r + 1'b1;
                    else if (phy_ready) begin clk_en <= 1'b1; state <= S_RESET; fedge <= 0; wait_r <= 0; end
                end
                S_RESET: begin                                // global reset: FFh, CE# low four clocks
                    if (fedge == 0) begin ce_n <= 1'b0; d_rise <= 16'h00FF; dq_oe <= 1'b1; end
                    fedge <= fedge + 1'b1;
                    if (fedge == 4) begin ce_n <= 1'b1; dq_oe <= 1'b0; state <= S_TRST; wait_r <= 0; end
                end
                S_TRST: begin
                    wait_r <= wait_r + 1'b1;
                    if (wait_r == TRST_CYCLES - 1) begin state <= S_MRW; step <= 0; fedge <= 0; end
                end
                S_MRW: begin                                  // MR0, MR4, MR8 in turn; the byte at rising edge 3 (latency 1)
                    if (fedge == 0) begin
                        load_frame(8'hC0, 25'd0, step == 0 ? 8'h00 : step == 1 ? 8'h04 : 8'h08);
                        reg_data <= step == 0 ? MR0[7:0] : step == 1 ? MR4[7:0] : MR8[7:0];
                        ce_n <= 1'b0; dq_oe <= 1'b1;
                    end
                    if (fedge == 0) d_rise <= {8'h00, frame[0]};
                    if (fedge == 1) d_rise <= {8'h00, frame[1]};
                    if (fedge == 2) d_rise <= {8'h00, frame[3]};
                    if (fedge == 3) d_rise <= {8'h00, reg_data};
                    fedge <= fedge + 1'b1;
                    if (fedge == 4) begin ce_n <= 1'b1; dq_oe <= 1'b0; fedge <= 0; wait_r <= 0; state <= S_CPH; end
                end
                S_MRR: begin                                  // MR1 then MR2, the byte captured on DQS0
                    if (fedge == 0) begin
                        load_frame(8'h40, 25'd0, step == 3 ? 8'h01 : 8'h02);
                        ce_n <= 1'b0; dq_oe <= 1'b1; cap_clear <= 1'b1; rp <= 0;
                    end
                    if (fedge == 0) d_rise <= {8'h00, frame[0]};
                    if (fedge == 1) d_rise <= {8'h00, frame[1]};
                    if (fedge == 2) d_rise <= {8'h00, frame[3]};
                    if (fedge == 3) begin dq_oe <= 1'b0; cap_clear <= 1'b0; end
                    if (fedge < 15) fedge <= fedge + 1'b1;
                    if (fedge >= 6 && wp_r_sync != rp) begin       // the synchronisers have settled after the clear
                        if (step == 3) mr1_r <= cap[rp[3:0]][7:0]; else mr2_r <= cap[rp[3:0]][7:0];
                        rp <= rp + 5'd1;
                        ce_n <= 1'b1; fedge <= 0; wait_r <= 0; state <= S_CPH;
                    end
                end
                S_CPH: begin                                  // CE# high for tCPH, then the next step
                    wait_r <= wait_r + 1'b1;
                    if (wait_r == TCPH_CYCLES - 1) begin
                        if (!init_done) begin
                            if (step < 2) begin step <= step + 1'b1; state <= S_MRW; fedge <= 0; end
                            else if (step == 2) begin step <= 3; state <= S_MRR; fedge <= 0; end
                            else if (step == 3) begin step <= 4; state <= S_MRR; fedge <= 0; end
                            else begin
                                init_done <= 1'b1;
                                device_ok <= (mr1_r[4:0] == 5'b01101) && (mr2_r[2:0] == 3'b110) && (mr2_r[7:5] == 3'b110);
                                state <= S_IDLE;
                            end
                        end else if (write_r) state <= S_DONE;
                        else begin state <= S_DRAIN; bcount <= 0; end
                    end
                end
                S_IDLE: if (xact_valid) begin
                    write_r <= xact_write; addr_r <= xact_addr; nbeats <= xact_beats; bcount <= 0; wcount <= 0; fedge <= 0;
                    state <= xact_write ? S_COLLECT : S_CMD;
                end
                S_COLLECT: if (wdata_valid) begin
                    buffer[bcount] <= wdata;
                    bcount <= bcount + 1'b1;
                    if (bcount == nbeats - 1) begin state <= S_CMD; fedge <= 0; end
                end
                S_CMD: begin
                    if (fedge == 0) begin
                        load_frame(write_r ? 8'hA0 : 8'h20, addr_r, 8'h00);
                        ce_n <= 1'b0; dq_oe <= 1'b1; dm_oe <= write_r; rp <= 0; cap_clear <= 1'b1;
                    end
                    if (fedge == 0) d_rise <= {8'h00, frame[0]};
                    if (fedge == 1) d_rise <= {8'h00, frame[1]};
                    if (fedge == 2) d_rise <= {8'h00, frame[3]};
                    fedge <= fedge + 1'b1;
                    if (fedge == 2) begin
                        wait_r <= 0; wcount <= 0;
                        state <= write_r ? S_WLAT : S_RDATA;
                    end
                end
                S_WLAT: begin                                 // word 0 goes out at rising edge 2 + WLC
                    wait_r <= wait_r + 1'b1;
                    if (wait_r == WLC - 1) begin
                        d_rise <= buffer[0][15:0]; dm_rise <= 2'b00; wdata_phase <= 1'b1;
                        wcount <= 1;
                        state <= (nbeats * 8 > 2) ? S_WDATA : S_WEND;
                    end
                end
                S_WDATA: begin                                // even words at rising edges; the odd ones below
                    d_rise <= buffer[(wcount + 1) >> 3][((wcount + 1) & 7)*16 +: 16];
                    wcount <= wcount + 2;
                    if (wcount + 3 >= nbeats * 8) state <= S_WEND;
                end
                S_WEND: begin                                 // the last odd word went out at the falling edge before this
                    ce_n <= 1'b1; dq_oe <= 1'b0; dm_oe <= 1'b0; wdata_phase <= 1'b0; cap_clear <= 1'b0;
                    state <= S_CPH; wait_r <= 0;
                end
                S_RDATA: begin                                // pairs arrive on DQS
                    dq_oe <= 1'b0; cap_clear <= 1'b0;
                    wait_r <= wait_r + 1'b1;
                    // The first pair is taken only once the pointer synchronisers have
                    // settled after the clear; data cannot arrive within the read latency.
                    if (pair_ready && wait_r >= 4) begin
                        buffer[wcount >> 3][(wcount & 7)*16 +: 16]       <= cap[rp[3:0]];
                        buffer[wcount >> 3][((wcount & 7) + 1)*16 +: 16] <= capf[rp[3:0]];
                        rp <= rp + 5'd1;
                        wcount <= wcount + 2;
                        if (wcount + 2 >= nbeats * 8) begin ce_n <= 1'b1; state <= S_CPH; wait_r <= 0; end
                    end
                    if (wait_r == TCEM_CYCLES) begin ce_n <= 1'b1; state <= S_CPH; wait_r <= 0; end   // no data: give up
                end
                S_DRAIN: begin
                    rdata_valid <= 1'b1;
                    if (rdata_valid && rdata_ready) begin
                        bcount <= bcount + 1'b1;
                        if (bcount == nbeats - 1) begin rdata_valid <= 1'b0; state <= S_DONE; end
                    end
                end
                S_DONE: begin xact_done <= 1'b1; state <= S_IDLE; end
                default: state <= S_IDLE;
            endcase
        end
    end

    // Falling edges: A2 after rising edge 1, A0 after rising edge 2 (fedge
    // has already advanced), and the odd write words.
    always @(negedge clk) begin
        if ((state == S_CMD || state == S_MRW || state == S_MRR) && fedge == 2) d_fall <= {8'h00, frame[2]};
        else if ((state == S_CMD || state == S_MRW || state == S_MRR || state == S_WLAT || state == S_RDATA) && fedge == 3) d_fall <= {8'h00, frame[4]};
        else if (wdata_phase) d_fall <= buffer[wcount >> 3][(wcount & 7)*16 +: 16];
        else d_fall <= d_rise;
        dm_fall <= wdata_phase ? 2'b00 : 2'b11;
    end

    // The capture counters live in the strobe domain; they are cleared from
    // here while the strobe is quiet (an asynchronous clear in silicon).
    always @(posedge clk) if (cap_clear) begin wp_r <= 0; wp_f <= 0; end
endmodule

// ---------------------------------------------------------------------------
// The die's memory port over NDEV devices: consecutive stripes on
// consecutive devices, a burst split into stripe chunks that run on their
// devices concurrently, read data returned in order from the channels'
// page buffers.
// ---------------------------------------------------------------------------
module fabric_hpi_stripe #(
    parameter int NDEV = 16,
    parameter int DW   = 128
) (
    input  wire               clk,
    input  wire               rst_n,
    // the port
    input  wire               req_valid,
    output wire               req_ready,
    input  wire               req_write,
    input  wire [31:0]        req_addr,
    input  wire [11:0]        req_beats,
    input  wire               wdata_valid,
    output wire               wdata_ready,
    input  wire [DW-1:0]      wdata,
    output wire               rdata_valid,
    output wire [DW-1:0]      rdata,
    // the channels
    output reg  [NDEV-1:0]    x_valid,
    input  wire [NDEV-1:0]    x_ready,
    output reg                x_write,
    output reg  [24:0]        x_addr,
    output reg  [7:0]         x_beats,
    output wire [NDEV-1:0]    x_wdata_valid,
    input  wire [NDEV-1:0]    x_wdata_ready,
    output wire [DW-1:0]      x_wdata,
    input  wire [NDEV-1:0]    x_rdata_valid,
    output wire [NDEV-1:0]    x_rdata_ready,
    input  wire [NDEV*DW-1:0] x_rdata,
    input  wire [NDEV-1:0]    x_done
);
    localparam int STRIPE = 1024;                         // fabric.hpi.STRIPE_BYTES; a test checks they agree
    localparam int BPS    = STRIPE / (DW / 8);            // beats per stripe
    localparam int DEVW   = $clog2(NDEV) + 1;
    localparam int SHIFT  = $clog2(STRIPE);
    localparam int DSHIFT = $clog2(NDEV);
    // Issue side.
    reg          busy, write_r;
    reg [31:0]   addr;
    reg [11:0]   left;
    reg [DEVW-1:0] cur;                                    // device of the chunk being fed
    reg [7:0]    cur_beats, fed;
    reg          feeding;                                  // write data of the current chunk in flight
    // Order queue of issued read chunks: device and beats.
    reg [DEVW-1:0] q_dev [0:31];
    reg [7:0]    q_beats [0:31];
    reg [5:0]    q_wr, q_rd;
    reg [7:0]    drained;
    reg [5:0]    pending_w;                                // write chunks issued and not yet done
    wire         q_empty = (q_wr == q_rd);
    wire         q_full  = ((q_wr - q_rd) == 6'd32);
    wire [31:0]  stripe_no = addr >> SHIFT;
    wire [DEVW-1:0] dev   = stripe_no % NDEV;
    wire [24:0]  dev_word = (((stripe_no / NDEV) << SHIFT) | (addr & (STRIPE - 1))) >> 1;
    wire [11:0]  room     = (STRIPE - (addr & (STRIPE - 1))) / (DW / 8);
    // Write chunks in flight per channel; their completions can coincide.
    reg [NDEV-1:0] w_out;
    integer dc;
    reg [5:0] done_count;
    always @* begin
        done_count = 0;
        for (dc = 0; dc < NDEV; dc = dc + 1) done_count = done_count + (x_done[dc] & w_out[dc]);
    end
    wire [7:0]   chunk    = (left < room) ? left[7:0] : room[7:0];

    assign req_ready = !busy;
    // Write data of the current chunk goes to its channel.
    assign wdata_ready = feeding && x_wdata_ready[cur];
    assign x_wdata = wdata;
    genvar g;
    generate
        for (g = 0; g < NDEV; g = g + 1) begin : g_x
            assign x_wdata_valid[g] = feeding && (cur == g) && wdata_valid;
            assign x_rdata_ready[g] = !q_empty && (q_dev[q_rd[4:0]] == g);
        end
    endgenerate
    // Read data comes from the queue head's channel.
    wire [DEVW-1:0] head = q_dev[q_rd[4:0]];
    assign rdata_valid = !q_empty && x_rdata_valid[head];
    assign rdata       = x_rdata[head*DW +: DW];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0; x_valid <= 0; feeding <= 1'b0; q_wr <= 0; q_rd <= 0; drained <= 0; left <= 0; addr <= 0;
            write_r <= 1'b0; cur <= 0; cur_beats <= 0; fed <= 0; pending_w <= 0; w_out <= 0;
        end else begin
            // Accept a port request.
            if (!busy && req_valid) begin
                busy <= 1'b1; write_r <= req_write; addr <= req_addr; left <= req_beats;
            end
            // Issue the next chunk when its device is free and, for writes, the previous chunk's data is in.
            if (busy && left != 0 && !feeding && x_valid == 0 && x_ready[dev] && !q_full) begin
                x_valid[dev] <= 1'b1; x_write <= write_r; x_addr <= dev_word; x_beats <= chunk;
                cur <= dev; cur_beats <= chunk; fed <= 0;
            end
            // Write chunks outstanding: issued ones count up, done pulses count down.
            pending_w <= pending_w + ((x_valid != 0 && (x_valid & x_ready) != 0 && write_r) ? 6'd1 : 6'd0) - done_count;
            w_out <= w_out & ~x_done;
            if (x_valid != 0 && (x_valid & x_ready) != 0) begin
                x_valid <= 0;
                if (write_r) w_out[cur] <= 1'b1;
                addr <= addr + cur_beats * (DW / 8);
                left <= left - cur_beats;
                if (write_r) feeding <= 1'b1;
                else begin q_dev[q_wr[4:0]] <= cur; q_beats[q_wr[4:0]] <= cur_beats; q_wr <= q_wr + 6'd1; end
            end
            if (feeding && wdata_valid && x_wdata_ready[cur]) begin
                fed <= fed + 1'b1;
                if (fed == cur_beats - 1) feeding <= 1'b0;
            end
            // Drain reads in order.
            if (rdata_valid) begin
                drained <= drained + 1'b1;
                if (drained == q_beats[q_rd[4:0]] - 1) begin drained <= 0; q_rd <= q_rd + 6'd1; end
            end
            // The request is done when every chunk is issued and drained (reads) or written (writes).
            if (busy && left == 0 && !feeding && x_valid == 0 && (write_r ? (pending_w == 0 && done_count == 0) : q_empty)) busy <= 1'b0;
        end
    end
endmodule

`default_nettype wire
