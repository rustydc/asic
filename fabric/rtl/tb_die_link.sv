// Testbench for fabric_die_link against fabric.die.emit_die_link_vectors:
// packets go in, a stand-in engine runs each lane's runs, and the Python
// side checks what the lanes were given and what left
// (fabric.die.check_die_link_run).  The stand-in adds to a lane's vectors,
// in place, a constant naming each run's program and layer, and takes longer
// in the higher lanes, so lanes finish out of the order they started; it
// records every push and every lane's done.  The link must not touch a
// lane's vectors while it runs.

`timescale 1ns/1ps
`default_nettype none

module tb_die_link #(
    parameter int D          = 16,
    parameter int CHUNK      = 3,
    parameter int LANES      = 4,
    parameter int SLOT_PAGES = 37,
    parameter int PAGE_BASE  = 5,
    parameter int LAYERS     = 1,
    parameter int PACKETS    = 8,
    parameter int IWORDS     = 64,
    parameter int OWORDS     = 64,
    parameter int CRC_ERRORS = 0,
    parameter int MALFORMED  = 0
);
    localparam int AW = 24;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    integer cycle = 0;
    always @(posedge clk) cycle <= cycle + 1;

    reg [31:0]  im [0:IWORDS-1];
    reg [15:0]  pk [0:PACKETS-1];
    reg [255:0] tm [0:3];

    reg         u_valid = 0, u_sop = 0, d_ready = 1;
    reg [31:0]  u_data = 0;
    wire        u_ready, d_valid, d_sop;
    wire [31:0] d_data;
    wire        e_push;
    wire [1:0]  e_lane, e_layer;
    wire [15:0] e_pc, e_steps;
    wire [20:0] e_page;
    wire [3:0]  e_set, e_first;
    wire [4*21-1:0] e_slot_page;
    wire [4*32-1:0] e_position;
    reg  [3:0]  e_done = 0;
    wire        v_wr_en, v_rd_en;
    wire [AW-1:0] v_wr_addr, v_rd_addr;
    wire [127:0] v_wr_data;
    reg  [127:0] v_rd_data;
    wire [15:0] crc_errors, malformed;

    fabric_die_link #(.D(D), .CHUNK(CHUNK), .LANES(LANES), .SLOT_PAGES(SLOT_PAGES), .PAGE_BASE(PAGE_BASE),
                      .AW(AW), .LAYERS(LAYERS), .TABLE_FILE("die_table.hex"), .LAYER_FILE("die_layers.hex")) dut (
        .clk(clk), .rst_n(rst_n),
        .u_valid(u_valid), .u_data(u_data), .u_sop(u_sop), .u_ready(u_ready),
        .d_valid(d_valid), .d_data(d_data), .d_sop(d_sop), .d_ready(d_ready),
        .e_push(e_push), .e_lane(e_lane), .e_pc(e_pc), .e_steps(e_steps), .e_layer(e_layer), .e_page(e_page), .e_set(e_set),
        .e_first(e_first), .e_slot_page(e_slot_page), .e_position(e_position), .e_done(e_done),
        .v_wr_en(v_wr_en), .v_wr_addr(v_wr_addr), .v_wr_data(v_wr_data),
        .v_rd_en(v_rd_en), .v_rd_addr(v_rd_addr), .v_rd_data(v_rd_data),
        .crc_errors(crc_errors), .malformed(malformed));

    // The vector buffer: beats, answering the cycle after the address.
    reg [127:0] vb [0:4095];
    integer errors = 0, ow = 0, rec, outf;
    initial begin rec = $fopen("pushes.txt", "w"); outf = $fopen("out_words.hex", "w"); end

    // The stand-in engine: a lane's runs, then its done.
    reg [15:0] r_pc  [0:3][0:3];
    reg [1:0]  r_lay [0:3][0:3];
    integer    n_run [0:3];
    reg [3:0]  lane_on = 0;                  // pushed and not yet done
    reg [AW-1:0] lo [0:3], hi [0:3];         // a running lane's vectors
    integer i, k, t, q, e, toks;
    initial for (i = 0; i < 4; i = i + 1) n_run[i] = 0;
    function automatic integer entry_of(input [15:0] pc);
        integer m;
        begin
            entry_of = -1;
            for (m = 0; m < 4; m = m + 1) if (tm[m][15:0] == pc) entry_of = m;
        end
    endfunction
    always @(posedge clk) if (e_push) begin
        $fdisplay(rec, "P %0d %0d %h", cycle, e_lane,
                  {e_position[32*e_lane +: 32], e_slot_page[21*e_lane +: 21], e_first[e_lane], e_page, e_layer, e_steps, e_pc});
        if (e_set != (4'b0001 << e_lane)) begin errors = errors + 1; $display("a push to lane %0d sets tokens %b", e_lane, e_set); end
        if (e_lane >= LANES || lane_on[e_lane] && n_run[e_lane] == 0) begin errors = errors + 1; $display("a push to lane %0d", e_lane); end
        r_pc[e_lane][n_run[e_lane]] = e_pc; r_lay[e_lane][n_run[e_lane]] = e_layer;
        n_run[e_lane] = n_run[e_lane] + 1;
        lane_on[e_lane] = 1'b1;
        e = entry_of(e_pc);
        toks = (e & 1) ? CHUNK : 1;
        lo[e_lane] = tm[e][32 + 24*e_lane +: 24];
        hi[e_lane] = lo[e_lane] + toks * D * 2;
        if (n_run[e_lane] == LAYERS) run_lane(e_lane);
    end
    // Each lane counts down once its runs are in, longer in the higher lanes,
    // then takes its runs in place and reports done -- one lane a cycle, as
    // the engine's drain allows.
    integer cd [0:3];
    initial for (i = 0; i < 4; i = i + 1) cd[i] = -1;
    integer l, jj, b, w, ee, tk;
    reg     fired;
    reg [127:0] word;
    reg [15:0]  x;
    always @(posedge clk) begin
        e_done <= 4'b0;
        fired = 1'b0;
        for (l = 0; l < 4; l = l + 1) begin
            if (cd[l] > 0) cd[l] = cd[l] - 1;
            else if (cd[l] == 0 && !fired) begin
                for (jj = 0; jj < LAYERS; jj = jj + 1) begin
                    ee = entry_of(r_pc[l][jj]);
                    tk = (ee & 1) ? CHUNK : 1;
                    for (b = 0; b < tk * D / 8; b = b + 1) begin
                        word = vb[(tm[ee][32 + 24*l +: 24] >> 4) + b];
                        for (w = 0; w < 8; w = w + 1) begin
                            x = word[16*w +: 16] + 16'd1 + 16'(7 * ee) + 16'(3 * r_lay[l][jj]);
                            word[16*w +: 16] = x;
                        end
                        vb[(tm[ee][32 + 24*l +: 24] >> 4) + b] = word;
                    end
                end
                e_done[l] <= 1'b1;
                n_run[l] = 0; lane_on[l] = 1'b0; cd[l] = -1; fired = 1'b1;
                $fdisplay(rec, "D %0d %0d", cycle, l);
            end
        end
    end
    task run_lane(input integer ln);
        cd[ln] = 40 + 29 * ln;
    endtask

    always @(posedge clk) begin
        if (v_rd_en) v_rd_data <= vb[v_rd_addr >> 4];
        if (v_wr_en) vb[v_wr_addr >> 4] <= v_wr_data;
        for (k = 0; k < 4; k = k + 1)
            if (lane_on[k] && ((v_wr_en && v_wr_addr >= lo[k] && v_wr_addr < hi[k]) || (v_rd_en && v_rd_addr >= lo[k] && v_rd_addr < hi[k]))) begin
                errors = errors + 1;
                if (errors <= 5) $display("the link used lane %0d's vectors while it ran", k);
            end
    end

    // What leaves, for the model.
    always @(posedge clk) if (d_valid && d_ready) begin
        $fdisplay(outf, "%h", d_data);
        ow = ow + 1;
    end
    // The next die stalls now and then.
    always @(posedge clk) d_ready <= ($random & 3) != 0;

    integer p, iw, guard, j;
    initial begin
        $readmemh("in.hex", im);
        $readmemh("pk.hex", pk);
        $readmemh("die_table.hex", tm);
        repeat (2) @(posedge clk);
        #1 rst_n = 1;
        iw = 0;
        for (p = 0; p < PACKETS; p = p + 1) begin
            for (j = 0; j < pk[p]; j = j + 1) begin
                u_valid = 1; u_sop = (j == 0); u_data = im[iw + j];
                @(posedge clk);
                while (!u_ready) @(posedge clk);
                #1;
                if (j % 7 == 6) begin u_valid = 0; @(posedge clk); #1; end     // a gap the link must wait out
            end
            u_valid = 0; u_sop = 0;
            iw = iw + pk[p];
            if (p % 5 == 4) repeat (30) @(posedge clk);                         // and a quiet now and then
            #1;
        end
        guard = 0;
        while (ow < OWORDS && guard < 200000) begin @(posedge clk); guard = guard + 1; end
        repeat (200) @(posedge clk);
        $fclose(rec); $fclose(outf);
        if (ow != OWORDS) begin errors = errors + 1; $display("%0d out words, expected %0d; lanes %0d%0d%0d%0d", ow, OWORDS,
                                                              dut.l_st[0], dut.l_st[1], dut.l_st[2], dut.l_st[3]); end
        if (crc_errors != CRC_ERRORS || malformed != MALFORMED) begin
            errors = errors + 1;
            $display("dropped %0d for the CRC and %0d malformed, expected %0d and %0d", crc_errors, malformed, CRC_ERRORS, MALFORMED);
        end
        if (errors == 0) $display("PASS: %0d packets, %0d words out", PACKETS, OWORDS);
        else $display("FAIL: %0d errors", errors);
        $finish;
    end
endmodule

`default_nettype wire
