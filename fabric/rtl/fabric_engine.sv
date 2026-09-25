// The layer engine: the token sequencer (fabric_sequencer.sv) driving the
// real units through one byte-addressed vector buffer.
//
// Every unit sits behind an adapter with the same face to the controller:
// a command (length, four 30-bit address operands src, dst, a2 and a3, a
// 32-bit argument, a tag) accepted when the engine is idle, and a done pulse
// carrying the tag one cycle after the step's last result was written to
// the vector buffer.  The adapters read the buffer 16 bytes per port per
// cycle (one cycle of latency) and write 16 bytes with byte enables; the
// memory unit's ports are 32 bytes wide, for the beat mover, whose moves are
// two beats a transfer on the memory port as well.
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
//   fabric_rotary_adapter  the rotary table, and a head's norm, rotation
//                          and requantization
//   fabric_attn_adapter    one attention core over a head's rows
//   fabric_mem_unit        the memory port: beats to and from the buffer,
//                          the token's append, the index scan and the
//                          record reader, behind one arbiter
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
    parameter int NB    = 1,       // banks
    parameter int BSH   = 12,      // a bank's own address bits: bank b holds [b << BSH, (b+1) << BSH)
    parameter [63:0] RCAP2 = 0,    // banks that need a second read port
    parameter [63:0] RCAP3 = 0,    // and a third
    parameter [63:0] WCAP2 = 0,    // banks a pair of engines contribute to at once
    // The crossbar each logical port folds onto.  The port face is sized by
    // construction -- every adapter operand has one -- but the controller
    // issues in order, so most of them can never be asking at once, and
    // `fabric.engine.port_colours` says which may share.  Four bits a port,
    // sixteen ports to a word.  NPR = NR and the identity map is the face
    // itself, which is what the checks below are written against.
    parameter int NPR   = NR,
    parameter int NPW   = NW,
    parameter [63:0] RMAP0 = 64'hFEDCBA9876543210,
    parameter [63:0] RMAP1 = 64'hFEDCBA9876543210,
    parameter [63:0] WMAP0 = 64'hFEDCBA9876543210,
    parameter [63:0] WMAP1 = 64'hFEDCBA9876543210,
    // One read port and one write port may be 32 bytes wide: the memory
    // unit's, for the beat mover.  Their second sixteen bytes are rd_hi and
    // wr_hi, the word after the one the port names, and a wide access names
    // a word boundary.  A bank's port already reads and writes a 32-byte
    // window -- a 16-byte access at any byte offset is a word from each of
    // its two memories -- so a wide access is that window whole, one word
    // from each memory, and costs the banks nothing.  -1 is none.
    parameter int WIDE_R = -1,
    parameter int WIDE_W = -1,
    parameter     INIT_FILE = ""
) (
    input  wire              clk,
    input  wire [NR-1:0]     rd_en,
    input  wire [NR*AW-1:0]  rd_addr,
    output reg  [NR*128-1:0] rd_data,
    output reg  [127:0]      rd_hi,
    input  wire [NW-1:0]     wr_en,
    input  wire [NW*AW-1:0]  wr_addr,
    input  wire [NW*128-1:0] wr_data,
    input  wire [NW*16-1:0]  wr_be,
    input  wire [127:0]      wr_hi,
    input  wire [15:0]       wr_hi_be
);
    // Banks of SRAM.  A buffer lives wholly in one, and a buffer's bank is the
    // high bits of its address, so an adapter's byte address carries it and no
    // port needs a bank field.  `fabric.engine.Layout` colours the buffers so
    // that no bank is asked for more ports than it has; the counts below are
    // that colouring put to the engine rather than taken on trust.
    //
    // A bank is two memories, not one.  The adapters read sixteen bytes at any
    // byte address -- the norm and SwiGLU step eight bytes a beat, the conv and
    // the gates four, the pass adapter two -- so a read straddles two words of
    // a sixteen-byte memory.  Holding the even words in one memory and the odd
    // in the other makes that one word from each, whatever the alignment, and
    // costs no port: of any two adjacent words exactly one is even.
    localparam int RMAX = 3, WMAX = 2;
    // A bank's slots are flattened at a power-of-two stride, not at RMAX, so
    // that "bank b's slot s" is a shift and an or rather than a multiply.  At
    // RMAX = 3 the padding is one unused slot a bank, which nothing drives and
    // synthesis drops; what it buys is that none of the two hundred-odd index
    // expressions in this module becomes a 32-bit multiplier -- yosys's
    // resource sharing spends about a quarter of a minute on each of those, and
    // the module did not finish synthesis at all until they went away.
    localparam int RSH  = (RMAX > 1) ? $clog2(RMAX) : 0;
    localparam int RPOT = 1 << RSH;
    localparam int WSH  = (WMAX > 1) ? $clog2(WMAX) : 0;
    localparam int WPOT = 1 << WSH;
    localparam int SB   = (NB*RPOT > 1) ? $clog2(NB*RPOT) : 1;
    localparam int WPB  = (1 << BSH) / 16;              // sixteen-byte words in a bank
    localparam int HALF = (WPB > 1) ? WPB / 2 : 1;      // words in each of its halves
    localparam int HW   = (HALF > 1) ? $clog2(HALF) : 1;
    localparam int WIB  = (WPB > 1) ? $clog2(WPB) : 1;
    localparam int BB   = (AW > BSH) ? AW - BSH : 1;

    function automatic integer bank_of(input [AW-1:0] a);
        bank_of = (AW > BSH) ? (a >> BSH) : 0;
    endfunction
    function automatic integer cap_of(input integer bb);
        cap_of = 1 + (RCAP2[bb] ? 1 : 0) + (RCAP3[bb] ? 1 : 0);
    endfunction
    function automatic integer wcap_of(input integer bb);
        wcap_of = 1 + (WCAP2[bb] ? 1 : 0);
    endfunction
    function automatic integer rmap_of(input integer i);
        rmap_of = (i < 16) ? RMAP0[i*4 +: 4] : RMAP1[(i-16)*4 +: 4];
    endfunction
    function automatic integer wmap_of(input integer i);
        wmap_of = (i < 16) ? WMAP0[i*4 +: 4] : WMAP1[(i-16)*4 +: 4];
    endfunction

    // The fold: the logical ports' addresses or'd onto the crossbar port they
    // share.  Sharers are never enabled together, so an or is a mux, and the
    // answer goes back on one set of wires that all of them read.  Everything
    // below -- the allocator, the banks' slots, the beat that comes back -- is
    // then NPR and NPW wide rather than NR and NW.
    reg [NPR-1:0]     p_en;
    reg [NPR*AW-1:0]  p_addr;
    reg [NPW-1:0]     q_en;
    reg [NPW*AW-1:0]  q_addr;
    reg [NPW*128-1:0] q_data, q_hi;
    reg [NPW*16-1:0]  q_be, q_hibe;
    integer fi;
    always @(*) begin
        p_en = 0; p_addr = 0;
        for (fi = 0; fi < NR; fi = fi + 1)
            if (rd_en[fi]) begin
                p_en[rmap_of(fi)] = 1'b1;
                p_addr[rmap_of(fi)*AW +: AW] = p_addr[rmap_of(fi)*AW +: AW] | rd_addr[fi*AW +: AW];
            end
        q_en = 0; q_addr = 0; q_data = 0; q_be = 0; q_hi = 0; q_hibe = 0;
        for (fi = 0; fi < NW; fi = fi + 1)
            if (wr_en[fi]) begin
                q_en[wmap_of(fi)] = 1'b1;
                q_addr[wmap_of(fi)*AW +: AW]   = q_addr[wmap_of(fi)*AW +: AW]   | wr_addr[fi*AW +: AW];
                q_data[wmap_of(fi)*128 +: 128] = q_data[wmap_of(fi)*128 +: 128] | wr_data[fi*128 +: 128];
                q_be[wmap_of(fi)*16 +: 16]     = q_be[wmap_of(fi)*16 +: 16]     | wr_be[fi*16 +: 16];
                if (fi == WIDE_W) begin
                    q_hi[wmap_of(fi)*128 +: 128] = q_hi[wmap_of(fi)*128 +: 128] | wr_hi;
                    q_hibe[wmap_of(fi)*16 +: 16] = q_hibe[wmap_of(fi)*16 +: 16] | wr_hi_be;
                end
            end
    end
`ifndef FABRIC_SYNTH
    integer fj;
    always @(posedge clk) begin
        for (fi = 0; fi < NR; fi = fi + 1)
            for (fj = 0; fj < NR; fj = fj + 1)
                if (fj < fi && rd_en[fi] && rd_en[fj] && rmap_of(fi) == rmap_of(fj))
                    $display("FAIL: read ports %0d and %0d share crossbar port %0d and are both asking",
                             fj, fi, rmap_of(fi));
        for (fi = 0; fi < NW; fi = fi + 1)
            for (fj = 0; fj < NW; fj = fj + 1)
                if (fj < fi && wr_en[fi] && wr_en[fj] && wmap_of(fi) == wmap_of(fj))
                    $display("FAIL: write ports %0d and %0d share crossbar port %0d and are both asking",
                             fj, fi, wmap_of(fi));
        if (WIDE_W >= 0 && wr_en[WIDE_W] && |wr_hi_be && wr_addr[WIDE_W*AW +: 4] != 4'd0)
            $display("FAIL: a wide write at byte %0d, not a word boundary", wr_addr[WIDE_W*AW +: AW]);
    end
`endif

    // Which of a bank's ports each access takes.  Reads on one address share a
    // port, which is what lets two units stream the same beat of one buffer.
    integer      rn [0:NB-1];
    reg [AW-1:0] ra [0:NB*RPOT-1];
    reg          rv [0:NB*RPOT-1];
    reg [1:0]    r_slot [0:NPR-1];
    reg [BB-1:0] r_bank [0:NPR-1];
    reg          r_got [0:NPR-1];
    integer      wn [0:NB-1];
    reg [AW-1:0] wa [0:NB*WPOT-1];
    reg [127:0]  wd [0:NB*WPOT-1], wdh [0:NB*WPOT-1];
    reg [15:0]   wm [0:NB*WPOT-1], wmh [0:NB*WPOT-1];
    reg          wv [0:NB*WPOT-1];
    // Every port's slot, decided in parallel rather than by a scan.  Written as
    // a walk over the ports carrying rn[b] from one to the next, the allocator
    // is an NR-deep chain of compare-and-update per bank, which is where this
    // module's 7.9 ns went -- the shape the sequencer's drain had.  The same
    // decision made all at once is: which ports name an address first in their
    // bank, how many such ports come before each of them, and for a port that
    // is not first, the slot of the one it matched.  All three are compares
    // and counts over the ports, of logarithmic depth.
    reg [NPR-1:0]  r_use, r_first;
    reg [1:0]      r_pre [0:NPR-1];                      // slot if first: distinct addresses before it
    reg [1:0]      r_ix  [0:NPR-1];                      // the slot it actually takes
    reg [NPW-1:0]  w_use;
    reg [1:0]      w_pre [0:NPW-1];
    reg [BB-1:0]   w_bank [0:NPW-1];
    integer i, j, b, s, n;
`ifndef FABRIC_SYNTH
    integer      max_rd [0:NB-1], max_wr [0:NB-1];       // over the run, for the report
    initial for (i = 0; i < NB; i = i + 1) begin max_rd[i] = 0; max_wr[i] = 0; end
`endif

    always @(*) begin
        // Reads: which ports are asking, and which of them names its address
        // first in its bank.  A later port with the same address shares that
        // port's slot, which is what lets two units stream one buffer.
        for (i = 0; i < NPR; i = i + 1) begin
            r_bank[i] = bank_of(p_addr[i*AW +: AW]);
            r_use[i]  = p_en[i] && (^p_addr[i*AW +: AW] !== 1'bx);
        end
        for (i = 0; i < NPR; i = i + 1) begin
            r_first[i] = r_use[i];
            for (j = 0; j < NPR; j = j + 1)
                if (j < i && r_use[j] && r_bank[j] == r_bank[i]
                    && p_addr[j*AW +: AW] == p_addr[i*AW +: AW]) r_first[i] = 1'b0;
        end
        for (i = 0; i < NPR; i = i + 1) begin
            n = 0;                                        // distinct addresses of this bank before i
            for (j = 0; j < NPR; j = j + 1)
                if (j < i && r_first[j] && r_bank[j] == r_bank[i] && n < RMAX) n = n + 1;
            r_pre[i] = n[1:0];
        end
        for (i = 0; i < NPR; i = i + 1) begin
            r_ix[i] = r_pre[i];                           // and a sharer takes the slot it matched
            if (!r_first[i])
                for (j = 0; j < NPR; j = j + 1)
                    if (j < i && r_first[j] && r_bank[j] == r_bank[i]
                        && p_addr[j*AW +: AW] == p_addr[i*AW +: AW]) r_ix[i] = r_pre[j];
            r_slot[i] = r_ix[i];
            r_got[i]  = r_use[i] && (r_ix[i] < RMAX);
        end
        for (b = 0; b < NB; b = b + 1) begin
            rn[b] = 0;
            for (i = 0; i < NPR; i = i + 1)
                if (r_first[i] && r_bank[i] == b) rn[b] = rn[b] + 1;
            for (s = 0; s < RPOT; s = s + 1) begin
                ra[(b << RSH) + s] = 0;
                rv[(b << RSH) + s] = 1'b0;
                for (i = 0; i < NPR; i = i + 1)            // one-hot: one port owns a slot
                    if (r_first[i] && r_bank[i] == b && r_pre[i] == s && s < RMAX) begin
                        ra[(b << RSH) + s] = ra[(b << RSH) + s] | p_addr[i*AW +: AW];
                        rv[(b << RSH) + s] = 1'b1;
                    end
            end
        end
        // Writes: no sharing, so a port's slot is how many of its bank come
        // before it.  A bank asked for more than it has is still counted, so
        // the check below sees it.
        for (i = 0; i < NPW; i = i + 1) begin
            w_bank[i] = bank_of(q_addr[i*AW +: AW]);
            w_use[i]  = q_en[i];
        end
        for (i = 0; i < NPW; i = i + 1) begin
            n = 0;
            for (j = 0; j < NPW; j = j + 1)
                if (j < i && w_use[j] && w_bank[j] == w_bank[i] && n < WMAX) n = n + 1;
            w_pre[i] = n[1:0];
        end
        for (b = 0; b < NB; b = b + 1) begin
            wn[b] = 0;
            for (i = 0; i < NPW; i = i + 1)
                if (w_use[i] && w_bank[i] == b) wn[b] = wn[b] + 1;
            for (s = 0; s < WPOT; s = s + 1) begin
                wa[(b << WSH) + s] = 0;
                wd[(b << WSH) + s] = 0;
                wm[(b << WSH) + s] = 0;
                wdh[(b << WSH) + s] = 0;
                wmh[(b << WSH) + s] = 0;
                wv[(b << WSH) + s] = 1'b0;
                for (i = 0; i < NPW; i = i + 1)
                    if (w_use[i] && w_bank[i] == b && w_pre[i] == s && s < WMAX) begin
                        wa[(b << WSH) + s] = wa[(b << WSH) + s] | q_addr[i*AW +: AW];
                        wd[(b << WSH) + s] = wd[(b << WSH) + s] | q_data[i*128 +: 128];
                        wm[(b << WSH) + s] = wm[(b << WSH) + s] | q_be[i*16 +: 16];
                        wdh[(b << WSH) + s] = wdh[(b << WSH) + s] | q_hi[i*128 +: 128];
                        wmh[(b << WSH) + s] = wmh[(b << WSH) + s] | q_hibe[i*16 +: 16];
                        wv[(b << WSH) + s] = 1'b1;
                    end
            end
        end
    end

`ifndef FABRIC_SYNTH
    // What the run asked of each bank, against what it has.
    always @(posedge clk) begin
        for (b = 0; b < NB; b = b + 1) begin
            if (rn[b] > max_rd[b]) max_rd[b] = rn[b];
            if (wn[b] > max_wr[b]) max_wr[b] = wn[b];
            if (rn[b] > cap_of(b))  $display("FAIL: bank %0d asked for %0d reads, it has %0d", b, rn[b], cap_of(b));
            if (wn[b] > wcap_of(b)) $display("FAIL: bank %0d asked for %0d writes, it has %0d", b, wn[b], wcap_of(b));
        end
        for (i = 0; i < NPR; i = i + 1)
            if (p_en[i] && !r_got[i]) begin
                if (^p_addr[i*AW +: AW] === 1'bx) $display("FAIL: crossbar read port %0d is enabled on an unknown address", i);
                else $display("FAIL: crossbar read port %0d got no port of bank %0d", i, r_bank[i]);
            end
    end
`endif

    wire [NB*RPOT*128-1:0] even_q, odd_q;
    genvar gb, gs;
    generate
        for (gb = 0; gb < NB; gb = gb + 1) begin : g_bank
            localparam int NRD = 1 + (RCAP2[gb] ? 1 : 0) + (RCAP3[gb] ? 1 : 0);
            localparam int NWR = 1 + (WCAP2[gb] ? 1 : 0);
            wire [NRD-1:0]     r_en;
            wire [NRD*HW-1:0]  e_ra, o_ra;
            wire [NRD*128-1:0] e_rd, o_rd;
            wire [NWR-1:0]     e_we, o_we;
            wire [NWR*HW-1:0]  e_wa, o_wa;
            wire [NWR*128-1:0] e_wd, o_wd;
            wire [NWR*16-1:0]  e_wm, o_wm;
            for (gs = 0; gs < NRD; gs = gs + 1) begin : g_rd
                wire [WIB-1:0] w = ra[gb*RPOT + gs][BSH-1:4];
                assign r_en[gs] = rv[gb*RPOT + gs];
                assign e_ra[gs*HW +: HW] = ((w + {{(WIB-1){1'b0}}, w[0]}) >> 1);
                assign o_ra[gs*HW +: HW] = (w >> 1);
                assign even_q[(gb*RPOT + gs)*128 +: 128] = e_rd[gs*128 +: 128];
                assign odd_q[(gb*RPOT + gs)*128 +: 128]  = o_rd[gs*128 +: 128];
            end
            for (gs = NRD; gs < RPOT; gs = gs + 1) begin : g_rd_none
                assign even_q[(gb*RPOT + gs)*128 +: 128] = {128{1'bx}};
                assign odd_q[(gb*RPOT + gs)*128 +: 128]  = {128{1'bx}};
            end
            for (gs = 0; gs < NWR; gs = gs + 1) begin : g_wr
                wire [WIB-1:0] w   = wa[gb*WPOT + gs][BSH-1:4];
                wire [3:0]     off = wa[gb*WPOT + gs][3:0];
                wire [255:0]   d32 = {wdh[gb*WPOT + gs], wd[gb*WPOT + gs]} << {off, 3'd0};
                wire [31:0]    m32 = {wmh[gb*WPOT + gs], wm[gb*WPOT + gs]} << off;
                // The window's low word is the one at w, the high word the next:
                // whichever of the two is even goes to the even memory.
                assign e_we[gs]            = wv[gb*WPOT + gs] && |(w[0] ? m32[31:16] : m32[15:0]);
                assign e_wa[gs*HW +: HW]   = ((w + {{(WIB-1){1'b0}}, w[0]}) >> 1);
                assign e_wd[gs*128 +: 128] = w[0] ? d32[255:128] : d32[127:0];
                assign e_wm[gs*16 +: 16]   = w[0] ? m32[31:16]   : m32[15:0];
                assign o_we[gs]            = wv[gb*WPOT + gs] && |(w[0] ? m32[15:0] : m32[31:16]);
                assign o_wa[gs*HW +: HW]   = (w >> 1);
                assign o_wd[gs*128 +: 128] = w[0] ? d32[127:0] : d32[255:128];
                assign o_wm[gs*16 +: 16]   = w[0] ? m32[15:0]  : m32[31:16];
            end
            fabric_sram #(.W(128), .D(HALF), .NRD(NRD), .NWR(NWR), .MB(8)) u_even (
                .clk(clk), .rd_en(r_en), .rd_addr(e_ra), .rd_data(e_rd),
                .wr_en(e_we), .wr_addr(e_wa), .wr_data(e_wd), .wr_mask(e_wm));
            fabric_sram #(.W(128), .D(HALF), .NRD(NRD), .NWR(NWR), .MB(8)) u_odd (
                .clk(clk), .rd_en(r_en), .rd_addr(o_ra), .rd_data(o_rd),
                .wr_en(o_we), .wr_addr(o_wa), .wr_data(o_wd), .wr_mask(o_wm));
        end
    endgenerate

    // The beat a port asked for, a cycle later: its two words in address order,
    // shifted down to the byte it started at.
    // A port's answer is a mux over every bank's slots, then a shift down to
    // the byte it started at, and both are selected by a handful of bits
    // against the whole 256-bit window.  Held once a port, the bank bit drove
    // 1,793 loads and 7.14 of this module's 7.88 nanoseconds: it is a
    // register at the edge of the combinational cone, so the mapper buffers
    // the mux but not what drives it.  Each byte of the answer therefore
    // keeps its own copy of what selects it, which is what the norm, the
    // rotary and the state engine do with their constants.  Both muxes are
    // written out a byte at a time rather than as a part-select of the whole
    // bank vector: a variable part-select of all NB*RPOT words is a barrel
    // shifter over every bank, and sixteen of those a port does not map.
    reg [NPR*128-1:0] p_data, p_hi;
    genvar gp, gg, gk;
    generate
        for (gp = 0; gp < NPR; gp = gp + 1) begin : g_read
            wire [SB-1:0] sel_w = (r_bank[gp] << RSH) + r_slot[gp];   // a concatenation: see RSH
            wire [3:0]    off_w = p_addr[gp*AW +: 4];
            wire          odd_w = p_addr[gp*AW + 4];
            wire          en_w  = p_en[gp] && r_got[gp];
            wire [255:0]  win;                          // the two words in address order
            for (gg = 0; gg < 16; gg = gg + 1) begin : g_byte
                wire [SB-1:0] sel_l;
                wire          odd_l;
                fabric_const_copy #(.W(SB)) u_sel (.clk(clk), .d(sel_w), .q(sel_l));
                fabric_const_copy #(.W(1))  u_odd (.clk(clk), .d(odd_w), .q(odd_l));
                wire [NB*RPOT*8-1:0] e_b, o_b;          // this byte of every slot of every bank
                for (gk = 0; gk < NB*RPOT; gk = gk + 1) begin : g_word
                    assign e_b[gk*8 +: 8] = even_q[gk*128 + gg*8 +: 8];
                    assign o_b[gk*8 +: 8] = odd_q [gk*128 + gg*8 +: 8];
                end
                wire [7:0] e = e_b[{sel_l, 3'd0} +: 8];
                wire [7:0] o = o_b[{sel_l, 3'd0} +: 8];
                assign win[gg*8 +: 8]       = odd_l ? o : e;
                assign win[128 + gg*8 +: 8] = odd_l ? e : o;
            end
            for (gg = 0; gg < 16; gg = gg + 1) begin : g_out
                wire [3:0] off_l;
                wire       en_l;
                fabric_const_copy #(.W(4)) u_off (.clk(clk), .d(off_w), .q(off_l));
                fabric_const_copy #(.W(1)) u_en  (.clk(clk), .d(en_w),  .q(en_l));
                wire [127:0] pick;                      // the sixteen bytes this one could come from
                for (gk = 0; gk < 16; gk = gk + 1) begin : g_pick
                    assign pick[gk*8 +: 8] = win[(gg + gk)*8 +: 8];
                end
                always @(*) p_data[gp*128 + gg*8 +: 8] = en_l ? pick[{off_l, 3'd0} +: 8] : 8'bx;
                // A wide read's second word: the window's high half, which is
                // the next word when the read names a word boundary.
                always @(*) p_hi[gp*128 + gg*8 +: 8] = en_l ? win[128 + gg*8 +: 8] : 8'bx;
            end
        end
    endgenerate

    // Every logical port that folded onto a crossbar port reads its answer:
    // wires, since at most one of them asked for it.
    always @(*) begin
        for (j = 0; j < NR; j = j + 1)
            rd_data[j*128 +: 128] = p_data[rmap_of(j)*128 +: 128];
        rd_hi = (WIDE_R >= 0) ? p_hi[rmap_of(WIDE_R)*128 +: 128] : 128'bx;
    end

`ifndef FABRIC_SYNTH
    // The image in and out.  The memories hold it in words, the file in bytes,
    // and the copy runs inside each bank because a generate instance cannot be
    // reached by a variable index.
    reg [7:0] mem [0:BYTES-1];
    reg       dumping = 1'b0;
    initial if (INIT_FILE != "") $readmemh(INIT_FILE, mem);
    generate
        for (gb = 0; gb < NB; gb = gb + 1) begin : g_img
            integer wi, k;
            initial begin
                #1;
                for (wi = 0; wi < HALF; wi = wi + 1)
                    for (k = 0; k < 16; k = k + 1) begin
                        g_bank[gb].u_even.mem[wi][k*8 +: 8] = (INIT_FILE != "") ? mem[gb*(1 << BSH) + (2*wi)*16 + k] : 8'd0;
                        g_bank[gb].u_odd.mem[wi][k*8 +: 8]  = (INIT_FILE != "") ? mem[gb*(1 << BSH) + (2*wi+1)*16 + k] : 8'd0;
                    end
            end
            always @(posedge dumping)
                for (wi = 0; wi < HALF; wi = wi + 1)
                    for (k = 0; k < 16; k = k + 1) begin
                        mem[gb*(1 << BSH) + (2*wi)*16 + k]   = g_bank[gb].u_even.mem[wi][k*8 +: 8];
                        mem[gb*(1 << BSH) + (2*wi+1)*16 + k] = g_bank[gb].u_odd.mem[wi][k*8 +: 8];
                    end
        end
    endgenerate
`endif
endmodule

// ---------------------------------------------------------------------------
// Norm engine.  arg[7:0] selects the constants (mult, shift, eps and the
// gain's requantizer), arg[8] int16 input, arg[9] the gain is silu of the
// int8 vector at a2; len is the beat count.
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
    input  wire [29:0]   cmd_src,
    input  wire [29:0]   cmd_dst,
    input  wire [29:0]   cmd_a2,
    input  wire [29:0]   cmd_a3,
    input  wire [31:0]   cmd_arg,
    input  wire [7:0]    cmd_tag,
    output wire          cmd_ready,
    output reg           done_valid,
    output reg  [7:0]    done_tag,
    output wire [AW-1:0] rd_addr_x,
    output wire [AW-1:0] rd_addr_g,
    output wire          rd_en_g,        // the gain is a vector only for a gated norm; else the port is idle
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
    assign rd_en_g   = busy && gated;

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
                busy <= 1'b1; issuing <= 1'b1; n <= cmd_len[BW-1:0]; src <= cmd_src[AW-1:0]; dst <= cmd_dst[AW-1:0];
                gaddr <= cmd_a2[AW-1:0]; int16 <= cmd_arg[8]; gated <= cmd_arg[9]; k <= consts[cmd_arg[7:0]];
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
//   [3:0] pass  [7:4] row block  [19:8] the tile whose partial sums it
//   continues (FFF: none)  [20] last row block  [21] raw: write the
//   accumulators as 32-bit words  [29:22] bytes to write  [53:30] byte
//   offset of its output inside the destination.
// A command runs pass arg[7:0] over arg[15:8] row blocks for the
// arg[23:16] tokens of a chunk (one at least, TMAX at most): each row
// block's tiles start together and consume ROWS activations of every
// token, P per cycle, token t's from src + t * a2 + rb * ROWS; when the
// last block's requantizer walk ends every last-block tile's output for
// token t goes to dst + t * a3 + offset.
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
    parameter int TMAX = 1,
    parameter int MODEL_TILES = 0,              // the behavioural columns, for full-size runs
    parameter int AW   = 16
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          cmd_valid,
    input  wire [15:0]   cmd_len,
    input  wire [29:0]   cmd_src,
    input  wire [29:0]   cmd_dst,
    input  wire [29:0]   cmd_a2,
    input  wire [29:0]   cmd_a3,
    input  wire [31:0]   cmd_arg,
    input  wire [7:0]    cmd_tag,
    output wire          cmd_ready,
    output reg           done_valid,
    output reg  [7:0]    done_tag,
    output wire [TMAX*AW-1:0] rd_addr,      // one port per token of the chunk
    input  wire [TMAX*128-1:0] rd_data,
    output reg           wr_en,
    output reg  [AW-1:0] wr_addr,
    output reg  [127:0]  wr_data,
    output reg  [15:0]   wr_be
);
    localparam int CYC = ROWS / P;
    localparam int PW  = TMAX * COLS * ACC;  // a tile's accumulators, token-major
    localparam int QW  = TMAX * COLS * AB;
    localparam int VW  = COLS * 32;          // one token's output vector as raw words
    reg [55:0]   tab [0:NT-1];
    reg [SB-1:0] mult_all [0:NT*COLS-1];
    reg [SHB-1:0] shift_all [0:NT*COLS-1];
    initial begin
        $readmemh("passes.hex", tab);
        $readmemh("tiles_mult.hex", mult_all);
        $readmemh("tiles_shift.hex", shift_all);
    end

    localparam [2:0] S_IDLE = 0, S_START = 1, S_STREAM = 2, S_WAIT = 3, S_WRITE = 4, S_DONE = 5;
    reg [2:0]    state;
    reg [7:0]    pass, nrb, rb, tag, ntok, tok;
    reg [11:0]   t;
    reg [AW-1:0] src, dst, in_stride, out_stride;
    reg [15:0]   i;
    reg [3:0]    j;
    reg [NT-1:0] sel;
    reg          xv;
    assign cmd_ready = (state == S_IDLE);
    wire [TMAX*P*AB-1:0] x_data;
    genvar gk;
    generate
        for (gk = 0; gk < TMAX; gk = gk + 1) begin : g_port
            assign rd_addr[gk*AW +: AW] = src + gk * in_stride + rb * ROWS + i * P;
            assign x_data[gk*P*AB +: P*AB] = rd_data[gk*128 +: P*AB];
        end
    endgenerate

    wire [NT-1:0]    q_valid_t, start_t;
    // One net per tile, not one flat vector of NT slices: a flat vector driven
    // in NT pieces is a chain of concatenations in Icarus, and every slice
    // update at the end of a pass would re-propagate the whole vector through
    // it, which at 834 tiles is hours of simulation for one pass.
    wire [PW-1:0]    psum_out_t [0:NT-1];
    wire [QW-1:0]    q_out_t    [0:NT-1];
    genvar gt, gc;
    generate
        for (gt = 0; gt < NT; gt = gt + 1) begin : g_tile
            wire [COLS*SB-1:0]  mult_w;
            wire [COLS*SHB-1:0] shift_w;
            for (gc = 0; gc < COLS; gc = gc + 1) begin : g_k
                assign mult_w[gc*SB +: SB]   = mult_all[gt*COLS + gc];
                assign shift_w[gc*SHB +: SHB] = shift_all[gt*COLS + gc];
            end
            wire [11:0]   chain   = tab[gt][19:8];
            wire [PW-1:0] psum_in = (chain == 12'hFFF) ? {PW{1'b0}} : psum_out_t[chain];
            assign start_t[gt] = (state == S_START) && (tab[gt][3:0] == pass[3:0]) && (tab[gt][7:4] == rb[3:0]);
            fabric_tile #(.ROWS(ROWS), .COLS(COLS), .WB(WB), .AB(AB), .P(P), .ACC(ACC), .SB(SB), .SHB(SHB), .T(TMAX), .MODEL(MODEL_TILES), .ROM_FILE("")) u_tile (
                .clk(clk), .rst_n(rst_n), .start(start_t[gt]), .psum_in(psum_in), .x_valid(xv), .x_data(x_data),
                .mult(mult_w), .shift(shift_w), .x_ready(), .done(), .psum_out(psum_out_t[gt]),
                .q_out(q_out_t[gt]), .q_valid(q_valid_t[gt]));
            initial $readmemh($sformatf("tile_%0d.hex", gt), u_tile.rom.rom);
        end
    endgenerate

    // The output vector of tile t for token tok: its requantized bytes, or its accumulators as words.
    wire [55:0]   cur     = tab[t];
    wire          cur_hit = (cur[3:0] == pass[3:0]) && cur[20];
    wire [7:0]    nbytes  = cur[29:22];
    wire [4:0]    beats   = (nbytes + 15) / 16;
    reg  [VW-1:0] vec;
    wire [PW-1:0] psum_cur = psum_out_t[t];
    wire [QW-1:0] q_cur    = q_out_t[t];
    integer c;
    always @* begin
        vec = {VW{1'b0}};
        if (cur[21]) begin
            for (c = 0; c < COLS; c = c + 1)
                vec[c*32 +: 32] = {{(32-ACC){psum_cur[(tok*COLS + c)*ACC + ACC - 1]}}, psum_cur[(tok*COLS + c)*ACC +: ACC]};
        end else vec[COLS*AB-1:0] = q_cur[tok*COLS*AB +: COLS*AB];
    end
    wire [8:0]  remaining = nbytes - j * 16;
    wire [15:0] be_w      = (remaining >= 16) ? 16'hFFFF : ((16'd1 << remaining[3:0]) - 1'b1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; done_valid <= 1'b0; wr_en <= 1'b0; xv <= 1'b0; sel <= 0; i <= 0; j <= 0; t <= 0; rb <= 0; tok <= 0;
        end else begin
            done_valid <= 1'b0; wr_en <= 1'b0; xv <= 1'b0;
            case (state)
                S_IDLE: if (cmd_valid) begin
                    pass <= cmd_arg[7:0]; nrb <= cmd_arg[15:8]; ntok <= (cmd_arg[23:16] == 0) ? 8'd1 : cmd_arg[23:16];
                    src <= cmd_src[AW-1:0]; dst <= cmd_dst[AW-1:0]; in_stride <= cmd_a2[AW-1:0]; out_stride <= cmd_a3[AW-1:0]; tag <= cmd_tag;
                    rb <= 0; tok <= 0; state <= S_START;
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
                        wr_en <= 1'b1; wr_addr <= dst + tok * out_stride + cur[53:30] + j * 16; wr_data <= vec[j*128 +: 128]; wr_be <= be_w;
                        if (j == beats - 1) begin
                            j <= 0;
                            if (tok == ntok - 1) begin tok <= 0; t <= t + 1'b1; if (t == NT - 1) state <= S_DONE; end
                            else tok <= tok + 1'b1;
                        end else j <= j + 1'b1;
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
// The causal convolution.  src: the new int8 samples; a2: the history (4
// bytes per channel, oldest first); dst: the int8 outputs; a3: the shifted
// history; len: beats of CL channels.
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
    input  wire [29:0]   cmd_src,
    input  wire [29:0]   cmd_dst,
    input  wire [29:0]   cmd_a2,
    input  wire [29:0]   cmd_a3,
    input  wire [31:0]   cmd_arg,
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
                busy <= 1'b1; issuing <= 1'b1; n <= cmd_len; src <= cmd_src[AW-1:0]; dst <= cmd_dst[AW-1:0]; hist <= cmd_a2[AW-1:0];
                hnext <= cmd_a3[AW-1:0]; tag <= cmd_tag; i <= 0; o <= 0;
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
// The head gates.  src: the b accumulators as 32-bit words, a2: the a
// accumulators, dst: 4 bytes per head (decay, beta), len: heads.
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
    input  wire [29:0]   cmd_src,
    input  wire [29:0]   cmd_dst,
    input  wire [29:0]   cmd_a2,
    input  wire [29:0]   cmd_a3,
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
                busy <= 1'b1; issuing <= 1'b1; n <= cmd_len; src <= cmd_src[AW-1:0]; dst <= cmd_dst[AW-1:0]; a_addr <= cmd_a2[AW-1:0];
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
// One int8 state engine.  src: the head's unit q then unit k (K bytes
// each), a2: v (V bytes), arg: the head's gates word, a3: the state slot
// (a header beat of g, e, peak, nsat then K rows of V bytes), dst: y (V
// int16).  The rows stream through fabric_delta_state8 and back into the
// slot; y and the new header are written last.
// ---------------------------------------------------------------------------
module fabric_delta_adapter #(
    parameter int K   = 16,
    parameter int V   = 16,
    // The state rows arrive a beat at a time, so the engine below needs no
    // more arithmetic lanes than a beat carries.
    parameter int VL  = (V < 16) ? V : 16,
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
    input  wire [29:0]   cmd_src,
    input  wire [29:0]   cmd_dst,
    input  wire [29:0]   cmd_a2,
    input  wire [29:0]   cmd_a3,
    input  wire [31:0]   cmd_arg,
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
    fabric_delta_state8 #(.K(K), .V(V), .VL(VL), .YSH(YSH), .E_MIN(E_MIN), .E_MAX(E_MAX), .PEAK_GROW(PEAK_GROW), .SAT_SHIFT(SAT_SHIFT)) u_delta (
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
                    q_addr <= cmd_src[AW-1:0]; k_addr <= cmd_src[AW-1:0] + K; y_addr <= cmd_dst[AW-1:0]; v_addr <= cmd_a2[AW-1:0];
                    g_addr <= cmd_arg[AW-1:0]; slot <= cmd_a3[AW-1:0]; tag <= cmd_tag;
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
// SwiGLU.  src: gate (int8), a2: up, arg[3:0]: constants, dst: act; len: beats of NL.
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
    input  wire [29:0]   cmd_src,
    input  wire [29:0]   cmd_dst,
    input  wire [29:0]   cmd_a2,
    input  wire [29:0]   cmd_a3,
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
                busy <= 1'b1; issuing <= 1'b1; n <= cmd_len; src <= cmd_src[AW-1:0]; dst <= cmd_dst[AW-1:0]; u_addr <= cmd_a2[AW-1:0];
                k <= consts[cmd_arg[3:0]]; tag <= cmd_tag; i <= 0; o <= 0;
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
// The residual add.  src: h (int16), a2: y (int8), arg[3:0]: constants, dst: h'; len: beats of NL.
// ---------------------------------------------------------------------------
module fabric_residual_adapter #(
    parameter int NL = 8,
    parameter int AW = 16
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          cmd_valid,
    input  wire [15:0]   cmd_len,
    input  wire [29:0]   cmd_src,
    input  wire [29:0]   cmd_dst,
    input  wire [29:0]   cmd_a2,
    input  wire [29:0]   cmd_a3,
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
                busy <= 1'b1; issuing <= 1'b1; n <= cmd_len; src <= cmd_src[AW-1:0]; dst <= cmd_dst[AW-1:0]; y_addr <= cmd_a2[AW-1:0];
                k <= consts[cmd_arg[3:0]]; tag <= cmd_tag; i <= 0; o <= 0;
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
// The rotary unit.  arg[3:0] = 0: the table of sines then cosines of the
// R/2 rotary frequencies at position a3, as int16, to dst.  1: one head:
// the int8 vector at src through the head norm with the kind's (arg[7:4],
// 0 q and 1 k) gains and constants to int16, rotated by the table at a2,
// requantized to int8 at dst; len is the beat count.
// ---------------------------------------------------------------------------
module fabric_rotary_adapter #(
    parameter int HD = 24,
    parameter int R  = 6,
    parameter int NL = 8,
    parameter int SW = 38,
    parameter int AW = 16,
    parameter     LUT_DIR = "./"
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          cmd_valid,
    input  wire [29:0]   cmd_src,
    input  wire [29:0]   cmd_dst,
    input  wire [29:0]   cmd_a2,
    input  wire [29:0]   cmd_a3,
    input  wire [31:0]   cmd_arg,
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
    localparam int H = R / 2, TB = 2 * R * 8, TBEATS = (2 * R + 15) / 16, BEATS = HD / NL;
    reg [44+SW-1:0] consts [0:1];
    reg [15:0]      gains [0:2*HD-1];
    reg [31:0]      invf [0:H-1];
    initial begin
        $readmemh("rot_consts.hex", consts);
        $readmemh("rot_gains.hex", gains);
        $readmemh("inv_freq.hex", invf);
    end
    wire [H*32-1:0] inv_flat;
    genvar gh;
    generate for (gh = 0; gh < H; gh = gh + 1) begin : g_inv assign inv_flat[gh*32 +: 32] = invf[gh]; end endgenerate

    localparam [2:0] S_IDLE = 0, S_TABLE = 1, S_TWRITE = 2, S_LOADT = 3, S_TLAND = 4, S_STREAM = 5, S_WAIT = 6, S_DONE = 7;
    reg [2:0]    state;
    reg [AW-1:0] src, dst, tab_addr;
    reg [31:0]   pos;
    reg [3:0]    kind;
    reg [7:0]    tag, i, o, j, j_d;
    reg          tstart, ldv, v1;
    reg [44+SW-1:0] consts_r;
    reg [TBEATS*128-1:0] tab_r;
    reg [NL*16-1:0] g1;
    wire [44+SW-1:0] k = consts_r;
    assign cmd_ready = (state == S_IDLE);

    wire            tdone;
    wire [H*16-1:0] sin_tab, cos_tab;
    fabric_rotary_table #(.R(R), .LUT_DIR(LUT_DIR)) u_table (
        .clk(clk), .rst_n(rst_n), .start(tstart), .pos(pos), .inv_freq(inv_flat), .done(tdone), .sin_tab(sin_tab), .cos_tab(cos_tab));
    wire [TBEATS*128-1:0] tabvec = {{(TBEATS*128-TB){1'b0}}, cos_tab, sin_tab};

    reg [NL*16-1:0] x_in;
    integer l;
    always @* for (l = 0; l < NL; l = l + 1) x_in[l*16 +: 16] = {{8{rd_data[l*8+7]}}, rd_data[l*8 +: 8]};
    wire             nv;
    wire [NL*16-1:0] ny;
    fabric_rmsnorm #(.D(HD), .XW(16), .OW(16), .L(NL), .SW(SW), .LUT_DIR(LUT_DIR)) u_norm (
        .clk(clk), .rst_n(rst_n), .in_valid(v1), .n_beats(BEATS[$clog2(BEATS):0]), .in_x(x_in), .in_gain(g1), .mult(k[15:0]),
        .shift(k[21:16]), .eps(k[44 +: SW]), .out_valid(nv), .out_y(ny));
    wire            rv;
    wire [NL*8-1:0] ry;
    fabric_rotary #(.HD(HD), .R(R), .L(NL)) u_rot (
        .clk(clk), .rst_n(rst_n), .in_valid(nv), .in_x(ny), .sin_tab(tab_r[R*8-1:0]), .cos_tab(tab_r[2*R*8-1:R*8]),
        .mult(k[37:22]), .shift(k[43:38]), .out_valid(rv), .out_y(ry));

    // The x beats are issued while S_STREAM holds; their data, and the gains registered with them, follow a cycle later.
    always @* begin
        rd_addr = src + i * NL;
        if (state == S_LOADT) rd_addr = tab_addr + j * 16;
    end
    always @(posedge clk) begin
        v1 <= (state == S_STREAM);
        for (l = 0; l < NL; l = l + 1) g1[l*16 +: 16] <= gains[kind * HD + i * NL + l];
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; done_valid <= 1'b0; wr_en <= 1'b0; tstart <= 1'b0; ldv <= 1'b0; i <= 0; o <= 0; j <= 0;
        end else begin
            done_valid <= 1'b0; wr_en <= 1'b0; tstart <= 1'b0;
            ldv <= (state == S_LOADT); j_d <= j;
            if (ldv) tab_r[j_d*128 +: 128] <= rd_data;
            case (state)
                S_IDLE: if (cmd_valid) begin
                    src <= cmd_src[AW-1:0]; dst <= cmd_dst[AW-1:0]; kind <= cmd_arg[7:4]; tab_addr <= cmd_a2[AW-1:0]; pos <= {2'd0, cmd_a3};
                    consts_r <= consts[cmd_arg[4]]; tag <= cmd_tag; i <= 0; o <= 0; j <= 0;
                    if (cmd_arg[3:0] == 0) begin tstart <= 1'b1; state <= S_TABLE; end
                    else state <= S_LOADT;
                end
                S_TABLE: if (tdone) begin j <= 0; state <= S_TWRITE; end
                S_TWRITE: begin
                    wr_en <= 1'b1; wr_addr <= dst + j * 16; wr_data <= tabvec[j*128 +: 128];
                    wr_be <= (2 * R - j * 16 >= 16) ? 16'hFFFF : ((16'd1 << (2 * R - j * 16)) - 1'b1);
                    j <= j + 1'b1;
                    if (j == TBEATS - 1) state <= S_DONE;
                end
                S_LOADT: begin
                    j <= j + 1'b1;
                    if (j == TBEATS - 1) state <= S_TLAND;
                end
                S_TLAND: state <= S_STREAM;                           // the last table beat lands
                S_STREAM: begin
                    i <= i + 1'b1;
                    if (i == BEATS - 1) state <= S_WAIT;
                end
                S_WAIT: ;
                default: begin done_valid <= 1'b1; done_tag <= tag; state <= S_IDLE; end
            endcase
            if (rv) begin
                wr_en <= 1'b1; wr_addr <= dst + o * NL; wr_data <= {{(128-NL*8){1'b0}}, ry}; wr_be <= (16'd1 << NL) - 1'b1;
                o <= o + 1'b1;
                if (o == BEATS - 1) state <= S_DONE;
            end
        end
    end
endmodule

// ---------------------------------------------------------------------------
// One attention core.  src: the group's G query rows (int8, HD each); a2:
// the first head's gate row (the heads' gates are 2 HD apart); a3: the
// head's rows, len records of a key then a value row; dst: the G output
// rows.
//
// A beat's address goes out every cycle and the beat is presented the cycle
// after, as the buffer answers.  One the core does not take -- it drops its
// ready while it exponentiates a key row -- waits in a skid register, and no
// address goes out while that register would have to take a second.  Read
// and then presented, a beat was two cycles: every row cost twice its beats,
// and the core, which takes a beat a cycle, was idle half of each.
// ---------------------------------------------------------------------------
module fabric_attn_adapter #(
    parameter int HD = 24,
    parameter int G  = 2,
    parameter int L  = 8,
    parameter int LW = 28,
    parameter int AW = 16,
    parameter     LUT_DIR = "./"
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          cmd_valid,
    input  wire [15:0]   cmd_len,
    input  wire [29:0]   cmd_src,
    input  wire [29:0]   cmd_dst,
    input  wire [29:0]   cmd_a2,
    input  wire [29:0]   cmd_a3,
    input  wire [31:0]   cmd_arg,
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
    localparam int BEATS = HD / L;
    reg [65:0] consts [0:0];
    initial $readmemh("attn_consts.hex", consts);
    localparam [2:0] S_IDLE = 0, S_START = 1, S_RUN = 2, S_DRAIN = 3, S_FINISH = 4, S_OUT = 5, S_DONE = 6;
    reg [2:0]    state;
    reg [AW-1:0] src, dst, gate, rows;
    reg [15:0]   n, r, o;
    reg [7:0]    tag, g, b;
    reg [1:0]    phase;
    reg          start, finish;
    reg [65:0]   k;
    assign cmd_ready = (state == S_IDLE);
    wire            in_ready, out_valid, done;
    wire [L*8-1:0]  out_data;
    // The beat the buffer is answering (v1, of kind k1) and the one waiting
    // in the skid register.  They are never both held: an address goes out
    // only when the register will be empty, so what it answers has a place.
    reg             v1, sk_v;
    reg  [1:0]      k1, sk_k;
    reg  [L*8-1:0]  sk_d;
    wire            in_valid = sk_v || v1;
    wire [1:0]      in_kind  = sk_v ? sk_k : k1;
    wire [L*8-1:0]  in_data  = sk_v ? sk_d : rd_data[L*8-1:0];
    wire            taken    = in_valid && in_ready;
    wire            sk_next  = in_valid && !taken;
    wire            issue    = (state == S_RUN) && !sk_next;
    fabric_attention #(.HD(HD), .G(G), .L(L), .LW(LW), .LUT_DIR(LUT_DIR)) u_core (
        .clk(clk), .rst_n(rst_n), .start(start), .in_valid(in_valid), .in_ready(in_ready), .in_kind(in_kind),
        .in_data(in_data), .finish(finish), .mult_s(k[15:0]), .sh_s(k[21:16]), .mult_gate(k[37:22]), .sh_gate(k[43:38]),
        .mult_o(k[59:44]), .sh_o(k[65:60]), .out_valid(out_valid), .out_data(out_data), .done(done));
    always @* begin
        case (phase)
            2'd0: rd_addr = src + g * HD + b * L;
            2'd1: rd_addr = gate + g * 2 * HD + b * L;
            2'd2: rd_addr = rows + r * 2 * HD + b * L;
            default: rd_addr = rows + r * 2 * HD + HD + b * L;
        endcase
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; done_valid <= 1'b0; wr_en <= 1'b0; start <= 1'b0; finish <= 1'b0; phase <= 0; g <= 0; b <= 0; r <= 0; o <= 0;
            v1 <= 1'b0; sk_v <= 1'b0;
        end else begin
            done_valid <= 1'b0; wr_en <= 1'b0; start <= 1'b0; finish <= 1'b0;
            v1 <= issue;
            if (issue) k1 <= phase;
            if (sk_next && !sk_v) begin sk_d <= rd_data[L*8-1:0]; sk_k <= k1; end
            sk_v <= sk_next;
            case (state)
                S_IDLE: if (cmd_valid) begin
                    src <= cmd_src[AW-1:0]; dst <= cmd_dst[AW-1:0]; gate <= cmd_a2[AW-1:0]; rows <= cmd_a3[AW-1:0]; n <= cmd_len; tag <= cmd_tag;
                    k <= consts[0]; phase <= 0; g <= 0; b <= 0; r <= 0; o <= 0; start <= 1'b1; state <= S_START;
                end
                S_START: state <= S_RUN;
                S_RUN: if (issue) begin
                    // The counters name the next beat to ask for.
                    if (b != BEATS - 1) b <= b + 1'b1;
                    else begin
                        b <= 0;
                        case (phase)
                            2'd0: if (g == G - 1) begin g <= 0; phase <= 2'd1; end else g <= g + 1'b1;
                            2'd1: if (g == G - 1) begin g <= 0; phase <= 2'd2; if (n == 0) state <= S_DRAIN; end else g <= g + 1'b1;
                            2'd2: phase <= 2'd3;
                            default: begin phase <= 2'd2; r <= r + 1'b1; if (r == n - 1) state <= S_DRAIN; end
                        endcase
                    end
                end
                // The last beat asked for: finish once the core has taken it.
                S_DRAIN: if (!sk_next) state <= S_FINISH;
                S_FINISH: begin finish <= 1'b1; state <= S_OUT; end
                S_OUT: if (done) state <= S_DONE;
                default: begin done_valid <= 1'b1; done_tag <= tag; state <= S_IDLE; end
            endcase
            if (out_valid) begin
                wr_en <= 1'b1; wr_addr <= dst + o * L; wr_data <= {{(128-L*8){1'b0}}, out_data}; wr_be <= (16'd1 << L) - 1'b1;
                o <= o + 1'b1;
            end
        end
    end
endmodule

// ---------------------------------------------------------------------------
// The memory unit.  arg[3:0] is the operation, arg[31:4] the position and
// a3 the context's memory page (2 KB units) where the global layer needs
// them:
//   0  memory beats src (a beat address) to the buffer at dst, len beats
//   1  buffer src to memory beats dst, len beats
//   2  append: keys at src, values at a2, index projection at dst, with
//      the context's running block sums read from and written back to
//      its sums record
//   3  scan: the int8 unit index query at src coded and scored over the
//      eligible index records; the top-K ids (a count then the ids, int16)
//      to dst
//   4  rows: the selection at src; KV head a2's window records then its
//      selected block records, as int8 key and value rows, to dst
// The four requesters (a beat mover, the append, the scan, the record
// reader) share the port through fabric_mem_arbiter.
// ---------------------------------------------------------------------------
module fabric_mem_unit #(
    parameter int HD      = 24,
    parameter int NKV     = 2,
    parameter int IDIM    = 32,
    parameter int BS      = 4,
    parameter int W       = 16,
    parameter int TOP     = 2,
    parameter int KV_BITS = 4,
    parameter int L       = 8,                  // int8 elements per beat out of the record reader
    parameter int REC_BYTES  = 32,
    parameter int RPB     = 64,                 // index records per scan request
    parameter int MAXR    = 64,                 // window records per read request
    parameter int WINDOW_OFF = 0,
    parameter int BLOCK_OFF  = 2048,
    parameter int INDEX_OFF  = 6144,
    parameter int SUMS_OFF   = 8192,
    parameter int AW      = 16,
    parameter int STATE_ONE = 16'hFFFF,         // a fresh state slot's scale: 1.0 in U16 (fabric.layer.ONE_U)
    parameter     LUT_DIR = "./"
) (
    input  wire          clk,
    input  wire          rst_n,
    // The tokens in flight: each one's slot, as a page (2 KB), and whether it
    // is FIRST.  A command names its token in a3[29:28]; its memory addresses
    // are offsets in that token's slot, and the page is added here, so one
    // program image serves a context in any slot.
    input  wire [4*21-1:0] slot_page,
    input  wire [3:0]    first,
    input  wire          cmd_valid,
    input  wire [15:0]   cmd_len,
    input  wire [29:0]   cmd_src,
    input  wire [29:0]   cmd_dst,
    input  wire [29:0]   cmd_a2,
    input  wire [29:0]   cmd_a3,
    input  wire [31:0]   cmd_arg,
    input  wire [7:0]    cmd_tag,
    output wire          cmd_ready,
    output reg           done_valid,
    output reg  [7:0]    done_tag,
    output reg  [AW-1:0] rd_addr,
    output wire          rd_en,          // the loader, or a move out of the buffer; else the port is idle
    input  wire [127:0]  rd_data,
    input  wire [127:0]  rd_hi,          // the next word, for a move out: the buffer's wide read port
    output reg           wr_en,
    output reg  [AW-1:0] wr_addr,
    output reg  [127:0]  wr_data,
    output reg  [15:0]   wr_be,
    output reg  [127:0]  wr_hi,          // and its wide write port
    output reg  [15:0]   wr_hi_be,
    output wire          m_req_valid,
    input  wire          m_req_ready,
    output wire          m_req_write,
    output wire          m_req_wide,
    output wire [31:0]   m_req_addr,
    output wire [11:0]   m_req_beats,
    output wire          m_wdata_valid,
    input  wire          m_wdata_ready,
    output wire [255:0]  m_wdata,
    input  wire          m_rdata_valid,
    input  wire [255:0]  m_rdata
);
    localparam int DW = 128;
    localparam int KB = (NKV * HD + 15) / 16, IB = (IDIM + 15) / 16;          // beats of the key rows, of an index vector
    localparam int SUMS_BITS = (2 * NKV * HD + IDIM) * 16, SUMS_BEATS = (SUMS_BITS + 127) / 128;
    localparam int SEL_BYTES = 2 * (1 + TOP), SEL_BEATS = (SEL_BYTES + 15) / 16;
    localparam int LOG_BS = $clog2(BS), BEATS = HD / L;
    localparam int NR = 4;

    // The command.
    reg [3:0]    op;
    reg [31:0]   pos, ctx_base;
    reg [15:0]   n;
    reg [AW-1:0] src, dst, arg_lo;
    reg [7:0]    tag, head;
    reg          cmd_first;                 // the command's token is FIRST
    wire [1:0]   cmd_tok  = cmd_a3[29:28];
    wire [20:0]  cmd_page = slot_page[cmd_tok*21 +: 21];
    wire [31:0]  cmd_slot = {cmd_page, 11'd0};

    // Requesters onto the port.
    // The port carries two beats a transfer for a wide request (the mover's
    // moves to and from the buffer) and one, in the low half, for the rest.
    wire [NR-1:0]     r_req_valid, r_req_ready, r_req_write, r_req_wide, r_wdata_valid, r_wdata_ready, r_rdata_valid;
    wire [NR*32-1:0]  r_req_addr;
    wire [NR*12-1:0]  r_req_beats;
    wire [NR*2*DW-1:0] r_wdata;
    wire [2*DW-1:0]   r_rdata;
    fabric_mem_arbiter #(.N(NR), .DW(DW), .XW(2)) u_arb (
        .clk(clk), .rst_n(rst_n), .r_req_valid(r_req_valid), .r_req_ready(r_req_ready), .r_req_write(r_req_write),
        .r_req_wide(r_req_wide), .r_req_addr(r_req_addr), .r_req_beats(r_req_beats), .r_wdata_valid(r_wdata_valid),
        .r_wdata_ready(r_wdata_ready), .r_wdata(r_wdata), .r_rdata_valid(r_rdata_valid), .r_rdata(r_rdata),
        .m_req_valid(m_req_valid), .m_req_ready(m_req_ready), .m_req_write(m_req_write), .m_req_wide(m_req_wide),
        .m_req_addr(m_req_addr), .m_req_beats(m_req_beats), .m_wdata_valid(m_wdata_valid), .m_wdata_ready(m_wdata_ready),
        .m_wdata(m_wdata), .m_rdata_valid(m_rdata_valid), .m_rdata(m_rdata));
    assign r_req_wide[3:1] = 3'b000;

    // Requester 0, the mover: a burst between memory and the buffer or the sums register.
    localparam [1:0] MV_RD_VB = 0, MV_WR_VB = 1, MV_RD_REG = 2, MV_WR_REG = 3;
    reg          mv_go, mv_busy, mv_req, mv_done, mv_present;
    // FIRST.  A move into the buffer flagged fresh, or the append's read of
    // the block sums, on a FIRST token is a fill instead of a read: the slot
    // may hold another context's state, or whatever the devices powered up
    // with, and this token starts from nothing.  Zeros, and for a state slot
    // its header beat with the scale at 1.0 -- the integer model's fresh
    // state -- two beats a cycle into the buffer and no request on the port.
    reg          mv_fill, mv_hdr;
    reg [1:0]    mv_mode;
    reg [31:0]   mv_maddr;
    reg [AW-1:0] mv_vaddr;
    reg [11:0]   mv_n, mv_i;
    // Writing the buffer out, the read addresses run a beat ahead of the data:
    // `mv_a` is the beat whose address has gone to the buffer, `mv_i` the beat
    // the port has taken, `mv_q` says the buffer is answering this cycle and
    // `mv_hold` catches that answer when the port is not ready.  Written as
    // address, then data, then present -- one beat at a time through all three
    // -- this was three cycles a beat, and the state DMA is most of a recurrent
    // token.
    //
    // Moves between memory and the buffer are wide: two beats a transfer, the
    // buffer's two words at the address and the one after, so a move is half
    // as many transfers and the state DMA half as long.  The sums register
    // moves stay a beat at a time.  `mv_i` and `mv_a` still count beats.
    reg          mv_q, mv_hv;
    reg [11:0]   mv_a;
    reg [2*DW-1:0] mv_hold;
    wire         mv_wide   = (mv_mode == MV_RD_VB) || (mv_mode == MV_WR_VB);
    wire [11:0]  mv_istep  = (mv_wide && (mv_n - mv_i > 1)) ? 12'd2 : 12'd1;   // beats the next transfer carries
    wire [11:0]  mv_astep  = (mv_n - mv_a > 1) ? 12'd2 : 12'd1;                 // and the next read of the buffer
    wire         mv_have   = mv_hv || mv_q;                 // a beat is ready to present
    wire         mv_take   = mv_have && r_wdata_ready[0];
    wire         mv_keep   = mv_q && !mv_take;              // it has to wait: hold it
    wire         mv_hv_nxt = mv_keep || (mv_hv && !mv_take);
    wire         mv_issue  = mv_busy && (mv_mode == MV_WR_VB) && !mv_req && (mv_a < mv_n) && !mv_hv_nxt;
    assign r_req_valid[0]   = mv_req;
    assign r_req_write[0]   = (mv_mode == MV_WR_VB) || (mv_mode == MV_WR_REG);
`ifndef FABRIC_SYNTH
    always @(posedge clk) if (mv_go && mv_wide && mv_vaddr[3:0] != 4'd0)
        $display("FAIL: a wide move at buffer byte %0d, not a word boundary", mv_vaddr);
`endif
    assign r_req_wide[0]    = mv_wide;
    assign r_req_addr[0*32 +: 32]  = mv_maddr;
    assign r_req_beats[0*12 +: 12] = mv_n;
    assign r_wdata_valid[0] = (mv_mode == MV_WR_VB) ? mv_have : mv_present;
    assign r_wdata[0 +: 2*DW] = (mv_mode == MV_WR_REG) ? {{DW{1'b0}}, ap_s_out_data}
                              : ((mv_mode == MV_WR_VB) && mv_hv) ? mv_hold : {rd_hi, rd_data};

    // Requester 1, the append.
    reg [NKV*HD*8-1:0] k_r, v_r;
    reg [IDIM*8-1:0]   idx_r, u_r;
    reg                ap_start;
    wire               ap_done;
    // The sums move a beat at a time between the memory and the append's own
    // memory: the mover reads one, the append adds the rows to it and keeps
    // it; the mover writes one back, and the append hands it over.
    wire        ap_s_in_valid = mv_busy && (mv_mode == MV_RD_REG) && (r_rdata_valid[0] || mv_fill);
    wire [DW-1:0] ap_s_out_data;
    fabric_kv_append #(.DW(DW), .HD(HD), .NKV(NKV), .IDIM(IDIM), .BS(BS), .KV_BITS(KV_BITS), .W(W), .LUT_DIR(LUT_DIR)) u_append (
        .clk(clk), .rst_n(rst_n), .start(ap_start), .pos(pos), .window_base(ctx_base + WINDOW_OFF), .block_base(ctx_base + BLOCK_OFF),
        .index_base(ctx_base + INDEX_OFF), .k_rows(k_r), .v_rows(v_r), .idx_k(idx_r),
        .s_in_valid(ap_s_in_valid), .s_in_addr({{(16-$clog2(SUMS_BEATS+1)){1'b0}}, mv_i[$clog2(SUMS_BEATS+1)-1:0]}), .s_in_data(mv_fill ? {DW{1'b0}} : r_rdata[DW-1:0]),
        .s_out_addr({{(16-$clog2(SUMS_BEATS+1)){1'b0}}, mv_i[$clog2(SUMS_BEATS+1)-1:0]}), .s_out_data(ap_s_out_data), .done(ap_done),
        .req_valid(r_req_valid[1]), .req_ready(r_req_ready[1]), .req_addr(r_req_addr[1*32 +: 32]), .req_beats(r_req_beats[1*12 +: 12]),
        .wdata_valid(r_wdata_valid[1]), .wdata_ready(r_wdata_ready[1]), .wdata(r_wdata[1*2*DW +: DW]));
    assign r_wdata[1*2*DW + DW +: DW] = 0;
    assign r_req_write[1] = 1'b1;

    // Requester 2, the index scan with its top-K.
    reg              sc_start, tk_clear, tk_finish, sc_done_d;
    reg [15:0]       n_blocks;
    reg [IDIM*4-1:0] q_codes;
    wire             sc_done, cand_valid, tk_out_valid, tk_out_last, tk_done;
    wire [15:0]      cand_id, tk_out_id;
    wire signed [31:0] cand_score, tk_out_score;
    fabric_index_scan #(.DW(DW), .IDIM(IDIM), .IDW(16), .RPB(RPB)) u_scan (
        .clk(clk), .rst_n(rst_n), .start(sc_start), .base(ctx_base + INDEX_OFF), .n_blocks(n_blocks), .q_codes(q_codes), .done(sc_done),
        .cand_valid(cand_valid), .cand_id(cand_id), .cand_score(cand_score),
        .req_valid(r_req_valid[2]), .req_ready(r_req_ready[2]), .req_addr(r_req_addr[2*32 +: 32]), .req_beats(r_req_beats[2*12 +: 12]),
        .rdata_valid(r_rdata_valid[2]), .rdata(r_rdata[DW-1:0]));
    assign r_req_write[2] = 1'b0;
    assign r_wdata_valid[2] = 1'b0;
    assign r_wdata[2*2*DW +: 2*DW] = 0;
    fabric_topk #(.K(TOP), .IDW(16), .SW(32)) u_topk (
        .clk(clk), .rst_n(rst_n), .clear(tk_clear), .cand_valid(cand_valid), .cand_id(cand_id), .cand_score(cand_score),
        .finish(tk_finish), .out_valid(tk_out_valid), .out_id(tk_out_id), .out_score(tk_out_score), .out_last(tk_out_last), .done(tk_done));
    reg [7:0]  scale;
    reg        rc_start;
    wire       rc_done;
    wire [16:0] rc_r;
    wire [5:0]  rc_lz;
    fabric_recip #(.LW(9), .LUT_DIR(LUT_DIR)) u_rc (.clk(clk), .start(rc_start), .l({scale, 1'b0}), .done(rc_done), .r(rc_r), .lz_out(rc_lz));
    reg [SEL_BEATS*128-1:0] sel_r;             // the selection: count then ids as int16
    reg [7:0]  sel_count;
    wire signed [31:0] span = $signed(pos) - W + 1;
    wire [15:0] eligible = (span < 0) ? 16'd0 : span[LOG_BS +: 16];

    // Requester 3, the record reader.
    reg          rr_addr_valid;
    reg [31:0]   rr_addr;
    reg [7:0]    rr_count;
    wire         rr_addr_ready, rr_out_valid, rr_rec_done;
    wire [1:0]   rr_kind;
    wire [L*8-1:0] rr_data;
    fabric_record_reader #(.DW(DW), .HD(HD), .KV_BITS(KV_BITS), .L(L), .MAXR(MAXR)) u_reader (
        .clk(clk), .rst_n(rst_n), .addr_valid(rr_addr_valid), .addr_ready(rr_addr_ready), .addr(rr_addr), .addr_count(rr_count),
        .req_valid(r_req_valid[3]), .req_ready(r_req_ready[3]), .req_addr(r_req_addr[3*32 +: 32]), .req_beats(r_req_beats[3*12 +: 12]),
        .rdata_valid(r_rdata_valid[3]), .rdata(r_rdata[DW-1:0]), .out_valid(rr_out_valid), .out_ready(1'b1), .out_kind(rr_kind),
        .out_data(rr_data), .rec_done(rr_rec_done));
    assign r_req_write[3] = 1'b0;
    assign r_wdata_valid[3] = 1'b0;
    assign r_wdata[3*2*DW +: 2*DW] = 0;
    reg [15:0] rw_p, rw_last, rw_j, rw_total, rw_rec;
    reg [7:0]  rw_ob;
    reg        rw_blocks, rw_reqs_done;
    wire [15:0] rw_wrap  = W - (rw_p % W);                                  // records before the window wraps
    wire [15:0] rw_left  = rw_last - rw_p + 1;
    wire [15:0] rw_run   = (rw_left < rw_wrap) ? rw_left : rw_wrap;
    wire [7:0]  rw_cnt   = (rw_run > MAXR) ? MAXR[7:0] : rw_run[7:0];

    // The loader: beats of the buffer into one of the operand registers.
    localparam [2:0] T_K = 0, T_V = 1, T_I = 2, T_U = 3, T_SEL = 4;
    reg          ld_on, ldv;
    reg [2:0]    ld_tgt, ld_tgt_d;
    reg [AW-1:0] ld_base;
    reg [7:0]    ld_i, ld_i_d, ld_n;
    always @* begin
        rd_addr = ld_base + ld_i * 16;
        if (mv_busy && mv_mode == MV_WR_VB) rd_addr = mv_vaddr + mv_a * 16;
    end
    assign rd_en = ld_on || mv_issue;

    // The query's largest magnitude, as a balanced tree rather than a chain of
    // compare-selects as long as the vector.
    localparam int ULV = $clog2(IDIM), UPP = 1 << ULV;
    reg [8:0] utree [0:ULV][0:UPP-1];
    integer ul, uc;
    always @* begin
        for (uc = 0; uc < UPP; uc = uc + 1) begin
            utree[0][uc] = 9'd1;
            if (uc < IDIM)
                utree[0][uc] = u_r[uc*8+7] ? (9'd256 - {1'b0, u_r[uc*8 +: 8]}) : {1'b0, u_r[uc*8 +: 8]};
        end
        for (ul = 1; ul <= ULV; ul = ul + 1)
            for (uc = 0; uc < (UPP >> ul); uc = uc + 1)
                utree[ul][uc] = (utree[ul-1][2*uc] > utree[ul-1][2*uc+1]) ? utree[ul-1][2*uc] : utree[ul-1][2*uc+1];
    end
    wire [8:0] u_absmax = (utree[ULV][0] > 9'd1) ? utree[ULV][0] : 9'd1;

    localparam [4:0] S_IDLE = 0, S_MV = 1, S_DONE = 2,
                     S_AP_LOAD = 3, S_AP_SUMS_RD = 4, S_AP_START = 5, S_AP_WAIT = 6, S_AP_SUMS_WR = 7,
                     S_SC_LOAD = 8, S_SC_SCALE = 9, S_SC_RECIP = 10, S_SC_CODES = 11, S_SC_RUN = 12, S_SC_COLLECT = 13, S_SC_WRITE = 14,
                     S_RW_LOAD = 15, S_RW_REQ = 16, S_RW_WAIT = 17;
    reg [4:0] state;
    assign cmd_ready = (state == S_IDLE);
    integer j;
    reg signed [63:0] t, mx;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; done_valid <= 1'b0; wr_en <= 1'b0; wr_hi_be <= 16'd0; ld_on <= 1'b0; cmd_first <= 1'b0; ldv <= 1'b0; ld_i <= 0; ld_n <= 0;
            mv_go <= 1'b0; mv_busy <= 1'b0; mv_req <= 1'b0; mv_done <= 1'b0; mv_present <= 1'b0; mv_i <= 0; mv_fill <= 1'b0; mv_hdr <= 1'b0;
            mv_a <= 0; mv_q <= 1'b0; mv_hv <= 1'b0;
            ap_start <= 1'b0; sc_start <= 1'b0; tk_clear <= 1'b0; tk_finish <= 1'b0; sc_done_d <= 1'b0; rc_start <= 1'b0;
            rr_addr_valid <= 1'b0; rw_p <= 0; rw_j <= 0; rw_rec <= 0; rw_ob <= 0; rw_blocks <= 1'b0; rw_reqs_done <= 1'b0; sel_count <= 0;
        end else begin
            done_valid <= 1'b0; wr_en <= 1'b0; wr_hi_be <= 16'd0; ap_start <= 1'b0; sc_start <= 1'b0; tk_clear <= 1'b0; tk_finish <= 1'b0; rc_start <= 1'b0;
            mv_go <= 1'b0; mv_done <= 1'b0;
            // Loader data lands a cycle after its issue.
            ldv <= ld_on; ld_i_d <= ld_i; ld_tgt_d <= ld_tgt;
            if (ld_on) begin
                ld_i <= ld_i + 1'b1;
                if (ld_i == ld_n - 1) ld_on <= 1'b0;
            end
            if (ldv) case (ld_tgt_d)
                T_K:   k_r[ld_i_d*128 +: 128] <= rd_data;
                T_V:   v_r[ld_i_d*128 +: 128] <= rd_data;
                T_I:   idx_r[ld_i_d*128 +: 128] <= rd_data;
                T_U:   u_r[ld_i_d*128 +: 128] <= rd_data;
                default: sel_r[ld_i_d*128 +: 128] <= rd_data;
            endcase
            // The mover.
            if (mv_go) begin
                mv_busy <= 1'b1; mv_req <= !mv_fill; mv_i <= 0; mv_present <= 1'b0;
                mv_a <= 0; mv_q <= 1'b0; mv_hv <= 1'b0;
            end
            else if (mv_busy) begin
                if (mv_req && r_req_ready[0]) mv_req <= 1'b0;
                case (mv_mode)
                    MV_RD_VB: if (r_rdata_valid[0] || mv_fill) begin
                        wr_en <= 1'b1; wr_addr <= mv_vaddr + mv_i * 16; wr_be <= 16'hFFFF;
                        wr_data <= !mv_fill ? r_rdata[DW-1:0] : ((mv_hdr && mv_i == 0) ? STATE_ONE[15:0] : {DW{1'b0}});
                        wr_hi <= mv_fill ? {DW{1'b0}} : r_rdata[2*DW-1:DW]; wr_hi_be <= (mv_istep == 2) ? 16'hFFFF : 16'd0;
                        mv_i <= mv_i + mv_istep;
                        if (mv_i + mv_istep == mv_n) begin mv_busy <= 1'b0; mv_done <= 1'b1; end
                    end
                    MV_RD_REG: if (r_rdata_valid[0] || mv_fill) begin
                        mv_i <= mv_i + 1'b1;             // the append takes the beat; see ap_s_in_valid
                        if (mv_i == mv_n - 1) begin mv_busy <= 1'b0; mv_done <= 1'b1; end
                    end
                    MV_WR_VB: begin
                        // One beat a cycle: the address of the next goes out while
                        // this one is on the port, and a beat the port did not take
                        // waits in `mv_hold` rather than being read again.
                        mv_q <= mv_issue;
                        mv_hv <= mv_hv_nxt;
                        if (mv_keep) mv_hold <= {rd_hi, rd_data};
                        if (mv_issue) mv_a <= mv_a + mv_astep;
                        if (mv_take) begin
                            mv_i <= mv_i + mv_istep;
                            if (mv_i + mv_istep == mv_n) begin mv_busy <= 1'b0; mv_done <= 1'b1; end
                        end
                    end
                    default: begin
                        if (!mv_req && !mv_present) mv_present <= 1'b1;
                        else if (mv_present && r_wdata_ready[0]) begin
                            mv_present <= 1'b0; mv_i <= mv_i + 1'b1;
                            if (mv_i == mv_n - 1) begin mv_busy <= 1'b0; mv_done <= 1'b1; end
                        end
                    end
                endcase
            end
            // Top-K collection and the row writes run whenever their units produce.
            sc_done_d <= sc_done;
            if (tk_out_valid) begin sel_r[16 + sel_count*16 +: 16] <= tk_out_id; sel_count <= sel_count + 1'b1; end
            if (rr_out_valid) begin
                wr_en <= 1'b1; wr_be <= (16'd1 << L) - 1'b1; wr_data <= {{(128-L*8){1'b0}}, rr_data};
                wr_addr <= dst + rw_rec * 2 * HD + ((rr_kind == 2'd3) ? HD : 0) + ((rw_ob < BEATS) ? rw_ob : rw_ob - BEATS) * L;
                rw_ob <= rw_ob + 1'b1;
            end
            if (rr_rec_done) begin rw_rec <= rw_rec + 1'b1; rw_ob <= 0; end
            case (state)
                S_IDLE: if (cmd_valid) begin
                    op <= cmd_arg[3:0]; pos <= {4'd0, cmd_arg[31:4]}; ctx_base <= {cmd_a3[20:0] + cmd_page, 11'd0};
                    cmd_first <= first[cmd_tok];
                    n <= cmd_len; src <= cmd_src[AW-1:0]; dst <= cmd_dst[AW-1:0]; arg_lo <= cmd_a2[AW-1:0]; head <= cmd_a2[7:0]; tag <= cmd_tag;
                    case (cmd_arg[3:0])
                        4'd0: begin mv_go <= 1'b1; mv_mode <= MV_RD_VB; mv_maddr <= {cmd_src[27:0], 4'd0} + cmd_slot; mv_vaddr <= cmd_dst[AW-1:0]; mv_n <= cmd_len[11:0]; state <= S_MV;
                              mv_fill <= first[cmd_tok] && (cmd_arg[5:4] != 2'b00); mv_hdr <= cmd_arg[5]; end
                        4'd1: begin mv_go <= 1'b1; mv_fill <= 1'b0; mv_mode <= MV_WR_VB; mv_maddr <= {cmd_dst[27:0], 4'd0} + cmd_slot; mv_vaddr <= cmd_src[AW-1:0]; mv_n <= cmd_len[11:0]; state <= S_MV; end
                        4'd2: begin ld_on <= 1'b1; ld_tgt <= T_K; ld_base <= cmd_src[AW-1:0]; ld_i <= 0; ld_n <= KB; state <= S_AP_LOAD; end
                        4'd3: begin ld_on <= 1'b1; ld_tgt <= T_U; ld_base <= cmd_src[AW-1:0]; ld_i <= 0; ld_n <= IB; state <= S_SC_LOAD; end
                        default: begin ld_on <= 1'b1; ld_tgt <= T_SEL; ld_base <= cmd_src[AW-1:0]; ld_i <= 0; ld_n <= SEL_BEATS; state <= S_RW_LOAD; end
                    endcase
                end
                S_MV: if (mv_done) state <= S_DONE;
                // Append: keys, values, the index projection, then the sums, then the unit.
                S_AP_LOAD: if (!ld_on && !ldv) begin
                    if (ld_tgt == T_K) begin ld_on <= 1'b1; ld_tgt <= T_V; ld_base <= arg_lo; ld_i <= 0; ld_n <= KB; end
                    else if (ld_tgt == T_V) begin ld_on <= 1'b1; ld_tgt <= T_I; ld_base <= dst; ld_i <= 0; ld_n <= IB; end
                    else begin
                        mv_go <= 1'b1; mv_mode <= MV_RD_REG; mv_fill <= cmd_first && (pos == 0); mv_maddr <= ctx_base + SUMS_OFF; mv_n <= SUMS_BEATS; state <= S_AP_SUMS_RD;
                    end
                end
                S_AP_SUMS_RD: if (mv_done) begin ap_start <= 1'b1; state <= S_AP_WAIT; end
                S_AP_WAIT: if (ap_done) begin
                    mv_go <= 1'b1; mv_mode <= MV_WR_REG; mv_fill <= 1'b0; mv_maddr <= ctx_base + SUMS_OFF; mv_n <= SUMS_BEATS; state <= S_AP_SUMS_WR;
                end
                S_AP_SUMS_WR: if (mv_done) state <= S_DONE;
                // Scan: the query's codes, then the scan and the top-K, then the selection.
                S_SC_LOAD: if (!ld_on && !ldv) state <= S_SC_SCALE;
                S_SC_SCALE: begin
                    scale <= u_absmax[7:0]; rc_start <= 1'b1; state <= S_SC_RECIP;
                end
                S_SC_RECIP: if (rc_done) begin
                    for (j = 0; j < IDIM; j = j + 1) begin
                        t = fx_rnd_shr(64'sd15 * ($signed(u_r[j*8 +: 8]) + $signed({56'b0, scale})) * $signed({47'b0, rc_r}), 24 - rc_lz);
                        if (t < 0) t = 0;
                        if (t > 15) t = 15;
                        q_codes[j*4 +: 4] <= t[3:0];
                    end
                    n_blocks <= eligible; tk_clear <= 1'b1; sel_count <= 0; sel_r <= 0; state <= S_SC_CODES;
                end
                S_SC_CODES: begin sc_start <= 1'b1; state <= S_SC_RUN; end
                S_SC_RUN: if (sc_done_d) begin tk_finish <= 1'b1; state <= S_SC_COLLECT; end   // after the last candidate entered
                S_SC_COLLECT: if (tk_done) begin rw_j <= 0; state <= S_SC_WRITE; end
                S_SC_WRITE: begin
                    wr_en <= 1'b1; wr_addr <= dst + rw_j * 16; wr_be <= (SEL_BYTES - rw_j * 16 >= 16) ? 16'hFFFF : ((16'd1 << (SEL_BYTES - rw_j * 16)) - 1'b1);
                    wr_data <= (rw_j == 0) ? {sel_r[127:16], 8'd0, sel_count} : sel_r[rw_j*128 +: 128];
                    rw_j <= rw_j + 1'b1;
                    if (rw_j == SEL_BEATS - 1) state <= S_DONE;
                end
                // Rows: the window run by run, then one request per selected block.
                S_RW_LOAD: if (!ld_on && !ldv) begin
                    rw_p <= (pos + 1 > W) ? pos + 1 - W : 0; rw_last <= pos[15:0]; rw_j <= 0; rw_rec <= 0; rw_ob <= 0;
                    rw_blocks <= 1'b0; rw_reqs_done <= 1'b0; rw_total <= ((pos + 1 > W) ? W : pos[15:0] + 1) + sel_r[7:0];
                    state <= S_RW_REQ;
                end
                S_RW_REQ: begin
                    if (rr_addr_valid && rr_addr_ready) begin
                        rr_addr_valid <= 1'b0;
                        if (!rw_blocks) rw_p <= rw_p + rw_cnt; else rw_j <= rw_j + 1'b1;
                    end else if (!rr_addr_valid) begin
                        if (!rw_blocks) begin
                            if (rw_p <= rw_last) begin
                                rr_addr_valid <= 1'b1; rr_count <= rw_cnt;
                                rr_addr <= ctx_base + WINDOW_OFF + (head * W + rw_p % W) * REC_BYTES;
                            end else rw_blocks <= 1'b1;
                        end else if (rw_j < sel_r[7:0]) begin
                            rr_addr_valid <= 1'b1; rr_count <= 8'd1;
                            rr_addr <= ctx_base + BLOCK_OFF + (sel_r[16 + rw_j*16 +: 16] * NKV + head) * REC_BYTES;
                        end else state <= S_RW_WAIT;
                    end
                end
                S_RW_WAIT: if (rw_rec == rw_total) state <= S_DONE;
                default: begin done_valid <= 1'b1; done_tag <= tag; state <= S_IDLE; end
            endcase
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
    parameter int NH   = 4,                     // the global layer
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
    parameter int ATT_L = 8,                    // the attention cores, the record reader and the rotary
    parameter int SW_L  = 16,                   // SwiGLU, whose operands are int8
    parameter int ROWS = 96,                    // the tiles
    parameter int COLS = 16,
    parameter int P    = 2,
    parameter int NT   = 56,
    parameter int TMAX = 1,                     // tokens a pass may carry (chunked prefill)
    parameter int MODEL_TILES = 0,              // behavioural columns in the tiles (full-size runs)
    parameter int WB   = 4,
    parameter int ACC  = 24,
    parameter int SB   = 16,
    parameter int SHB  = 5,
    parameter int SW   = 38,
    parameter int YSH  = 9,
    parameter int VB_BYTES = 4096,
    parameter int AW   = 24,                    // vector-buffer address bits
    parameter int VB_BANKS = 1,                 // and its banks, from the program's colouring
    parameter int VB_BANK_SHIFT = 12,
    parameter [63:0] VB_RCAP2 = 0,
    parameter [63:0] VB_RCAP3 = 0,
    parameter [63:0] VB_WCAP2 = 0,
    parameter int VB_NPR = 24,                  // crossbar ports the logical ones fold onto
    parameter int VB_NPW = 19,
    parameter [63:0] VB_RMAP0 = 64'hFEDCBA9876543210,
    parameter [63:0] VB_RMAP1 = 64'hFEDCBA9876543210,
    parameter [63:0] VB_WMAP0 = 64'hFEDCBA9876543210,
    parameter [63:0] VB_WMAP1 = 64'hFEDCBA9876543210,
    parameter     VB_FILE   = "vb_init.hex",
    parameter     PROG_FILE = "program.hex",
    parameter     LUT_DIR   = "./"
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire [3:0]   first,               // FIRST, per token in flight (latched at start)
    input  wire [4*21-1:0] slot_page,        // each token in flight's slot, in 2 KB pages (latched at start)
    input  wire [15:0]  n_steps,
    output wire         running,
    output wire         done,
    // the memory port (fabric_memory.sv's request protocol)
    output wire         m_req_valid,
    input  wire         m_req_ready,
    output wire         m_req_write,
    output wire         m_req_wide,          // two beats a transfer: see fabric_memory.sv
    output wire [31:0]  m_req_addr,
    output wire [11:0]  m_req_beats,
    output wire         m_wdata_valid,
    input  wire         m_wdata_ready,
    output wire [255:0] m_wdata,
    input  wire         m_rdata_valid,
    input  wire [255:0] m_rdata
);
    // A buffer beat is sixteen bytes: a unit whose operands are int16 takes
    // eight of them a beat, one whose operands are int8 takes sixteen.  NL is
    // the norm's and the residual's, whose vectors are int16; CL is the conv's,
    // whose history is four bytes a channel.  The int8 units -- SwiGLU, the
    // rotary, the attention cores and the record reader -- take SW_L and ATT_L.
    localparam int NU = 10, NE = 4, NL = 8, CL = 4, GROUP = NH / NKV;
    localparam int U_TILES = 0, U_NORM = 1, U_CONV = 2, U_GATES = 3, U_DELTA = 4, U_SWIGLU = 5, U_RESIDUAL = 6, U_ROTARY = 7, U_ATTN = 8, U_MEM = 9;
    // Vector-buffer ports.
    localparam int R_NORM = 0, R_TILES = 4, R_CONV = R_TILES + TMAX, R_GATES = R_CONV + 2, R_DELTA = R_GATES + 2, R_SWIGLU = R_DELTA + 4,
                   R_RESIDUAL = R_SWIGLU + 2, R_MEM = R_RESIDUAL + 2, R_ROTARY = R_MEM + 1, R_ATTN = R_ROTARY + 2, NR = R_ATTN + 4;
    localparam int W_NORM = 0, W_TILES = 2, W_CONV = 3, W_GATES = 5, W_DELTA = 6, W_SWIGLU = 10, W_RESIDUAL = 11, W_MEM = 12,
                   W_ROTARY = 13, W_ATTN = 15, NW = 19;

    wire [NU-1:0]      cmd_valid, cmd_ready;
    wire [3:0]         cmd_engine;
    wire [15:0]        cmd_len;
    wire [29:0]        cmd_src, cmd_dst, cmd_a2, cmd_a3;
    wire [31:0]        cmd_arg;
    wire [7:0]         cmd_tag;
    wire [NU*NE-1:0]   done_valid;
    wire [NU*NE*8-1:0] done_tag;
    fabric_sequencer #(.NU(NU), .NE(NE), .PROG_FILE(PROG_FILE)) u_seq (
        .clk(clk), .rst_n(rst_n), .start(start), .n_steps(n_steps), .running(running), .done(done),
        .cmd_valid(cmd_valid), .cmd_engine(cmd_engine), .cmd_len(cmd_len), .cmd_src(cmd_src), .cmd_dst(cmd_dst),
        .cmd_a2(cmd_a2), .cmd_a3(cmd_a3), .cmd_arg(cmd_arg), .cmd_tag(cmd_tag), .cmd_ready(cmd_ready), .done_valid(done_valid), .done_tag(done_tag));

    wire [NR*AW-1:0]  rd_addr;
    wire [NR*128-1:0] rd_data;
    wire [NW-1:0]     wr_en;
    wire [NW*AW-1:0]  wr_addr;
    wire [NW*128-1:0] wr_data;
    wire [NW*16-1:0]  wr_be;
    // A read port is asking only while its adapter holds a command: the
    // address it presents at any other time is the last one it used, and a
    // bank must not be charged for it.  `cmd_ready` is the adapter idle, so
    // its complement is the enable, and the cycle a command is taken the
    // address is still the previous command's -- the first beat's address
    // is registered at that edge and read the cycle after, inside the busy.
    wire [NR-1:0] rd_en;
    wire [1:0]    norm_gain_en;
    assign rd_en[R_TILES +: TMAX]  = {TMAX{!ready_tiles}};
    assign rd_en[R_CONV +: 2]      = {2{!ready_conv}};
    assign rd_en[R_GATES +: 2]     = {2{!ready_gates}};
    assign rd_en[R_SWIGLU +: 2]    = {2{!ready_swiglu}};
    assign rd_en[R_RESIDUAL +: 2]  = {2{!ready_residual}};

    genvar ge;
    generate
        for (ge = 0; ge < 2; ge = ge + 1) begin : g_en2
            assign rd_en[R_NORM + 2*ge]     = !ready_norm[ge];
            assign rd_en[R_NORM + 2*ge + 1] = norm_gain_en[ge];
            assign rd_en[R_ROTARY + ge]      = !ready_rotary[ge];
        end
        for (ge = 0; ge < NE; ge = ge + 1) begin : g_en4
            assign rd_en[R_DELTA + ge] = !ready_delta[ge];
            assign rd_en[R_ATTN + ge]  = !ready_attn[ge];
        end
    endgenerate

    wire [127:0] mem_rd_hi, mem_wr_hi;               // the memory unit's second word: the buffer's wide port
    reg  [3:0]      first_r;
    reg  [4*21-1:0] slot_r;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin first_r <= 4'd0; slot_r <= 0; end
        else if (start) begin first_r <= first; slot_r <= slot_page; end
    wire [15:0]  mem_wr_hi_be;
    fabric_vb #(.BYTES(VB_BYTES), .NR(NR), .NW(NW), .AW(AW), .NB(VB_BANKS), .BSH(VB_BANK_SHIFT),
                .RCAP2(VB_RCAP2), .RCAP3(VB_RCAP3), .WCAP2(VB_WCAP2),
                .NPR(VB_NPR), .NPW(VB_NPW), .RMAP0(VB_RMAP0), .RMAP1(VB_RMAP1),
                .WMAP0(VB_WMAP0), .WMAP1(VB_WMAP1), .WIDE_R(R_MEM), .WIDE_W(W_MEM), .INIT_FILE(VB_FILE)) u_vb (
        .clk(clk), .rd_en(rd_en), .rd_addr(rd_addr), .rd_data(rd_data), .rd_hi(mem_rd_hi), .wr_en(wr_en), .wr_addr(wr_addr),
        .wr_data(wr_data), .wr_be(wr_be), .wr_hi(mem_wr_hi), .wr_hi_be(mem_wr_hi_be));

    wire [NE-1:0] ready_norm, ready_delta, ready_rotary, ready_attn;
    wire ready_tiles, ready_conv, ready_gates, ready_swiglu, ready_residual, ready_mem;
    assign cmd_ready[U_TILES]    = (cmd_engine == 0) && ready_tiles;
    assign cmd_ready[U_NORM]     = (cmd_engine < 2) && ready_norm[cmd_engine];
    assign cmd_ready[U_CONV]     = (cmd_engine == 0) && ready_conv;
    assign cmd_ready[U_GATES]    = (cmd_engine == 0) && ready_gates;
    assign cmd_ready[U_DELTA]    = ready_delta[cmd_engine];
    assign cmd_ready[U_SWIGLU]   = (cmd_engine == 0) && ready_swiglu;
    assign cmd_ready[U_RESIDUAL] = (cmd_engine == 0) && ready_residual;
    assign cmd_ready[U_ROTARY]   = (cmd_engine < 2) && ready_rotary[cmd_engine];
    assign cmd_ready[U_ATTN]     = ready_attn[cmd_engine];
    assign cmd_ready[U_MEM]      = (cmd_engine == 0) && ready_mem;
    assign done_valid[U_TILES*NE + 1 +: 3] = 0;    assign done_tag[(U_TILES*NE + 1)*8 +: 24] = 0;
    assign done_valid[U_NORM*NE + 2 +: 2] = 0;     assign done_tag[(U_NORM*NE + 2)*8 +: 16] = 0;
    assign done_valid[U_CONV*NE + 1 +: 3] = 0;     assign done_tag[(U_CONV*NE + 1)*8 +: 24] = 0;
    assign done_valid[U_GATES*NE + 1 +: 3] = 0;    assign done_tag[(U_GATES*NE + 1)*8 +: 24] = 0;
    assign done_valid[U_SWIGLU*NE + 1 +: 3] = 0;   assign done_tag[(U_SWIGLU*NE + 1)*8 +: 24] = 0;
    assign done_valid[U_RESIDUAL*NE + 1 +: 3] = 0; assign done_tag[(U_RESIDUAL*NE + 1)*8 +: 24] = 0;
    assign done_valid[U_ROTARY*NE + 2 +: 2] = 0;   assign done_tag[(U_ROTARY*NE + 2)*8 +: 16] = 0;
    assign done_valid[U_MEM*NE + 1 +: 3] = 0;      assign done_tag[(U_MEM*NE + 1)*8 +: 24] = 0;
    assign ready_norm[3:2] = 0;
    assign ready_rotary[3:2] = 0;

    genvar e;
    generate
        for (e = 0; e < 2; e = e + 1) begin : g_norm
            fabric_norm_adapter #(.NL(NL), .DMAX(D), .SW(SW), .NCONST(4), .AW(AW), .LUT_DIR(LUT_DIR)) u (
                .clk(clk), .rst_n(rst_n), .cmd_valid(cmd_valid[U_NORM] && cmd_engine == e), .cmd_len(cmd_len), .cmd_src(cmd_src),
                .cmd_dst(cmd_dst), .cmd_a2(cmd_a2), .cmd_a3(cmd_a3), .cmd_arg(cmd_arg), .cmd_tag(cmd_tag), .cmd_ready(ready_norm[e]),
                .done_valid(done_valid[U_NORM*NE + e]), .done_tag(done_tag[(U_NORM*NE + e)*8 +: 8]),
                .rd_addr_x(rd_addr[(R_NORM + 2*e)*AW +: AW]), .rd_addr_g(rd_addr[(R_NORM + 2*e + 1)*AW +: AW]), .rd_en_g(norm_gain_en[e]),
                .rd_data_x(rd_data[(R_NORM + 2*e)*128 +: 128]), .rd_data_g(rd_data[(R_NORM + 2*e + 1)*128 +: 128]),
                .wr_en(wr_en[W_NORM + e]), .wr_addr(wr_addr[(W_NORM + e)*AW +: AW]), .wr_data(wr_data[(W_NORM + e)*128 +: 128]),
                .wr_be(wr_be[(W_NORM + e)*16 +: 16]));
        end
        for (e = 0; e < NE; e = e + 1) begin : g_delta
            fabric_delta_adapter #(.K(HK), .V(HV), .YSH(YSH), .AW(AW)) u (
                .clk(clk), .rst_n(rst_n), .cmd_valid(cmd_valid[U_DELTA] && cmd_engine == e), .cmd_src(cmd_src), .cmd_dst(cmd_dst),
                .cmd_a2(cmd_a2), .cmd_a3(cmd_a3), .cmd_arg(cmd_arg), .cmd_tag(cmd_tag), .cmd_ready(ready_delta[e]),
                .done_valid(done_valid[U_DELTA*NE + e]), .done_tag(done_tag[(U_DELTA*NE + e)*8 +: 8]),
                .rd_addr(rd_addr[(R_DELTA + e)*AW +: AW]), .rd_data(rd_data[(R_DELTA + e)*128 +: 128]),
                .wr_en(wr_en[W_DELTA + e]), .wr_addr(wr_addr[(W_DELTA + e)*AW +: AW]), .wr_data(wr_data[(W_DELTA + e)*128 +: 128]),
                .wr_be(wr_be[(W_DELTA + e)*16 +: 16]));
        end
        for (e = 0; e < 2; e = e + 1) begin : g_rotary
            fabric_rotary_adapter #(.HD(HD), .R(RD), .NL(ATT_L), .SW(SW), .AW(AW), .LUT_DIR(LUT_DIR)) u (
                .clk(clk), .rst_n(rst_n), .cmd_valid(cmd_valid[U_ROTARY] && cmd_engine == e), .cmd_src(cmd_src), .cmd_dst(cmd_dst),
                .cmd_a2(cmd_a2), .cmd_a3(cmd_a3), .cmd_arg(cmd_arg), .cmd_tag(cmd_tag), .cmd_ready(ready_rotary[e]),
                .done_valid(done_valid[U_ROTARY*NE + e]), .done_tag(done_tag[(U_ROTARY*NE + e)*8 +: 8]),
                .rd_addr(rd_addr[(R_ROTARY + e)*AW +: AW]), .rd_data(rd_data[(R_ROTARY + e)*128 +: 128]),
                .wr_en(wr_en[W_ROTARY + e]), .wr_addr(wr_addr[(W_ROTARY + e)*AW +: AW]), .wr_data(wr_data[(W_ROTARY + e)*128 +: 128]),
                .wr_be(wr_be[(W_ROTARY + e)*16 +: 16]));
        end
        for (e = 0; e < NE; e = e + 1) begin : g_attn
            fabric_attn_adapter #(.HD(HD), .G(GROUP), .L(ATT_L), .AW(AW), .LUT_DIR(LUT_DIR)) u (
                .clk(clk), .rst_n(rst_n), .cmd_valid(cmd_valid[U_ATTN] && cmd_engine == e), .cmd_len(cmd_len), .cmd_src(cmd_src),
                .cmd_dst(cmd_dst), .cmd_a2(cmd_a2), .cmd_a3(cmd_a3), .cmd_arg(cmd_arg), .cmd_tag(cmd_tag), .cmd_ready(ready_attn[e]),
                .done_valid(done_valid[U_ATTN*NE + e]), .done_tag(done_tag[(U_ATTN*NE + e)*8 +: 8]),
                .rd_addr(rd_addr[(R_ATTN + e)*AW +: AW]), .rd_data(rd_data[(R_ATTN + e)*128 +: 128]),
                .wr_en(wr_en[W_ATTN + e]), .wr_addr(wr_addr[(W_ATTN + e)*AW +: AW]), .wr_data(wr_data[(W_ATTN + e)*128 +: 128]),
                .wr_be(wr_be[(W_ATTN + e)*16 +: 16]));
        end
    endgenerate

    fabric_pass_adapter #(.NT(NT), .ROWS(ROWS), .COLS(COLS), .WB(WB), .AB(8), .P(P), .ACC(ACC), .SB(SB), .SHB(SHB), .TMAX(TMAX),
                          .MODEL_TILES(MODEL_TILES), .AW(AW)) u_tiles (
        .clk(clk), .rst_n(rst_n), .cmd_valid(cmd_valid[U_TILES] && cmd_engine == 0), .cmd_len(cmd_len), .cmd_src(cmd_src), .cmd_dst(cmd_dst),
        .cmd_a2(cmd_a2), .cmd_a3(cmd_a3), .cmd_arg(cmd_arg), .cmd_tag(cmd_tag), .cmd_ready(ready_tiles), .done_valid(done_valid[U_TILES*NE]), .done_tag(done_tag[U_TILES*NE*8 +: 8]),
        .rd_addr(rd_addr[R_TILES*AW +: TMAX*AW]), .rd_data(rd_data[R_TILES*128 +: TMAX*128]),
        .wr_en(wr_en[W_TILES]), .wr_addr(wr_addr[W_TILES*AW +: AW]), .wr_data(wr_data[W_TILES*128 +: 128]), .wr_be(wr_be[W_TILES*16 +: 16]));

    fabric_conv_adapter #(.CL(CL), .KK(KK), .C(CONV), .AW(AW), .LUT_DIR(LUT_DIR)) u_conv (
        .clk(clk), .rst_n(rst_n), .cmd_valid(cmd_valid[U_CONV] && cmd_engine == 0), .cmd_len(cmd_len), .cmd_src(cmd_src), .cmd_dst(cmd_dst),
        .cmd_a2(cmd_a2), .cmd_a3(cmd_a3), .cmd_arg(cmd_arg), .cmd_tag(cmd_tag), .cmd_ready(ready_conv), .done_valid(done_valid[U_CONV*NE]),
        .done_tag(done_tag[U_CONV*NE*8 +: 8]),
        .rd_addr_x(rd_addr[R_CONV*AW +: AW]), .rd_addr_h(rd_addr[(R_CONV+1)*AW +: AW]),
        .rd_data_x(rd_data[R_CONV*128 +: 128]), .rd_data_h(rd_data[(R_CONV+1)*128 +: 128]),
        .wr_en_y(wr_en[W_CONV]), .wr_addr_y(wr_addr[W_CONV*AW +: AW]), .wr_data_y(wr_data[W_CONV*128 +: 128]), .wr_be_y(wr_be[W_CONV*16 +: 16]),
        .wr_en_h(wr_en[W_CONV+1]), .wr_addr_h(wr_addr[(W_CONV+1)*AW +: AW]), .wr_data_h(wr_data[(W_CONV+1)*128 +: 128]), .wr_be_h(wr_be[(W_CONV+1)*16 +: 16]));

    fabric_gates_adapter #(.NVMAX(NV), .ACC(ACC), .AW(AW), .LUT_DIR(LUT_DIR)) u_gates (
        .clk(clk), .rst_n(rst_n), .cmd_valid(cmd_valid[U_GATES] && cmd_engine == 0), .cmd_len(cmd_len), .cmd_src(cmd_src), .cmd_dst(cmd_dst),
        .cmd_a2(cmd_a2), .cmd_a3(cmd_a3), .cmd_arg(cmd_arg), .cmd_tag(cmd_tag), .cmd_ready(ready_gates), .done_valid(done_valid[U_GATES*NE]), .done_tag(done_tag[U_GATES*NE*8 +: 8]),
        .rd_addr_b(rd_addr[R_GATES*AW +: AW]), .rd_addr_a(rd_addr[(R_GATES+1)*AW +: AW]),
        .rd_data_b(rd_data[R_GATES*128 +: 128]), .rd_data_a(rd_data[(R_GATES+1)*128 +: 128]),
        .wr_en(wr_en[W_GATES]), .wr_addr(wr_addr[W_GATES*AW +: AW]), .wr_data(wr_data[W_GATES*128 +: 128]), .wr_be(wr_be[W_GATES*16 +: 16]));

    fabric_swiglu_adapter #(.NL(SW_L), .AW(AW), .LUT_DIR(LUT_DIR)) u_swiglu (
        .clk(clk), .rst_n(rst_n), .cmd_valid(cmd_valid[U_SWIGLU] && cmd_engine == 0), .cmd_len(cmd_len), .cmd_src(cmd_src), .cmd_dst(cmd_dst),
        .cmd_a2(cmd_a2), .cmd_a3(cmd_a3), .cmd_arg(cmd_arg), .cmd_tag(cmd_tag), .cmd_ready(ready_swiglu), .done_valid(done_valid[U_SWIGLU*NE]), .done_tag(done_tag[U_SWIGLU*NE*8 +: 8]),
        .rd_addr_g(rd_addr[R_SWIGLU*AW +: AW]), .rd_addr_u(rd_addr[(R_SWIGLU+1)*AW +: AW]),
        .rd_data_g(rd_data[R_SWIGLU*128 +: 128]), .rd_data_u(rd_data[(R_SWIGLU+1)*128 +: 128]),
        .wr_en(wr_en[W_SWIGLU]), .wr_addr(wr_addr[W_SWIGLU*AW +: AW]), .wr_data(wr_data[W_SWIGLU*128 +: 128]), .wr_be(wr_be[W_SWIGLU*16 +: 16]));

    fabric_residual_adapter #(.NL(NL), .AW(AW)) u_residual (
        .clk(clk), .rst_n(rst_n), .cmd_valid(cmd_valid[U_RESIDUAL] && cmd_engine == 0), .cmd_len(cmd_len), .cmd_src(cmd_src), .cmd_dst(cmd_dst),
        .cmd_a2(cmd_a2), .cmd_a3(cmd_a3), .cmd_arg(cmd_arg), .cmd_tag(cmd_tag), .cmd_ready(ready_residual), .done_valid(done_valid[U_RESIDUAL*NE]), .done_tag(done_tag[U_RESIDUAL*NE*8 +: 8]),
        .rd_addr_h(rd_addr[R_RESIDUAL*AW +: AW]), .rd_addr_y(rd_addr[(R_RESIDUAL+1)*AW +: AW]),
        .rd_data_h(rd_data[R_RESIDUAL*128 +: 128]), .rd_data_y(rd_data[(R_RESIDUAL+1)*128 +: 128]),
        .wr_en(wr_en[W_RESIDUAL]), .wr_addr(wr_addr[W_RESIDUAL*AW +: AW]), .wr_data(wr_data[W_RESIDUAL*128 +: 128]), .wr_be(wr_be[W_RESIDUAL*16 +: 16]));

    fabric_mem_unit #(.HD(HD), .NKV(NKV), .IDIM(IDIM), .BS(BS), .W(W), .TOP(TOP), .KV_BITS(KV_BITS), .L(ATT_L), .REC_BYTES(REC_BYTES),
                      .RPB(RPB), .MAXR(MAXR), .WINDOW_OFF(WINDOW_OFF), .BLOCK_OFF(BLOCK_OFF), .INDEX_OFF(INDEX_OFF), .SUMS_OFF(SUMS_OFF),
                      .AW(AW), .LUT_DIR(LUT_DIR)) u_mem (
        .clk(clk), .rst_n(rst_n), .slot_page(slot_r), .first(first_r), .cmd_valid(cmd_valid[U_MEM] && cmd_engine == 0), .cmd_len(cmd_len), .cmd_src(cmd_src), .cmd_dst(cmd_dst),
        .cmd_a2(cmd_a2), .cmd_a3(cmd_a3), .cmd_arg(cmd_arg), .cmd_tag(cmd_tag), .cmd_ready(ready_mem), .done_valid(done_valid[U_MEM*NE]), .done_tag(done_tag[U_MEM*NE*8 +: 8]),
        .rd_addr(rd_addr[R_MEM*AW +: AW]), .rd_en(rd_en[R_MEM]), .rd_data(rd_data[R_MEM*128 +: 128]), .rd_hi(mem_rd_hi),
        .wr_en(wr_en[W_MEM]), .wr_addr(wr_addr[W_MEM*AW +: AW]), .wr_data(wr_data[W_MEM*128 +: 128]), .wr_be(wr_be[W_MEM*16 +: 16]),
        .wr_hi(mem_wr_hi), .wr_hi_be(mem_wr_hi_be),
        .m_req_valid(m_req_valid), .m_req_ready(m_req_ready), .m_req_write(m_req_write), .m_req_wide(m_req_wide), .m_req_addr(m_req_addr),
        .m_req_beats(m_req_beats),
        .m_wdata_valid(m_wdata_valid), .m_wdata_ready(m_wdata_ready), .m_wdata(m_wdata), .m_rdata_valid(m_rdata_valid), .m_rdata(m_rdata));
endmodule

`default_nettype wire
