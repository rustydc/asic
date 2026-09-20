// Fixed-weight fabric tile: via-programmed coefficient ROM feeding COLS
// multiply-accumulate columns.
//
// One pass processes ROWS activations, P per cycle, and leaves COLS partial
// sums in the accumulators.  Coefficients are symmetric signed WB-bit values;
// every magnitude 0..7 is a signed pair of power-of-two shifts, so the column
// selects two taps of the shifted activation per bank and sums them in
// carry-save form.  No carry chain runs per cycle anywhere in the column;
// carries resolve once per pass in the shared requantizer walk.
//
// With T > 1 a pass carries T tokens of one context (chunked prefill): the
// activation bus holds P rows of each token, the ROM word and its tap
// decode are read once and serve T accumulators per column, and the
// requantizer walks the T x COLS results token by token.
//
// The tile is split at the hard-macro boundary:
//   fabric_rom      the via-programmed ROM (here a constant array from a hex
//                   image; in silicon a ROM macro with P banks)
//   fabric_columns  the synthesizable column datapath and pass control
//   fabric_tile     the two wired together
// Synthesis of fabric_columns alone measures the MAC-column area and timing
// that the density model needs.
//
// Golden model: fabric/tile.py (tile_forward, requantize).

`default_nettype none

// ---------------------------------------------------------------------------
// ROM stand-in: P words per cycle, bank b holds rows congruent to b mod P.
// ---------------------------------------------------------------------------
module fabric_rom #(
    parameter int ROWS = 4096,
    parameter int COLS = 64,
    parameter int WB   = 4,
    parameter int P    = 2,
    parameter     ROM_FILE = ""
) (
    input  wire [$clog2(ROWS/P)-1:0] cycle,     // which group of P rows
    output wire [P*COLS*WB-1:0]      words      // bank b at words[b*COLS*WB +: COLS*WB]
);
    localparam int ROWW = COLS * WB;
    reg [ROWW-1:0] rom [0:ROWS-1];
    initial begin
        if (ROM_FILE != "") $readmemh(ROM_FILE, rom);
    end
    genvar b;
    generate
        for (b = 0; b < P; b = b + 1) begin : g_bank
            assign words[b*ROWW +: ROWW] = rom[cycle * P + b];
        end
    endgenerate
endmodule

// ---------------------------------------------------------------------------
// One registered copy of a control strobe.  Kept as its own hierarchy so
// synthesis cannot merge the copies back into a single high-fanout net.
// ---------------------------------------------------------------------------
(* keep_hierarchy *)
module fabric_strobe_copy (
    input  wire clk,
    input  wire rst_n,
    input  wire d,
    output reg  q
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) q <= 1'b0;
        else        q <= d;
    end
endmodule

// ---------------------------------------------------------------------------
// Carry-save reduction of N operands to a (sum, carry) pair: value = s + 2c.
// Each layer turns groups of three operands into two; depth is logarithmic.
// ---------------------------------------------------------------------------
module fabric_csa_tree #(
    parameter int N = 3,
    parameter int W = 41
) (
    input  wire [N*W-1:0] ops,
    output wire [W-1:0]   s,
    output wire [W-1:0]   c
);
    generate
        if (N == 1) begin : g_one
            assign s = ops[W-1:0];
            assign c = {W{1'b0}};
        end else if (N == 2) begin : g_two
            wire [W-1:0] a = ops[W-1:0];
            wire [W-1:0] b = ops[2*W-1:W];
            assign s = a ^ b;
            assign c = a & b;
        end else begin : g_layer
            localparam int G = N / 3;
            localparam int R = N % 3;
            localparam int M = 2 * G + R;
            wire [M*W-1:0] next;
            genvar i;
            for (i = 0; i < G; i = i + 1) begin : g_csa
                wire [W-1:0] a = ops[(3*i)*W +: W];
                wire [W-1:0] b = ops[(3*i+1)*W +: W];
                wire [W-1:0] d = ops[(3*i+2)*W +: W];
                wire [W-1:0] cy = (a & b) | (a & d) | (b & d);
                assign next[(2*i)*W +: W]   = a ^ b ^ d;
                assign next[(2*i+1)*W +: W] = {cy[W-2:0], 1'b0};   // carry at weight 2, as a plain operand
            end
            for (i = 0; i < R; i = i + 1) begin : g_pass
                assign next[(2*G+i)*W +: W] = ops[(3*G+i)*W +: W];
            end
            fabric_csa_tree #(.N(M), .W(W)) sub (.ops(next), .s(s), .c(c));
        end
    endgenerate
endmodule

// ---------------------------------------------------------------------------
// Column datapath and pass control.
// ---------------------------------------------------------------------------
module fabric_columns #(
    parameter int ROWS = 4096,          // activations per pass
    parameter int COLS = 64,            // outputs per tile
    parameter int WB   = 4,             // coefficient bits (symmetric signed)
    parameter int AB   = 8,             // activation and output bits (signed)
    parameter int P    = 2,             // rows consumed per cycle
    parameter int ACC  = 24,            // accumulator bits
    parameter int SB   = 16,            // requantization multiplier bits (unsigned)
    parameter int SHB  = 5,             // requantization shift bits
    parameter int T    = 1              // tokens per pass
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,      // begin a pass: load psum_in into the accumulators
    input  wire [T*COLS*ACC-1:0]    psum_in,    // chained partial sums (zero for a fresh pass), token-major
    input  wire                     x_valid,    // P activations of each token for rows cycle*P .. cycle*P+P-1
    input  wire [T*P*AB-1:0]        x_data,     // token t, row b at (t*P + b)*AB
    input  wire [P*COLS*WB-1:0]     rom_words,  // coefficient words for the same P rows
    input  wire [COLS*SB-1:0]       mult,       // per-column requantization multiplier
    input  wire [COLS*SHB-1:0]      shift,      // per-column requantization shift
    output wire [$clog2(ROWS/P)-1:0] cycle,     // ROM address (row group)
    output wire                     x_ready,    // high while a pass is consuming activations
    output reg                      done,       // one-cycle pulse after the last rows are accumulated
    output wire [T*COLS*ACC-1:0]    psum_out,   // resolved accumulators, valid when q_valid, token-major
    output wire [T*COLS*AB-1:0]     q_out,      // requantized outputs, valid when q_valid, token-major
    output wire                     q_valid     // pulses a few cycles after the T*COLS-cycle walk
);
    /*verilator hier_block*/           // one compiled block for every tile under Verilator --hierarchical
    localparam int CYCLES = ROWS / P;
    localparam int CW     = $clog2(CYCLES);
    localparam int ROWW   = COLS * WB;
    localparam int NA     = T * COLS;   // accumulators

    // ------------------------------------------------------------------
    // Pass control.  Activations are consumed at the input rate; behind
    // them sit stage A (inputs registered), stage B (term reduction) and
    // stage C (accumulate), so `done` follows the last consumed rows by two
    // cycles.  A new `start` must not be issued until `done` has been seen,
    // and activations may not be presented until the cycle after `start`.
    // ------------------------------------------------------------------
    reg          busy;
    reg [CW-1:0] cycle_r;
    wire         consume = busy && x_valid;
    wire         last    = (cycle_r == CYCLES - 1);
    reg          vA, vB, lastA, lastB;

    assign x_ready = busy;
    assign cycle   = cycle_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy    <= 1'b0;
            cycle_r <= {CW{1'b0}};
            done    <= 1'b0;
            vA <= 1'b0; vB <= 1'b0; lastA <= 1'b0; lastB <= 1'b0;
        end else begin
            if (start) begin
                busy    <= 1'b1;
                cycle_r <= {CW{1'b0}};
            end else if (consume) begin
                cycle_r <= cycle_r + 1'b1;
                if (last) busy <= 1'b0;
            end
            vA    <= consume;
            lastA <= consume && last;
            vB    <= vA;
            lastB <= lastA;
            done  <= lastB;
        end
    end

    // ------------------------------------------------------------------
    // Stage A: register the activations and the ROM words every cycle (no
    // enable, so no control net fans out to the datapath).  No arithmetic:
    // every coefficient magnitude 0..7 is a signed pair of power-of-two
    // shifts, so the multiples are wiring and the adds happen in the column.
    //   1 = 1      2 = 2      3 = 2 + 1    4 = 4
    //   5 = 4 + 1  6 = 4 + 2  7 = 8 - 1
    // The hard macro does the same: the via ROM word selects two taps of
    // the shifted activation bus per bank.
    // ------------------------------------------------------------------
    reg  [T*P*AB-1:0] x_a;
    reg  [P*ROWW-1:0] words_a;
    always @(posedge clk) begin
        x_a     <= x_data;
        words_a <= rom_words;
    end
    genvar b, c, g, t;

    // ------------------------------------------------------------------
    // Columns.  Stage B: select two taps per bank by ROM word, one's
    // complement, and reduce the 2P terms plus the negation count to a
    // carry-save pair at accumulator width.  Stage C: carry-save accumulate
    // of that pair into the (sum, carry) accumulator.  Value = s + 2c mod
    // 2^ACC throughout, so widths never change and no wrap is possible.
    //
    // The accumulate enable and the load strobe are registered once per
    // column so no single net drives every accumulator (2*ACC + 1 loads).
    // ------------------------------------------------------------------
    reg [ACC-1:0] acc_s [0:NA-1];         // accumulator sum vector, token t column c at t*COLS + c
    reg [ACC-1:0] acc_c [0:NA-1];         // accumulator carry vector (weight 2)
    wire [COLS-1:0] en_g;
    wire [COLS-1:0] load_g;
    generate
        for (g = 0; g < COLS; g = g + 1) begin : g_en
            fabric_strobe_copy u_en   (.clk(clk), .rst_n(rst_n), .d(vA),    .q(en_g[g]));
            fabric_strobe_copy u_load (.clk(clk), .rst_n(rst_n), .d(start), .q(load_g[g]));
        end
    endgenerate

    localparam int TW = AB + WB;          // 8x of an AB-bit value, with room for negation
    localparam int NT = 2 * P + 1;        // terms per cycle plus the negation count
    localparam int NCW = $clog2(NT);

    generate
        for (c = 0; c < COLS; c = c + 1) begin : g_col
            // The tap decode of each bank's coefficient, once per column.
            wire [P*2-1:0] shA_b, shB_b;
            wire [P-1:0]   enA_b, enB_b, nA_b, nB_b;
            for (b = 0; b < P; b = b + 1) begin : g_dec
                wire signed [WB-1:0] word = words_a[b*ROWW + c*WB +: WB];
                wire                 neg  = word[WB-1];
                wire signed [WB:0]   wext = {word[WB-1], word};
                wire [WB-1:0]        mag  = neg ? (-wext) : wext;      // 0..7
                // Signed-digit decode of the magnitude: tap A, tap B, sign of B.
                reg  [1:0] shA;   reg enA;
                reg  [1:0] shB;   reg enB;   reg negB;
                always @* begin
                    shA = 2'd0; enA = 1'b0; shB = 2'd0; enB = 1'b0; negB = 1'b0;
                    case (mag[2:0])
                        3'd1: begin enA = 1'b1; shA = 2'd0; end
                        3'd2: begin enA = 1'b1; shA = 2'd1; end
                        3'd3: begin enA = 1'b1; shA = 2'd1; enB = 1'b1; shB = 2'd0; end
                        3'd4: begin enA = 1'b1; shA = 2'd2; end
                        3'd5: begin enA = 1'b1; shA = 2'd2; enB = 1'b1; shB = 2'd0; end
                        3'd6: begin enA = 1'b1; shA = 2'd2; enB = 1'b1; shB = 2'd1; end
                        3'd7: begin enA = 1'b1; shA = 2'd3; enB = 1'b1; shB = 2'd0; negB = 1'b1; end
                        default: ;
                    endcase
                end
                assign shA_b[b*2 +: 2] = shA;
                assign shB_b[b*2 +: 2] = shB;
                assign enA_b[b] = enA;
                assign enB_b[b] = enB;
                assign nA_b[b]  = neg;
                assign nB_b[b]  = neg ^ negB;
            end
            // Negation as one's complement; the +1s are counted and enter
            // the reduction as one more operand, so no incrementer chains.
            // The count depends on the coefficients only, so it is shared by the tokens.
            reg [NCW-1:0] negcount;
            integer nb;
            always @* begin
                negcount = {NCW{1'b0}};
                for (nb = 0; nb < P; nb = nb + 1) negcount = negcount + nA_b[nb] + nB_b[nb];
            end
            for (t = 0; t < T; t = t + 1) begin : g_tok
                wire [NT*ACC-1:0] b_ops;
                for (b = 0; b < P; b = b + 1) begin : g_term
                    wire signed [AB-1:0] xb   = x_a[(t*P + b)*AB +: AB];
                    wire signed [TW-1:0] xext = {{(TW-AB){xb[AB-1]}}, xb};
                    wire signed [TW-1:0] tapA = enA_b[b] ? (xext <<< shA_b[b*2 +: 2]) : {TW{1'b0}};
                    wire signed [TW-1:0] tapB = enB_b[b] ? (xext <<< shB_b[b*2 +: 2]) : {TW{1'b0}};
                    wire signed [TW-1:0] tA   = nA_b[b] ? ~tapA : tapA;
                    wire signed [TW-1:0] tB   = nB_b[b] ? ~tapB : tapB;
                    assign b_ops[(2*b)*ACC +: ACC]   = {{(ACC-TW){tA[TW-1]}}, tA};
                    assign b_ops[(2*b+1)*ACC +: ACC] = {{(ACC-TW){tB[TW-1]}}, tB};
                end
                assign b_ops[(2*P)*ACC +: ACC] = {{(ACC-NCW){1'b0}}, negcount};

                // Stage B: reduce to a carry-save pair, registered.
                wire [ACC-1:0] bs_w, bc_w;
                fabric_csa_tree #(.N(NT), .W(ACC)) u_b_tree (.ops(b_ops), .s(bs_w), .c(bc_w));
                reg  [ACC-1:0] sb_s, sb_c;
                always @(posedge clk) begin
                    sb_s <= bs_w;
                    sb_c <= bc_w;
                end

                // Stage C: carry-save accumulate of four operands.  Datapath
                // registers carry no reset: the registered `start` loads the
                // accumulators one cycle after it is seen, two cycles before the
                // first accumulate can arrive.
                localparam int A = t * COLS + c;
                wire [4*ACC-1:0] c_ops = {{sb_c[ACC-2:0], 1'b0}, sb_s, {acc_c[A][ACC-2:0], 1'b0}, acc_s[A]};
                wire [ACC-1:0]   cs_w, cc_w;
                fabric_csa_tree #(.N(4), .W(ACC)) u_c_tree (.ops(c_ops), .s(cs_w), .c(cc_w));
                always @(posedge clk) begin
                    if (load_g[c]) begin
                        acc_s[A] <= psum_in[A*ACC +: ACC];
                        acc_c[A] <= {ACC{1'b0}};
                    end else if (en_g[c]) begin
                        acc_s[A] <= cs_w;
                        acc_c[A] <= cc_w;
                    end
                end
            end
        end
    endgenerate

    // ------------------------------------------------------------------
    // Shared requantizer: walks the accumulators after the pass, one per
    // cycle (token by token), computing sat_AB((acc * mult + 2^(shift-1)) >>> shift):
    //   S0  one-hot select of the column's carry-save pair and constants
    //   R1  carry resolve, low half        R2  carry resolve, high half
    //   M   carry-save multiply with the rounding constant folded in
    //   A1..A3  three-chunk carry-propagate add of the product pair
    //   Q   arithmetic shift, saturate, write
    // Every carry chain is at most ACC/2 or PW/3 bits long.
    // ------------------------------------------------------------------
    localparam int QW  = $clog2(NA) + (NA == 1);
    localparam int PW  = ACC + SB + 1;
    localparam int RL  = ACC / 2;                 // resolve chunk
    localparam int C0  = PW / 3;                  // product add chunks
    localparam int C1  = PW / 3;
    localparam int C2  = PW - C0 - C1;

    reg              q_busy;
    reg  [QW-1:0]    q_col;
    reg  [NA-1:0]    sel_oh;
    reg  [NA*AB-1:0]  q_reg;
    reg  [NA*ACC-1:0] psum_reg;
    reg              q_valid_r;
    assign q_out    = q_reg;
    assign psum_out = psum_reg;
    assign q_valid  = q_valid_r;

    // One-hot walking select: each bit drives only its own column's AND
    // gates, so the select fanout is fixed at 2*ACC+SB+SHB regardless of COLS.
    reg  [ACC-1:0]   accs_sel, accc_sel;
    reg  [SB-1:0]    m_sel;
    reg  [SHB-1:0]   s_sel;
    integer sc;
    always @* begin
        accs_sel = {ACC{1'b0}};
        accc_sel = {ACC{1'b0}};
        m_sel    = {SB{1'b0}};
        s_sel    = {SHB{1'b0}};
        for (sc = 0; sc < NA; sc = sc + 1) begin
            accs_sel = accs_sel | (acc_s[sc] & {ACC{sel_oh[sc]}});
            accc_sel = accc_sel | (acc_c[sc] & {ACC{sel_oh[sc]}});
            m_sel    = m_sel    | (mult[(sc % COLS)*SB +: SB] & {SB{sel_oh[sc]}});
            s_sel    = s_sel    | (shift[(sc % COLS)*SHB +: SHB] & {SHB{sel_oh[sc]}});
        end
    end

    // Pipeline registers.
    reg              vS0, vR1, vR2, vM, vA1, vA2, vA3;
    reg  [QW-1:0]    colS0, colR1, colR2, colM, colA1, colA2, colA3;
    reg  [SHB-1:0]   sS0, sR1, sR2, sM, sA1, sA2, sA3;
    reg  [SB-1:0]    mS0, mR1, mR2;
    reg  [ACC-1:0]   ssS0, scS0;                  // selected sum and shifted carry
    reg  [RL:0]      rloR1;                       // low resolve with carry out
    reg  [ACC-1:RL]  rhsR1, rhcR1;
    reg  signed [ACC-1:0] accR2;                  // resolved accumulator
    reg  [PW-1:0]    psM, pcM;                    // product pair, value = ps + 2 pc
    reg  [C0:0]      a0A1;                        // chunk 0 with carry out
    reg  [PW-1:C0]   ahA1, bhA1;
    reg  [C0+C1:0]   a01A2;                       // chunks 0..1 with carry out
    reg  [PW-1:C0+C1] ahA2, bhA2;
    reg  signed [PW-1:0] prodA3;

    wire signed [PW-1:0] one    = {{(PW-1){1'b0}}, 1'b1};
    wire signed [PW-1:0] rndR2  = (sR2 == 0) ? {PW{1'b0}} : (one <<< (sR2 - 1));
    wire signed [PW-1:0] accR2x = {{(PW-ACC){accR2[ACC-1]}}, accR2};
    wire [(SB+1)*PW-1:0] pp_ops;
    genvar pb;
    generate
        for (pb = 0; pb < SB; pb = pb + 1) begin : g_pp
            assign pp_ops[pb*PW +: PW] = mR2[pb] ? (accR2x <<< pb) : {PW{1'b0}};
        end
    endgenerate
    assign pp_ops[SB*PW +: PW] = rndR2;
    wire [PW-1:0] ps_w, pc_w;
    fabric_csa_tree #(.N(SB + 1), .W(PW)) u_mul_tree (.ops(pp_ops), .s(ps_w), .c(pc_w));
    wire [PW-1:0] pcsM = {pcM[PW-2:0], 1'b0};

    wire signed [PW-1:0] shr   = prodA3 >>> sA3;
    wire signed [PW-1:0] qmax  = (one <<< (AB - 1)) - one;
    wire signed [PW-1:0] qmin  = -(one <<< (AB - 1));
    wire [AB-1:0]        q_sat = (shr > qmax) ? qmax[AB-1:0] :
                                 (shr < qmin) ? qmin[AB-1:0] : shr[AB-1:0];

    // Control: reset.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q_busy    <= 1'b0;
            q_col     <= {QW{1'b0}};
            sel_oh    <= {COLS{1'b0}};
            q_valid_r <= 1'b0;
            vS0 <= 1'b0; vR1 <= 1'b0; vR2 <= 1'b0; vM <= 1'b0;
            vA1 <= 1'b0; vA2 <= 1'b0; vA3 <= 1'b0;
        end else begin
            if (done) begin
                q_busy <= 1'b1;
                q_col  <= {QW{1'b0}};
                sel_oh <= {{(NA-1){1'b0}}, 1'b1};
            end else if (q_busy) begin
                if (q_col == NA - 1) q_busy <= 1'b0;
                q_col  <= q_col + 1'b1;
                sel_oh <= {sel_oh[NA-2:0], 1'b0};
            end
            vS0 <= q_busy;
            vR1 <= vS0;
            vR2 <= vR1;
            vM  <= vR2;
            vA1 <= vM;
            vA2 <= vA1;
            vA3 <= vA2;
            q_valid_r <= vA3 && (colA3 == NA - 1);
        end
    end

    // Datapath: no reset.
    integer qi, pi;
    always @(posedge clk) begin
        // S0: select.
        colS0 <= q_col;
        ssS0  <= accs_sel;
        scS0  <= {accc_sel[ACC-2:0], 1'b0};
        mS0   <= m_sel;
        sS0   <= s_sel;
        // R1: resolve low half.
        colR1 <= colS0;
        rloR1 <= {1'b0, ssS0[RL-1:0]} + {1'b0, scS0[RL-1:0]};
        rhsR1 <= ssS0[ACC-1:RL];
        rhcR1 <= scS0[ACC-1:RL];
        mR1   <= mS0;
        sR1   <= sS0;
        // R2: resolve high half.
        colR2 <= colR1;
        accR2 <= {rhsR1 + rhcR1 + rloR1[RL], rloR1[RL-1:0]};
        mR2   <= mR1;
        sR2   <= sR1;
        // M: carry-save multiply; publish the resolved sum for chaining.
        colM  <= colR2;
        psM   <= ps_w;
        pcM   <= pc_w;
        sM    <= sR2;
        if (vR2)
            for (pi = 0; pi < NA; pi = pi + 1)
                if (pi == colR2) psum_reg[pi*ACC +: ACC] <= accR2;
        // A1: chunk 0 of the product add.
        colA1 <= colM;
        a0A1  <= {1'b0, psM[C0-1:0]} + {1'b0, pcsM[C0-1:0]};
        ahA1  <= psM[PW-1:C0];
        bhA1  <= pcsM[PW-1:C0];
        sA1   <= sM;
        // A2: chunk 1.
        colA2 <= colA1;
        a01A2 <= {{1'b0, ahA1[C0+C1-1:C0]} + {1'b0, bhA1[C0+C1-1:C0]} + a0A1[C0], a0A1[C0-1:0]};
        ahA2  <= ahA1[PW-1:C0+C1];
        bhA2  <= bhA1[PW-1:C0+C1];
        sA2   <= sA1;
        // A3: chunk 2.
        colA3  <= colA2;
        prodA3 <= {ahA2 + bhA2 + a01A2[C0+C1], a01A2[C0+C1-1:0]};
        sA3    <= sA2;
        // Q: shift, saturate, write.
        if (vA3)
            for (qi = 0; qi < NA; qi = qi + 1)
                if (qi == colA3) q_reg[qi*AB +: AB] <= q_sat;
    end

endmodule


// ---------------------------------------------------------------------------
// Behavioural columns for full-size engine runs: the same ports and
// cycle-level behaviour as fabric_columns (activations from the cycle
// after start, done two cycles after the last rows, the outputs after a
// walk of T*COLS cycles) with the accumulation as plain arithmetic, so a
// die's eight hundred tiles simulate in minutes.  The columns proper are
// checked bit for bit on their own; fabric_tile selects these with MODEL.
// ---------------------------------------------------------------------------
module fabric_columns_model #(
    parameter int ROWS = 4096,
    parameter int COLS = 64,
    parameter int WB   = 4,
    parameter int AB   = 8,
    parameter int P    = 2,
    parameter int ACC  = 24,
    parameter int SB   = 16,
    parameter int SHB  = 5,
    parameter int T    = 1
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,
    input  wire [T*COLS*ACC-1:0]    psum_in,
    input  wire                     x_valid,
    input  wire [T*P*AB-1:0]        x_data,
    input  wire [P*COLS*WB-1:0]     rom_words,
    input  wire [COLS*SB-1:0]       mult,
    input  wire [COLS*SHB-1:0]      shift,
    output wire [$clog2(ROWS/P)-1:0] cycle,
    output wire                     x_ready,
    output reg                      done,
    output reg  [T*COLS*ACC-1:0]    psum_out,
    output reg  [T*COLS*AB-1:0]     q_out,
    output reg                      q_valid
);
    /*verilator hier_block*/           // one compiled block for every tile under Verilator --hierarchical
    localparam int CYCLES = ROWS / P;
    localparam int CW     = $clog2(CYCLES);
    localparam int NA     = T * COLS;
    localparam int ROWW   = COLS * WB;
    reg          busy;
    reg [CW-1:0] cycle_r;
    reg          vA, vB;
    reg [ACC-1:0] acc [0:NA-1];
    reg [15:0]   walk;
    reg          walking;
    wire consume = busy && x_valid;
    assign x_ready = busy;
    assign cycle   = cycle_r;
    integer t, b, c;
    reg signed [63:0] a, prod;
    reg signed [31:0] mac;      // the accumulate at 32 bits: two 12-bit products on a 24-bit accumulator, exact in ACC bits
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0; cycle_r <= 0; done <= 1'b0; vA <= 1'b0; vB <= 1'b0; q_valid <= 1'b0; walk <= 0; walking <= 1'b0;
        end else begin
            done <= vB; vB <= vA; vA <= consume && (cycle_r == CYCLES - 1);
            q_valid <= 1'b0;
            if (start) begin
                busy <= 1'b1; cycle_r <= 0;
                for (t = 0; t < NA; t = t + 1) acc[t] <= psum_in[t*ACC +: ACC];
            end else if (consume) begin
                cycle_r <= cycle_r + 1'b1;
                if (cycle_r == CYCLES - 1) busy <= 1'b0;
                for (t = 0; t < T; t = t + 1)
                    for (c = 0; c < COLS; c = c + 1) begin
                        mac = $signed(acc[t*COLS + c]);
                        for (b = 0; b < P; b = b + 1)
                            mac = mac + $signed(x_data[(t*P + b)*AB +: AB]) * $signed(rom_words[b*ROWW + c*WB +: WB]);
                        acc[t*COLS + c] <= mac[ACC-1:0];
                    end
            end
            // The requantizer walk: everything resolved at done, published NA + 7 cycles later.
            if (done) begin
                walking <= 1'b1; walk <= 0;
                for (t = 0; t < NA; t = t + 1) begin
                    psum_out[t*ACC +: ACC] <= acc[t];
                    a = $signed(acc[t]);
                    prod = a * $signed({48'b0, mult[(t % COLS)*SB +: SB]});
                    if (shift[(t % COLS)*SHB +: SHB] != 0) prod = prod + (64'sd1 <<< (shift[(t % COLS)*SHB +: SHB] - 1));
                    prod = prod >>> shift[(t % COLS)*SHB +: SHB];
                    if (prod > (64'sd1 <<< (AB - 1)) - 1) prod = (64'sd1 <<< (AB - 1)) - 1;
                    if (prod < -(64'sd1 <<< (AB - 1))) prod = -(64'sd1 <<< (AB - 1));
                    q_out[t*AB +: AB] <= prod[AB-1:0];
                end
            end else if (walking) begin
                walk <= walk + 1'b1;
                if (walk == NA + 6) begin walking <= 1'b0; q_valid <= 1'b1; end
            end
        end
    end
endmodule

// ---------------------------------------------------------------------------
// Tile: ROM plus columns (or, with MODEL, the behavioural columns).
// ---------------------------------------------------------------------------
module fabric_tile #(
    parameter int ROWS = 4096,
    parameter int COLS = 64,
    parameter int WB   = 4,
    parameter int AB   = 8,
    parameter int P    = 2,
    parameter int ACC  = 24,
    parameter int SB   = 16,
    parameter int SHB  = 5,
    parameter int T    = 1,
    parameter int MODEL = 0,
    parameter     ROM_FILE = ""
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  start,
    input  wire [T*COLS*ACC-1:0] psum_in,
    input  wire                  x_valid,
    input  wire [T*P*AB-1:0]     x_data,
    input  wire [COLS*SB-1:0]    mult,
    input  wire [COLS*SHB-1:0]   shift,
    output wire                  x_ready,
    output wire                  done,
    output wire [T*COLS*ACC-1:0] psum_out,
    output wire [T*COLS*AB-1:0]  q_out,
    output wire                  q_valid
);
    wire [$clog2(ROWS/P)-1:0] cycle;
    wire [P*COLS*WB-1:0]      rom_words;

    fabric_rom #(.ROWS(ROWS), .COLS(COLS), .WB(WB), .P(P), .ROM_FILE(ROM_FILE)) rom (
        .cycle(cycle), .words(rom_words));

    generate
        if (MODEL) begin : g_model
            fabric_columns_model #(.ROWS(ROWS), .COLS(COLS), .WB(WB), .AB(AB), .P(P), .ACC(ACC), .SB(SB), .SHB(SHB), .T(T)) columns (
                .clk(clk), .rst_n(rst_n), .start(start), .psum_in(psum_in), .x_valid(x_valid), .x_data(x_data),
                .rom_words(rom_words), .mult(mult), .shift(shift), .cycle(cycle), .x_ready(x_ready), .done(done),
                .psum_out(psum_out), .q_out(q_out), .q_valid(q_valid));
        end else begin : g_real
            fabric_columns #(.ROWS(ROWS), .COLS(COLS), .WB(WB), .AB(AB), .P(P), .ACC(ACC), .SB(SB), .SHB(SHB), .T(T)) columns (
                .clk(clk), .rst_n(rst_n), .start(start), .psum_in(psum_in), .x_valid(x_valid), .x_data(x_data),
                .rom_words(rom_words), .mult(mult), .shift(shift), .cycle(cycle), .x_ready(x_ready), .done(done),
                .psum_out(psum_out), .q_out(q_out), .q_valid(q_valid));
        end
    endgenerate
endmodule

`default_nettype wire
