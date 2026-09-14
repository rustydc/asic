// Fixed-weight fabric tile: via-programmed coefficient ROM feeding COLS
// multiply-accumulate columns.
//
// One pass processes ROWS activations, P per cycle, and leaves COLS partial
// sums in the accumulators.  Coefficients are symmetric signed WB-bit values;
// the column datapath forms the shared multiples 0..WMAX of each activation
// once per bank and every column selects its multiple through the ROM word,
// then adds or subtracts it.
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
    parameter int SHB  = 5              // requantization shift bits
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,      // begin a pass: load psum_in into the accumulators
    input  wire [COLS*ACC-1:0]      psum_in,    // chained partial sums (zero for a fresh pass)
    input  wire                     x_valid,    // P activations for rows cycle*P .. cycle*P+P-1
    input  wire [P*AB-1:0]          x_data,
    input  wire [P*COLS*WB-1:0]     rom_words,  // coefficient words for the same P rows
    input  wire [COLS*SB-1:0]       mult,       // per-column requantization multiplier
    input  wire [COLS*SHB-1:0]      shift,      // per-column requantization shift
    output wire [$clog2(ROWS/P)-1:0] cycle,     // ROM address (row group)
    output wire                     x_ready,    // high while a pass is consuming activations
    output reg                      done,       // one-cycle pulse after the last rows are accumulated
    output wire [COLS*ACC-1:0]      psum_out,   // raw accumulators
    output wire [COLS*AB-1:0]       q_out,      // requantized outputs, valid when q_valid
    output wire                     q_valid     // pulses COLS cycles after done
);
    localparam int WMAX   = (1 << (WB - 1)) - 1;
    localparam int MW     = AB + WB - 1;            // multiple width: WMAX * x fits
    localparam int CYCLES = ROWS / P;
    localparam int CW     = $clog2(CYCLES);
    localparam int ROWW   = COLS * WB;

    // ------------------------------------------------------------------
    // Pass control
    // ------------------------------------------------------------------
    reg          busy;
    reg [CW-1:0] cycle_r;
    wire         consume = busy && x_valid;
    wire         last    = (cycle_r == CYCLES - 1);

    assign x_ready = busy;
    assign cycle   = cycle_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy    <= 1'b0;
            cycle_r <= {CW{1'b0}};
            done    <= 1'b0;
        end else begin
            done <= 1'b0;
            if (start) begin
                busy    <= 1'b1;
                cycle_r <= {CW{1'b0}};
            end else if (consume) begin
                cycle_r <= cycle_r + 1'b1;
                if (last) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            end
        end
    end

    // ------------------------------------------------------------------
    // Shared multiples per bank, packed: multiples[b][k] at
    // mult_bus[(b*(WMAX+1) + k)*MW +: MW].  The hard macro forms these with
    // shifts and three adders; a constant multiply is the same value.
    // ------------------------------------------------------------------
    wire [P*(WMAX+1)*MW-1:0] mult_bus;
    genvar b, k, c;
    generate
        for (b = 0; b < P; b = b + 1) begin : g_bank
            wire signed [AB-1:0] xb = x_data[b*AB +: AB];
            for (k = 0; k <= WMAX; k = k + 1) begin : g_mult
                wire signed [MW-1:0] m = xb * k;
                assign mult_bus[(b*(WMAX+1) + k)*MW +: MW] = m;
            end
        end
    endgenerate

    // ------------------------------------------------------------------
    // Columns: select the multiple by ROM word, add or subtract, accumulate
    // ------------------------------------------------------------------
    reg signed [ACC-1:0] acc [0:COLS-1];

    // Term width: a signed multiple plus its negation needs MW+1 bits; the
    // sum of P terms needs log2(P) more.  Only the final accumulate is ACC wide.
    localparam int TW = MW + 1;
    localparam int SW = TW + ((P > 1) ? $clog2(P) : 0);

    generate
        for (c = 0; c < COLS; c = c + 1) begin : g_col
            wire signed [TW-1:0] term_bus [0:P-1];
            for (b = 0; b < P; b = b + 1) begin : g_term
                wire signed [WB-1:0] word = rom_words[b*ROWW + c*WB +: WB];
                wire                 neg  = word[WB-1];
                wire signed [WB:0]   wext = {word[WB-1], word};
                wire [WB-1:0]        mag  = neg ? (-wext) : wext;      // 0..WMAX
                wire signed [MW-1:0] sel  = mult_bus[(b*(WMAX+1) + mag)*MW +: MW];
                wire signed [TW-1:0] sext = {sel[MW-1], sel};
                assign term_bus[b] = neg ? (-sext) : sext;
            end

            // Sum of this cycle's P terms (small P, so a linear chain).
            wire signed [SW-1:0] partial [0:P];
            assign partial[0] = {SW{1'b0}};
            for (b = 0; b < P; b = b + 1) begin : g_sum
                wire signed [SW-1:0] t = {{(SW-TW){term_bus[b][TW-1]}}, term_bus[b]};
                assign partial[b+1] = partial[b] + t;
            end
            wire signed [SW-1:0]  sum  = partial[P];
            wire signed [ACC-1:0] sumx = {{(ACC-SW){sum[SW-1]}}, sum};

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)       acc[c] <= {ACC{1'b0}};
                else if (start)   acc[c] <= psum_in[c*ACC +: ACC];
                else if (consume) acc[c] <= acc[c] + sumx;
            end
            assign psum_out[c*ACC +: ACC] = acc[c];
        end
    endgenerate

    // ------------------------------------------------------------------
    // Shared requantizer: one multiplier walks the columns after the pass.
    // sat_AB((acc * mult + 2^(shift-1)) >>> shift), one column per cycle.
    // ------------------------------------------------------------------
    localparam int QW = $clog2(COLS);
    reg              q_busy;
    reg  [QW-1:0]    q_col;
    reg  [COLS*AB-1:0] q_reg;
    reg              q_valid_r;
    assign q_out   = q_reg;
    assign q_valid = q_valid_r;

    wire signed [ACC-1:0]   acc_sel = acc[q_col];
    wire [SB-1:0]           m_c     = mult[q_col*SB +: SB];
    wire [SHB-1:0]          s_c     = shift[q_col*SHB +: SHB];
    wire signed [SB:0]      m_s     = {1'b0, m_c};
    wire signed [ACC+SB:0]  prod    = acc_sel * m_s;
    wire signed [ACC+SB:0]  one     = {{(ACC+SB){1'b0}}, 1'b1};
    wire signed [ACC+SB:0]  rnd     = (s_c == 0) ? {(ACC+SB+1){1'b0}} : (one <<< (s_c - 1));
    wire signed [ACC+SB:0]  shr     = (prod + rnd) >>> s_c;
    wire signed [ACC+SB:0]  qmax    = (one <<< (AB - 1)) - one;
    wire signed [ACC+SB:0]  qmin    = -(one <<< (AB - 1));
    wire [AB-1:0]           q_sat   = (shr > qmax) ? qmax[AB-1:0] :
                                      (shr < qmin) ? qmin[AB-1:0] : shr[AB-1:0];

    integer qi;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q_busy    <= 1'b0;
            q_col     <= {QW{1'b0}};
            q_valid_r <= 1'b0;
            q_reg     <= {(COLS*AB){1'b0}};
        end else begin
            q_valid_r <= 1'b0;
            if (done) begin
                q_busy <= 1'b1;
                q_col  <= {QW{1'b0}};
            end else if (q_busy) begin
                for (qi = 0; qi < COLS; qi = qi + 1)
                    if (qi == q_col) q_reg[qi*AB +: AB] <= q_sat;
                if (q_col == COLS - 1) begin
                    q_busy    <= 1'b0;
                    q_valid_r <= 1'b1;
                end
                q_col <= q_col + 1'b1;
            end
        end
    end

endmodule

// ---------------------------------------------------------------------------
// Tile: ROM plus columns.
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
    parameter     ROM_FILE = ""
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 start,
    input  wire [COLS*ACC-1:0]  psum_in,
    input  wire                 x_valid,
    input  wire [P*AB-1:0]      x_data,
    input  wire [COLS*SB-1:0]   mult,
    input  wire [COLS*SHB-1:0]  shift,
    output wire                 x_ready,
    output wire                 done,
    output wire [COLS*ACC-1:0]  psum_out,
    output wire [COLS*AB-1:0]   q_out,
    output wire                 q_valid
);
    wire [$clog2(ROWS/P)-1:0] cycle;
    wire [P*COLS*WB-1:0]      rom_words;

    fabric_rom #(.ROWS(ROWS), .COLS(COLS), .WB(WB), .P(P), .ROM_FILE(ROM_FILE)) rom (
        .cycle(cycle), .words(rom_words));

    fabric_columns #(.ROWS(ROWS), .COLS(COLS), .WB(WB), .AB(AB), .P(P), .ACC(ACC), .SB(SB), .SHB(SHB)) columns (
        .clk(clk), .rst_n(rst_n), .start(start), .psum_in(psum_in), .x_valid(x_valid), .x_data(x_data),
        .rom_words(rom_words), .mult(mult), .shift(shift), .cycle(cycle), .x_ready(x_ready), .done(done),
        .psum_out(psum_out), .q_out(q_out), .q_valid(q_valid));
endmodule

`default_nettype wire
