// Fixed-weight fabric tile: via-programmed coefficient ROM feeding COLS
// multiply-accumulate columns.
//
// One pass processes ROWS activations, P per cycle, and leaves COLS partial
// sums in the accumulators.  Coefficients are symmetric signed WB-bit values;
// the column datapath forms the shared multiples 0..WMAX of each activation
// once per bank and every column selects its multiple through the ROM word,
// then adds or subtracts it.  This is the arithmetic the via-programmed hard
// macro implements; here the ROM is a constant array so the tile simulates and
// synthesizes as a baseline.
//
// Golden model: fabric/tile.py (tile_forward, requantize).

`default_nettype none

module fabric_tile #(
    parameter int ROWS = 4096,          // activations per pass (ROM wordlines)
    parameter int COLS = 64,            // outputs per tile
    parameter int WB   = 4,             // coefficient bits (symmetric signed)
    parameter int AB   = 8,             // activation and output bits (signed)
    parameter int P    = 2,             // rows consumed per cycle (ROM banks)
    parameter int ACC  = 24,            // accumulator bits
    parameter int SB   = 16,            // requantization multiplier bits (unsigned)
    parameter int SHB  = 5,             // requantization shift bits
    parameter     ROM_FILE = ""         // hex image, one ROWS-entry per line, column 0 in the low nibble
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 start,      // begin a pass: load psum_in into the accumulators
    input  wire [COLS*ACC-1:0]  psum_in,    // chained partial sums (zero for a fresh pass)
    input  wire                 x_valid,    // P activations for rows cycle*P .. cycle*P+P-1
    input  wire [P*AB-1:0]      x_data,
    input  wire [COLS*SB-1:0]   mult,       // per-column requantization multiplier
    input  wire [COLS*SHB-1:0]  shift,      // per-column requantization shift
    output wire                 x_ready,    // high while a pass is consuming activations
    output reg                  done,       // one-cycle pulse after the last rows are accumulated
    output wire [COLS*ACC-1:0]  psum_out,   // raw accumulators
    output wire [COLS*AB-1:0]   q_out       // requantized outputs
);
    localparam int WMAX   = (1 << (WB - 1)) - 1;
    localparam int MW     = AB + WB - 1;            // multiple width: WMAX * x fits
    localparam int CYCLES = ROWS / P;
    localparam int CW     = $clog2(CYCLES + 1);
    localparam int ROWW   = COLS * WB;

    // ------------------------------------------------------------------
    // Coefficient ROM (stand-in for the via-programmed macro)
    // ------------------------------------------------------------------
    reg [ROWW-1:0] rom [0:ROWS-1];
    initial begin
        if (ROM_FILE != "") $readmemh(ROM_FILE, rom);
    end

    // ------------------------------------------------------------------
    // Pass control
    // ------------------------------------------------------------------
    reg          busy;
    reg [CW-1:0] cycle;
    wire         consume = busy && x_valid;
    wire         last    = (cycle == CYCLES - 1);

    assign x_ready = busy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy  <= 1'b0;
            cycle <= '0;
            done  <= 1'b0;
        end else begin
            done <= 1'b0;
            if (start) begin
                busy  <= 1'b1;
                cycle <= '0;
            end else if (consume) begin
                cycle <= cycle + 1'b1;
                if (last) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            end
        end
    end

    // ------------------------------------------------------------------
    // Shared multiples per bank: m[b][k] = k * x_b
    // ------------------------------------------------------------------
    wire signed [MW-1:0] multiples [0:P-1][0:WMAX];
    genvar b, k, c;
    generate
        for (b = 0; b < P; b = b + 1) begin : g_bank
            wire signed [AB-1:0] xb = x_data[b*AB +: AB];
            for (k = 0; k <= WMAX; k = k + 1) begin : g_mult
                // The hard macro forms these with shifts and three adders;
                // a constant multiply is the same value.
                assign multiples[b][k] = MW'(xb) * MW'(k);
            end
        end
    endgenerate

    // ------------------------------------------------------------------
    // Columns: select the multiple by ROM word, add or subtract, accumulate
    // ------------------------------------------------------------------
    reg signed [ACC-1:0] acc [0:COLS-1];

    generate
        for (c = 0; c < COLS; c = c + 1) begin : g_col
            wire signed [ACC-1:0] term [0:P-1];
            for (b = 0; b < P; b = b + 1) begin : g_term
                wire [$clog2(ROWS)-1:0] row  = cycle * P + b;
                wire signed [WB-1:0]    word = rom[row][c*WB +: WB];
                wire                    neg  = word[WB-1];
                wire [WB-1:0]           mag  = neg ? -word : word;   // 0..WMAX (never 2^(WB-1))
                wire signed [MW-1:0]    sel  = multiples[b][mag];
                assign term[b] = neg ? -ACC'(sel) : ACC'(sel);
            end

            // Sum of this cycle's P terms.
            reg signed [ACC-1:0] sum;
            integer i;
            always @* begin
                sum = '0;
                for (i = 0; i < P; i = i + 1) sum = sum + term[i];
            end

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)       acc[c] <= '0;
                else if (start)   acc[c] <= psum_in[c*ACC +: ACC];
                else if (consume) acc[c] <= acc[c] + sum;
            end
            assign psum_out[c*ACC +: ACC] = acc[c];

            // Requantization: sat_AB((acc * mult + 2^(shift-1)) >>> shift).
            wire [SB-1:0]             m_c   = mult[c*SB +: SB];
            wire [SHB-1:0]            s_c   = shift[c*SHB +: SHB];
            wire signed [ACC+SB:0]    prod  = acc[c] * $signed({1'b0, m_c});
            wire signed [ACC+SB:0]    rnd   = (s_c == 0) ? '0 : ((ACC+SB+1)'(1) <<< (s_c - 1));
            wire signed [ACC+SB:0]    shr   = (prod + rnd) >>> s_c;
            localparam signed [ACC+SB:0] QMAX = (1 <<< (AB - 1)) - 1;
            localparam signed [ACC+SB:0] QMIN = -(1 <<< (AB - 1));
            assign q_out[c*AB +: AB] = (shr > QMAX) ? QMAX[AB-1:0] :
                                       (shr < QMIN) ? QMIN[AB-1:0] : shr[AB-1:0];
        end
    endgenerate

endmodule

`default_nettype wire
