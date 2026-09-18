// RMS norm over a vector of D elements, L per beat.
//
// The vector streams in and is buffered while its sum of squares
// accumulates; the inverse square root of (ss + eps) is then formed once
// and the vector streams back out as
//   n = sat16((x * R) >> (1 + SW/2 - a/2))          (x / sqrt(ss) in Q.14)
//   y = sat_OW((n * gain * mult + 2^(shift-1)) >> shift)
// The per-element gain is the norm weight when it cannot be folded into the
// following matrix, or the gate of the gated norm, and travels with the
// input beat.  sqrt(D) is in mult; with mult = 1 and shift = 7 the unit is
// an L2 normaliser emitting an int8 unit vector.
//
// Latency: D/L input beats, 7 cycles, then D/L output beats.
// Golden model: fabric/layer.py rmsnorm_int.

`default_nettype none
`include "fabric_fx.svh"

module fabric_rmsnorm #(
    parameter int D  = 4096,
    parameter int XW = 16,          // input element width
    parameter int OW = 8,           // output element width
    parameter int L  = 2,           // elements per beat
    parameter int SW = 44,          // width of the sum of squares (even)
    parameter int GW = 16,          // gain width
    parameter     LUT_DIR = "./"
) (
    input  wire            clk,
    input  wire            rst_n,
    input  wire            in_valid,
    input  wire [L*XW-1:0] in_x,
    input  wire [L*GW-1:0] in_gain,
    input  wire [15:0]     mult,
    input  wire [5:0]      shift,
    input  wire [SW-1:0]   eps,
    output reg             out_valid,
    output reg  [L*OW-1:0] out_y
);
    localparam int BEATS = D / L;
    localparam int BW    = $clog2(BEATS) + 1;

    reg [L*XW-1:0] xmem [0:BEATS-1];
    reg [L*GW-1:0] gmem [0:BEATS-1];
    reg [BW-1:0]   wr;
    reg [SW-1:0]   ss;

    // Sum of the beat's squares (a plain adder here; carry-save in silicon).
    reg signed [63:0] beat_sq;
    integer i;
    always @* begin
        beat_sq = 0;
        for (i = 0; i < L; i = i + 1)
            beat_sq = beat_sq + $signed(in_x[i*XW +: XW]) * $signed(in_x[i*XW +: XW]);
    end

    // Phases: 0 fill, 1 rsqrt in flight, 2 drain.
    reg [1:0]  phase;
    reg        rs_start;
    wire       rs_done;
    wire [16:0] r_w;
    wire [6:0]  a_w;
    reg  [16:0] r;
    reg  [6:0]  a;
    wire [SW-1:0] ss_eps = ss + eps;
    wire [SW-1:0] ss_in  = (ss_eps == 0) ? {{(SW-1){1'b0}}, 1'b1} : ss_eps;
    fabric_rsqrt #(.SW(SW), .LUT_DIR(LUT_DIR)) rsq (.clk(clk), .start(rs_start), .ss(ss_in), .done(rs_done), .r(r_w), .a(a_w));

    reg [BW-1:0] rd, rd_addr;
    reg          rd_valid;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase <= 2'd0; wr <= 0; ss <= 0; rs_start <= 1'b0; rd <= 0; rd_valid <= 1'b0;
        end else begin
            rs_start <= 1'b0;
            rd_valid <= 1'b0;
            case (phase)
                2'd0: if (in_valid) begin
                    xmem[wr] <= in_x;
                    gmem[wr] <= in_gain;
                    ss <= ss + beat_sq[SW-1:0];
                    wr <= wr + 1'b1;
                    if (wr == BEATS - 1) begin
                        phase <= 2'd1;
                        rs_start <= 1'b1;
                    end
                end
                2'd1: begin
                    if (rs_done) begin
                        r <= r_w;
                        a <= a_w;
                        phase <= 2'd2;
                        rd <= 0;
                    end
                end
                default: begin
                    rd_valid <= 1'b1;
                    rd_addr <= rd;
                    rd <= rd + 1'b1;
                    if (rd == BEATS - 1) begin
                        phase <= 2'd0;
                        wr <= 0;
                        ss <= 0;
                    end
                end
            endcase
        end
    end
    // The start pulse is registered with the last beat's sum, so the
    // square-root unit samples the complete ss.

    // Drain pipeline: P0 read, P1 n, P2 y.
    reg [L*XW-1:0] x0;
    reg [L*GW-1:0] g0;
    reg            v0, v1;
    reg [L*16-1:0] n1;
    reg [L*GW-1:0] g1;
    integer sh;
    always @* sh = 1 + SW / 2 - a / 2;
    integer k;
    always @(posedge clk) begin
        v0 <= rd_valid;
        x0 <= xmem[rd_addr];
        g0 <= gmem[rd_addr];
        v1 <= v0;
        g1 <= g0;
        for (k = 0; k < L; k = k + 1)
            n1[k*16 +: 16] <= fx_sat(fx_rnd_shr($signed(x0[k*XW +: XW]) * $signed({47'b0, r}), sh), 16);
        out_valid <= v1;
        for (k = 0; k < L; k = k + 1)
            out_y[k*OW +: OW] <= fx_requant($signed(n1[k*16 +: 16]) * $signed(g1[k*GW +: GW]), mult, shift, OW);
    end
endmodule

`default_nettype wire
