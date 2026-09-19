// The PHY's two delay lines and the DLL that sets them.
//
// The controller needs two quarter-period delays: the device clock is the
// controller clock delayed by a quarter period, and the capture strobe is
// the device's DQS delayed by a quarter period.  A tap delay line gives a
// delay of code x tap, but the tap's length moves with process, voltage and
// temperature, so a master DLL measures the clock period in taps with a
// replica line and a phase detector, and the quarter is a quarter of that
// code.  The measurement keeps running; the slave lines take a new code
// only when told it is safe (between bursts, CE# high).
//
// The line must span a period at the fastest tap: 256 taps of 25 ps is
// 6.4 ns against the 4 ns period, so a tap between 16 and 100 ps is in
// range.  A line that cannot span the period is reported on range_err
// rather than searched forever.
//
// fabric_delay_line is the behavioural stand-in for the hard cell: a chain
// of TAPS buffers of TAP_PS each and a multiplexer.  Everything else is
// synthesizable.

`timescale 1ns/1ps
`default_nettype none

module fabric_delay_line #(
    parameter int  TAPS   = 256,
    parameter real TAP_PS = 60.0,
    parameter int  CW     = $clog2(TAPS)
) (
    input  wire          in,
    input  wire [CW-1:0] code,
    output reg           out
);
    initial out = 1'b0;
    // A transport delay of code taps; consecutive edges each keep their own delay.
    always @(in) out <= #(code * TAP_PS / 1000.0) in;
endmodule

// ---------------------------------------------------------------------------
// The master loop.  The replica line delays clk by `code` taps; sampling
// clk with the delayed clock tells whether the delayed edge landed in the
// high or the low half of the reference.  From code 0 the sample is high
// (the delayed edge is just past the reference edge), turns low past half a
// period and high again past a full period: the search walks the code up
// until it has seen the low half and then the return to high, which is one
// period.  After that it dithers by one tap around the boundary, tracking
// drift, and `quarter` is the period code over four.
// ---------------------------------------------------------------------------
module fabric_dll #(
    parameter int  TAPS     = 256,
    parameter real TAP_PS   = 60.0,
    parameter int  SETTLE   = 4,                // clocks between phase samples, letting the line settle
    parameter int  CW       = $clog2(TAPS)
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          update_ok,            // the slaves may take a new code now
    output reg           locked,
    output reg           range_err,            // the line cannot span a period: the tap is too short
    output reg  [CW-1:0] period_code,
    output reg  [CW-1:0] quarter               // the code the slave lines use
);
    reg  [CW-1:0] code;
    wire          dclk;
    fabric_delay_line #(.TAPS(TAPS), .TAP_PS(TAP_PS)) replica (.in(clk), .code(code), .out(dclk));

    // The phase detector: the reference sampled by the delayed clock, then
    // brought into the reference domain through two flops.
    reg pd, pd_s1, pd_s2;
    always @(posedge dclk) pd <= clk;
    always @(posedge clk) begin pd_s1 <= pd; pd_s2 <= pd_s1; end

    localparam [1:0] S_UP_HIGH = 0, S_UP_LOW = 1, S_TRACK = 2;
    reg [1:0] state;
    reg [3:0] settle;
    reg [3:0] toggles;                          // direction reversals seen while tracking
    reg       last_dir;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            code <= 0; state <= S_UP_HIGH; settle <= 0; locked <= 1'b0; range_err <= 1'b0; period_code <= 0; quarter <= 0;
            toggles <= 0; last_dir <= 1'b1;
        end else begin
            if (settle != SETTLE - 1) settle <= settle + 1'b1;
            else begin
                settle <= 0;
                case (state)
                    S_UP_HIGH: begin                        // walking up through the first high half
                        if (code == TAPS - 1) begin code <= 0; range_err <= 1'b1; end   // never reached half a period
                        else code <= code + 1'b1;
                        if (!pd_s2 && code > 2) state <= S_UP_LOW;
                    end
                    S_UP_LOW: begin                         // through the low half until the next high
                        if (pd_s2) begin state <= S_TRACK; period_code <= code; end
                        else if (code == TAPS - 1) begin state <= S_UP_HIGH; code <= 0; range_err <= 1'b1; end
                        else code <= code + 1'b1;
                    end
                    default: begin                          // dither by one tap around the period
                        if (pd_s2) begin code <= code - 1'b1; if (last_dir) toggles <= toggles + 1'b1; last_dir <= 1'b0; end
                        else       begin code <= code + 1'b1; if (!last_dir) toggles <= toggles + 1'b1; last_dir <= 1'b1; end
                        period_code <= pd_s2 ? code : code + 1'b1;
                        if (toggles >= 4) locked <= 1'b1;
                        if (toggles == 4'd15) toggles <= 4'd8;
                    end
                endcase
            end
            // The slaves take the quarter only when locked and allowed.
            if (locked && update_ok) quarter <= (period_code + 2'd2) >> 2;
        end
    end
endmodule

`default_nettype wire
