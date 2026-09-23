// Fixed-point helpers shared by the vector units.  Everything is evaluated
// at 64 bits so a chain of two or three multiplies never truncates; the
// caller takes the bits it keeps.  Golden model: fabric/layer.py
// (rnd_shr, sat, requant).

`ifndef FABRIC_FX_SVH
`define FABRIC_FX_SVH

// Arithmetic right shift with round-half-up; a shift of zero is the identity.
//
// Written as shift-then-round rather than round-then-shift.  Writing v as
// q*2^sh + r with 0 <= r < 2^sh, floor((v + 2^(sh-1)) / 2^sh) is q + 1 exactly
// when r >= 2^(sh-1), which is bit sh-1 of v, so the two forms are the same
// number.  In hardware they are not the same cost: the first spends a 64-bit
// carry-propagate add *ahead* of the barrel shifter, the second replaces it
// with a 64:1 mux that runs alongside the shift and a carry chain that is only
// an incrementer.  Every requantize in the design sits on that path.
// The shift is taken on its own line: folded into the add, the unsigned round
// bit would make the whole expression unsigned and `>>>` a logical shift.
function automatic signed [63:0] fx_rnd_shr(input signed [63:0] v, input integer sh);
    reg signed [63:0] sv;
    reg               rb;
    begin
        if (sh <= 0) fx_rnd_shr = v;
        else begin
            sv = v >>> sh;
            rb = v[sh - 1];
            fx_rnd_shr = sv + $signed({63'b0, rb});
        end
    end
endfunction

// Saturate to a signed width.
function automatic signed [63:0] fx_sat(input signed [63:0] v, input integer bits);
    reg signed [63:0] hi, lo;
    begin
        hi = (64'sd1 <<< (bits - 1)) - 64'sd1;
        lo = -(64'sd1 <<< (bits - 1));
        fx_sat = (v > hi) ? hi : ((v < lo) ? lo : v);
    end
endfunction

// The tile's requantizer: sat_bits((v * mult + 2^(sh-1)) >> sh).
function automatic signed [63:0] fx_requant(input signed [63:0] v, input [15:0] mult, input integer sh, input integer bits);
    fx_requant = fx_sat(fx_rnd_shr(v * $signed({48'b0, mult}), sh), bits);
endfunction

`endif
