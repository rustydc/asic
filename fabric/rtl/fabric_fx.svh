// Fixed-point helpers shared by the vector units.  Everything is evaluated
// at 64 bits so a chain of two or three multiplies never truncates; the
// caller takes the bits it keeps.  Golden model: fabric/layer.py
// (rnd_shr, sat, requant).

`ifndef FABRIC_FX_SVH
`define FABRIC_FX_SVH

// Arithmetic right shift with round-half-up; a shift of zero is the identity.
function automatic signed [63:0] fx_rnd_shr(input signed [63:0] v, input integer sh);
    if (sh <= 0) fx_rnd_shr = v;
    else         fx_rnd_shr = (v + (64'sd1 <<< (sh - 1))) >>> sh;
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
