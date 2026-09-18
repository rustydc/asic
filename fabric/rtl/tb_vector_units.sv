// Self-checking testbench for the scalar units: sigmoid, SiLU, exp(-t),
// softplus, rsqrt and reciprocal against vectors from
// fabric.layer.emit_unit_vectors, run from the vector directory.

`timescale 1ns/1ps
`default_nettype none

module tb_vector_units #(
    parameter int N  = 512,
    parameter int SW = 44,
    parameter int LW = 28
);
    reg clk = 0;
    always #5 clk = ~clk;

    reg [15:0]   tmem [0:N-1];
    reg [21:0]   tumem [0:N-1];
    reg [SW-1:0] ssmem [0:N-1];
    reg [LW-1:0] lmem [0:N-1];
    reg [15:0]   e_sig [0:N-1], e_silu [0:N-1], e_sp [0:N-1], e_exp [0:N-1];
    reg [16:0]   e_rs [0:N-1], e_rc [0:N-1];
    reg [5:0]    e_rsa [0:N-1];
    reg [4:0]    e_rclz [0:N-1];

    reg               vin = 0;
    reg signed [15:0] t = 0;
    reg [21:0]        tu = 0;
    reg               st = 0;
    reg [SW-1:0]      ss = 0;
    reg [LW-1:0]      l = 0;
    wire        v_sig, v_silu, v_exp, v_sp, d_rs, d_rc;
    wire [15:0] y_sig, y_silu, y_exp, y_sp;
    wire [16:0] r_rs, r_rc;
    wire [6:0]  a_rs;
    wire [5:0]  lz_rc;

    fabric_sigmoid  u_sig  (.clk(clk), .valid_in(vin), .t(t),  .valid_out(v_sig),  .y(y_sig));
    fabric_silu     u_silu (.clk(clk), .valid_in(vin), .t(t),  .valid_out(v_silu), .y(y_silu));
    fabric_exp_neg  u_exp  (.clk(clk), .valid_in(vin), .t(tu), .valid_out(v_exp),  .y(y_exp));
    fabric_softplus u_sp   (.clk(clk), .valid_in(vin), .t(t),  .valid_out(v_sp),   .y(y_sp));
    fabric_rsqrt #(.SW(SW)) u_rs (.clk(clk), .start(st), .ss(ss), .done(d_rs), .r(r_rs), .a(a_rs));
    fabric_recip #(.LW(LW)) u_rc (.clk(clk), .start(st), .l(l),  .done(d_rc), .r(r_rc), .lz_out(lz_rc));

    integer i, errors;
    integer n_sig = 0, n_silu = 0, n_exp = 0, n_sp = 0, n_rs = 0, n_rc = 0;
    always @(posedge clk) begin
        if (v_sig)  begin if (y_sig  !== e_sig[n_sig])   begin errors = errors + 1; if (errors <= 5) $display("sigmoid %0d: got %h expected %h", n_sig, y_sig, e_sig[n_sig]); end n_sig = n_sig + 1; end
        if (v_silu) begin if (y_silu !== e_silu[n_silu]) begin errors = errors + 1; if (errors <= 5) $display("silu %0d: got %h expected %h", n_silu, y_silu, e_silu[n_silu]); end n_silu = n_silu + 1; end
        if (v_exp)  begin if (y_exp  !== e_exp[n_exp])   begin errors = errors + 1; if (errors <= 5) $display("exp %0d: got %h expected %h", n_exp, y_exp, e_exp[n_exp]); end n_exp = n_exp + 1; end
        if (v_sp)   begin if (y_sp   !== e_sp[n_sp])     begin errors = errors + 1; if (errors <= 5) $display("softplus %0d: got %h expected %h", n_sp, y_sp, e_sp[n_sp]); end n_sp = n_sp + 1; end
        if (d_rs)   begin if (r_rs !== e_rs[n_rs] || a_rs !== e_rsa[n_rs]) begin errors = errors + 1; if (errors <= 5) $display("rsqrt %0d: got %h/%0d expected %h/%0d", n_rs, r_rs, a_rs, e_rs[n_rs], e_rsa[n_rs]); end n_rs = n_rs + 1; end
        if (d_rc)   begin if (r_rc !== e_rc[n_rc] || lz_rc !== e_rclz[n_rc]) begin errors = errors + 1; if (errors <= 5) $display("recip %0d: got %h/%0d expected %h/%0d", n_rc, r_rc, lz_rc, e_rc[n_rc], e_rclz[n_rc]); end n_rc = n_rc + 1; end
    end

    initial begin
        $readmemh("t.hex", tmem);
        $readmemh("tu.hex", tumem);
        $readmemh("ss.hex", ssmem);
        $readmemh("l.hex", lmem);
        $readmemh("exp_sigmoid.hex", e_sig);
        $readmemh("exp_silu.hex", e_silu);
        $readmemh("exp_softplus.hex", e_sp);
        $readmemh("exp_exp.hex", e_exp);
        $readmemh("exp_rsqrt.hex", e_rs);
        $readmemh("exp_rsqrt_a.hex", e_rsa);
        $readmemh("exp_recip.hex", e_rc);
        $readmemh("exp_recip_lz.hex", e_rclz);
        errors = 0;
        repeat (2) @(posedge clk);
        for (i = 0; i < N; i = i + 1) begin
            @(negedge clk);
            vin = 1; t = tmem[i]; tu = tumem[i];
            st = 1; ss = ssmem[i]; l = lmem[i];
        end
        @(negedge clk);
        vin = 0; st = 0;
        repeat (12) @(posedge clk);
        if (n_sig != N || n_silu != N || n_exp != N || n_sp != N || n_rs != N || n_rc != N) begin
            $display("FAIL: counts %0d %0d %0d %0d %0d %0d of %0d", n_sig, n_silu, n_exp, n_sp, n_rs, n_rc, N);
        end else if (errors == 0) $display("PASS: %0d vectors", N);
        else $display("FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule

`default_nettype wire
