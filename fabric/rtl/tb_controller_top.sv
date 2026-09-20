// Self-checking testbench for fabric_controller_top against
// fabric.controller.emit_top_vectors: a request becomes the packet the model
// says it should, and the reply that comes back becomes the token the model
// draws, for the slot it belongs to.

`timescale 1ns/1ps
`default_nettype none

module tb_controller_top #(
    parameter int D      = 8,
    parameter int K      = 8,
    parameter int CASES  = 8,
    parameter int EWORDS = 256,
    parameter int OWORDS = 64,
    parameter int RWORDS = 352
);
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    reg [31:0]  em [0:EWORDS-1];
    reg [127:0] rq [0:CASES-1];
    reg [31:0]  om [0:OWORDS-1];
    reg [31:0]  rm [0:RWORDS-1];
    reg [15:0]  rc [0:CASES-1];
    reg [63:0]  tm [0:CASES-1];

    reg         req_valid = 0;
    reg [15:0]  req_slot = 0, req_inv_t = 0, req_top_p = 0;
    reg [31:0]  req_position = 0, req_token = 0, req_rnd = 0;
    reg [7:0]   req_flags = 8'h01, req_top_k = 0;
    wire        req_ready, emb_en, l_valid, l_sop, r_ready;
    wire [31:0] emb_addr, l_data;
    reg  [31:0] emb_data;
    reg         r_valid = 0, r_sop = 0;
    reg  [31:0] r_data = 0;
    wire        tok_valid;
    wire [31:0] tok_row;
    wire [15:0] tok_slot;
    wire [7:0]  tok_index;

    always @(posedge clk) emb_data <= em[emb_addr];       // the table answers a cycle later

    fabric_controller_top #(.D(D), .K(K), .LUT_DIR("./")) dut (
        .clk(clk), .rst_n(rst_n), .req_valid(req_valid), .req_ready(req_ready), .req_slot(req_slot),
        .req_position(req_position), .req_flags(req_flags), .req_token(req_token), .req_inv_t(req_inv_t),
        .req_top_k(req_top_k), .req_top_p(req_top_p), .req_rnd(req_rnd),
        .emb_en(emb_en), .emb_addr(emb_addr), .emb_data(emb_data),
        .l_valid(l_valid), .l_data(l_data), .l_sop(l_sop), .l_ready(1'b1),
        .r_valid(r_valid), .r_data(r_data), .r_sop(r_sop), .r_ready(r_ready),
        .tok_valid(tok_valid), .tok_row(tok_row), .tok_slot(tok_slot), .tok_index(tok_index));

    integer c, i, errors, ow, rbase, guard, seen_tok;
    reg [31:0] got_row;
    reg [15:0] got_slot;
    reg [7:0]  got_index;

    // The packet the controller puts on the ring, against the model's.
    always @(posedge clk) if (l_valid) begin
        if (l_data !== om[ow]) begin
            errors = errors + 1;
            if (errors <= 5) $display("out word %0d: got %h expected %h", ow, l_data, om[ow]);
        end
        ow = ow + 1;
    end
    always @(posedge clk) if (tok_valid) begin
        seen_tok = seen_tok + 1; got_row = tok_row; got_slot = tok_slot; got_index = tok_index;
    end

    initial begin
        #400000;
        $display("FAIL: stuck after %0d of %0d requests, %0d out words", seen_tok, CASES, ow);
        $finish;
    end

    initial begin
        $readmemh("emb.hex", em);
        $readmemh("req.hex", rq);
        $readmemh("out.hex", om);
        $readmemh("reply.hex", rm);
        $readmemh("rcount.hex", rc);
        $readmemh("token.hex", tm);
        errors = 0; ow = 0; rbase = 0; seen_tok = 0;
        repeat (2) @(posedge clk);
        #1 rst_n = 1;
        for (c = 0; c < CASES; c = c + 1) begin
            req_valid = 1;
            req_slot = rq[c][15:0]; req_position = rq[c][47:16]; req_token = rq[c][55:48];
            req_inv_t = rq[c][71:56]; req_top_k = rq[c][79:72]; req_top_p = rq[c][95:80]; req_rnd = rq[c][127:96];
            @(posedge clk);
            while (!req_ready) @(posedge clk);
            #1 req_valid = 0;
            // Let the packet go out, then hand back the reply a word a cycle.
            guard = 0;
            while (ow < (c + 1) * (D/2 + 4) && guard < 2000) begin @(posedge clk); guard = guard + 1; end
            for (i = 0; i < rc[c]; i = i + 1) begin
                r_valid = 1; r_sop = (i == 0); r_data = rm[rbase + i];
                @(posedge clk);
                while (!r_ready) @(posedge clk);
                #1;
            end
            r_valid = 0; r_sop = 0;
            rbase = rbase + rc[c];
            guard = 0;
            while (seen_tok < c + 1 && guard < 4000) begin @(posedge clk); guard = guard + 1; end
            if (seen_tok < c + 1) begin $display("FAIL: request %0d drew no token", c); $finish; end
            if (got_row !== tm[c][63:24] || got_index !== tm[c][23:16] || got_slot !== tm[c][15:0]) begin
                errors = errors + 1;
                if (errors <= 5)
                    $display("request %0d: got row %0d index %0d slot %0d, expected row %0d index %0d slot %0d",
                             c, got_row, got_index, got_slot, tm[c][63:24], tm[c][23:16], tm[c][15:0]);
            end
        end
        if (errors == 0) $display("PASS: %0d requests", CASES);
        else $display("FAIL: %0d errors", errors);
        $finish;
    end
endmodule

`default_nettype wire
