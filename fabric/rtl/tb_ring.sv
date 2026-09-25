// Self-checking testbench for fabric_ring_tx and fabric_ring_rx against
// fabric.controller.emit_ring_vectors: the words the transmitter puts on the
// link are the model's packet, the receiver gives the header and payload back,
// and a flipped bit anywhere in a packet is caught.

`timescale 1ns/1ps
`default_nettype none

module tb_ring #(
    parameter int CASES  = 12,
    parameter int PWORDS = 256,
    parameter int KWORDS = 320
);
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    reg [95:0] hm [0:CASES-1];
    reg [31:0] pm [0:PWORDS-1];
    reg [31:0] km [0:KWORDS-1];

    reg         hdr_valid = 0, p_valid = 0;
    reg [7:0]   hdr_kind = 0, hdr_flags = 0, hdr_tokens = 0;
    reg [15:0]  hdr_context = 0;
    reg [23:0]  hdr_length = 0;
    reg [31:0]  hdr_position = 0, p_data = 0;
    wire        hdr_ready, p_ready, l_valid, l_sop, l_ready;
    wire [31:0] l_data;
    fabric_ring_tx tx (
        .clk(clk), .rst_n(rst_n), .hdr_valid(hdr_valid), .hdr_ready(hdr_ready), .hdr_kind(hdr_kind),
        .hdr_flags(hdr_flags), .hdr_context(hdr_context), .hdr_position(hdr_position),
        .hdr_tokens(hdr_tokens), .hdr_length(hdr_length), .p_valid(p_valid), .p_ready(p_ready), .p_data(p_data),
        .l_valid(l_valid), .l_data(l_data), .l_sop(l_sop), .l_ready(l_ready));

    // The link, with a bit flipped in one word of one packet when asked.
    reg        corrupt = 0;
    reg [15:0] corrupt_at = 0;
    integer    lword = 0;
    wire [31:0] rx_data = (corrupt && lword == corrupt_at) ? (l_data ^ 32'h0000_0010) : l_data;

    reg         rx_ready = 1;
    wire        r_hdr_valid, r_p_valid, r_p_last, r_done, r_ok;
    wire [7:0]  r_kind, r_flags, r_tokens;
    wire [15:0] r_context;
    wire [23:0] r_length;
    wire [31:0] r_position, r_p_data;
    fabric_ring_rx rx (
        .clk(clk), .rst_n(rst_n), .l_valid(l_valid), .l_data(rx_data), .l_sop(l_sop), .l_ready(l_ready),
        .rx_ready(rx_ready), .hdr_valid(r_hdr_valid), .hdr_kind(r_kind), .hdr_flags(r_flags),
        .hdr_context(r_context), .hdr_position(r_position), .hdr_tokens(r_tokens), .hdr_length(r_length),
        .p_valid(r_p_valid), .p_data(r_p_data), .p_last(r_p_last), .done(r_done), .ok(r_ok));

    integer c, i, errors, pbase, kbase, at, guard, got_p, seen_done, seen_hdr;
    reg [31:0] seen_pay [0:PWORDS-1];
    reg        last_ok;

    // The link words, checked against the model's packet as they go by.
    always @(posedge clk) if (l_valid && l_ready) begin
        if (!corrupt && l_data !== km[lword]) begin
            errors = errors + 1;
            if (errors <= 5) $display("word %0d: got %h expected %h", lword, l_data, km[lword]);
        end
        lword = lword + 1;
    end
    always @(posedge clk) begin
        if (r_hdr_valid) begin
            if (seen_hdr < CASES && {r_tokens, r_length, r_position, r_context, r_flags, r_kind} !== hm[seen_hdr]) begin
                errors = errors + 1;
                if (errors <= 5) $display("header %0d: got %h expected %h", seen_hdr,
                                          {r_tokens, r_length, r_position, r_context, r_flags, r_kind}, hm[seen_hdr]);
            end
            seen_hdr = seen_hdr + 1;
        end
        if (r_p_valid) begin seen_pay[got_p] = r_p_data; got_p = got_p + 1; end
        if (r_done) begin seen_done = seen_done + 1; last_ok = r_ok; end
    end

    // A word is held until the clock edge that takes it, and ready is read at
    // that edge, where it still carries the ending cycle's value.  Reading it
    // at the half cycle instead races whatever else moves there -- the
    // receiver's stall did -- and drops the word.
    task send(input integer c, input integer pbase, input integer n);
        begin
            hdr_valid = 1; hdr_kind = hm[c][7:0]; hdr_flags = hm[c][15:8]; hdr_context = hm[c][31:16];
            hdr_position = hm[c][63:32]; hdr_length = hm[c][87:64]; hdr_tokens = hm[c][95:88];
            @(posedge clk);
            while (!hdr_ready) @(posedge clk);
            #1 hdr_valid = 0;
            for (i = 0; i < n; i = i + 1) begin
                p_valid = 1; p_data = pm[pbase + i];
                @(posedge clk);
                while (!p_ready) @(posedge clk);
                #1;
                if (i % 4 == 3) begin p_valid = 0; @(posedge clk); #1; end   // a gap: the link must wait
            end
            p_valid = 0;
        end
    endtask

    // The receiver stalls of its own accord, not from inside the sending loop,
    // which could not then run on to let it start again.
    initial begin
        wait (lword > 20);
        @(posedge clk); #1 rx_ready = 0;
        repeat (6) @(posedge clk);
        #1 rx_ready = 1;
    end

    initial begin                                      // a watchdog, so a deadlock says where
        #200000;
        $display("FAIL: stuck at link word %0d, %0d done, tx state %0d, rx state %0d, c %0d i %0d",
                 lword, seen_done, tx.state, rx.state, c, i);
        $finish;
    end

    initial begin
        $readmemh("hdr.hex", hm);
        $readmemh("payload.hex", pm);
        $readmemh("packet.hex", km);
        errors = 0; lword = 0; got_p = 0; seen_done = 0; seen_hdr = 0; pbase = 0;
        repeat (2) @(posedge clk);
        #1 rst_n = 1;
        for (c = 0; c < CASES; c = c + 1) begin
            send(c, pbase, hm[c][87:64] / 4);
            pbase = pbase + hm[c][87:64] / 4;
        end
        guard = 0;
        while (seen_done < CASES && guard < 20000) begin @(posedge clk); guard = guard + 1; end
        if (seen_done != CASES) begin $display("FAIL: %0d packets received of %0d", seen_done, CASES); $finish; end
        if (seen_hdr != CASES) begin $display("FAIL: %0d headers of %0d", seen_hdr, CASES); $finish; end
        if (got_p != pbase) begin $display("FAIL: %0d payload words of %0d", got_p, pbase); $finish; end
        for (i = 0; i < pbase; i = i + 1)
            if (seen_pay[i] !== pm[i]) begin
                errors = errors + 1;
                if (errors <= 5) $display("payload %0d: got %h expected %h", i, seen_pay[i], pm[i]);
            end
        if (!last_ok) begin $display("FAIL: a good packet was rejected"); $finish; end
        // The same packet again with one bit of its payload flipped.
        #1 corrupt = 1; corrupt_at = lword + 4;
        seen_done = 0;
        send(0, 0, hm[0][87:64] / 4);
        guard = 0;
        while (seen_done < 1 && guard < 4000) begin @(posedge clk); guard = guard + 1; end
        if (seen_done != 1) $display("FAIL: the corrupted packet never finished");
        else if (last_ok) begin errors = errors + 1; $display("FAIL: a flipped bit passed the CRC"); end
        if (errors == 0) $display("PASS: %0d packets, %0d words", CASES, pbase);
        else $display("FAIL: %0d errors", errors);
        $finish;
    end
endmodule

`default_nettype wire
