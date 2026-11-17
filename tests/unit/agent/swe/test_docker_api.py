import io
import struct

from lumenrl.agent.swe.docker_api import CappedBuffer, demux


def _frame(stream_id: int, data: bytes) -> bytes:
    return bytes([stream_id, 0, 0, 0]) + struct.pack(">I", len(data)) + data


def test_demux_interleaves_stdout_and_stderr_in_arrival_order():
    sink = CappedBuffer(1 << 20)
    demux(io.BytesIO(_frame(1, b"out1\n") + _frame(2, b"err\n") + _frame(1, b"out2")), sink)

    assert sink.text() == "out1\nerr\nout2" and sink.total == 13


def test_a_truncated_stream_keeps_what_arrived():
    sink = CappedBuffer(1 << 20)
    demux(io.BytesIO(_frame(1, b"whole") + _frame(1, b"partial")[:-3]), sink)

    assert sink.text() == "wholepart"


def test_capped_output_keeps_head_and_tail_and_says_how_much_is_gone():
    sink = CappedBuffer(8)
    for chunk in (b"abc", b"defgh", b"ijklmn"):
        sink.write(chunk)

    assert sink.total == 14
    assert sink.text() == "abcd\n[... 6 bytes elided ...]\nklmn"


def test_output_under_the_cap_is_untouched():
    sink = CappedBuffer(100)
    sink.write("héllo".encode())

    assert sink.text() == "héllo"
