from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

from pybind11_tests import iostream as m


def test_captured(capsys):
    msg = "I've been redirected to Python, I hope!"
    m.captured_output(msg)
    stdout, stderr = capsys.readouterr()
    assert stdout == msg
    assert stderr == ""

    m.captured_output_default(msg)
    stdout, stderr = capsys.readouterr()
    assert stdout == msg
    assert stderr == ""

    m.captured_err(msg)
    stdout, stderr = capsys.readouterr()
    assert stdout == ""
    assert stderr == msg


def test_captured_large_string(capsys):
    # Make this bigger than the buffer used on the C++ side: 1024 chars
    msg = "I've been redirected to Python, I hope!"
    msg = msg * (1024 // len(msg) + 1)

    m.captured_output_default(msg)
    stdout, stderr = capsys.readouterr()
    assert stdout == msg
    assert stderr == ""


def test_captured_utf8_2byte_offset0(capsys):
    msg = "\u07FF"
    msg = "" + msg * (1024 // len(msg) + 1)

    m.captured_output_default(msg)
    stdout, stderr = capsys.readouterr()
    assert stdout == msg
    assert stderr == ""


def test_captured_utf8_2byte_offset1(capsys):
    msg = "\u07FF"
    msg = "1" + msg * (1024 // len(msg) + 1)

    m.captured_output_default(msg)
    stdout, stderr = capsys.readouterr()
    assert stdout == msg
    assert stderr == ""


def test_captured_utf8_3byte_offset0(capsys):
    msg = "\uFFFF"
    msg = "" + msg * (1024 // len(msg) + 1)

    m.captured_output_default(msg)
    stdout, stderr = capsys.readouterr()
    assert stdout == msg
    assert stderr == ""


def test_captured_utf8_3byte_offset1(capsys):
    msg = "\uFFFF"
    msg = "1" + msg * (1024 // len(msg) + 1)

    m.captured_output_default(msg)
    stdout, stderr = capsys.readouterr()
    assert stdout == msg
    assert stderr == ""


def test_captured_utf8_3byte_offset2(capsys):
    msg = "\uFFFF"
    msg = "12" + msg * (1024 // len(msg) + 1)

    m.captured_output_default(msg)
    stdout, stderr = capsys.readouterr()
    assert stdout == msg
    assert stderr == ""


def test_captured_utf8_4byte_offset0(capsys):
    msg = "\U0010FFFF"
    msg = "" + msg * (1024 // len(msg) + 1)

    m.captured_output_default(msg)
    stdout, stderr = capsys.readouterr()
    assert stdout == msg
    assert stderr == ""


def test_captured_utf8_4byte_offset1(capsys):
    msg = "\U0010FFFF"
    msg = "1" + msg * (1024 // len(msg) + 1)

    m.captured_output_default(msg)
    stdout, stderr = capsys.readouterr()
    assert stdout == msg
    assert stderr == ""


def test_captured_utf8_4byte_offset2(capsys):
    msg = "\U0010FFFF"
    msg = "12" + msg * (1024 // len(msg) + 1)

    m.captured_output_default(msg)
    stdout, stderr = capsys.readouterr()
    assert stdout == msg
    assert stderr == ""


def test_captured_utf8_4byte_offset3(capsys):
    msg = "\U0010FFFF"
    msg = "123" + msg * (1024 // len(msg) + 1)

    m.captured_output_default(msg)
    stdout, stderr = capsys.readouterr()
    assert stdout == msg
    assert stderr == ""


def test_guard_capture(capsys):
    msg = "I've been redirected to Python, I hope!"
    m.guard_output(msg)
    stdout, stderr = capsys.readouterr()
    assert stdout == msg
    assert stderr == ""


def test_series_captured(capture):
    with capture:
        m.captured_output("a")
        m.captured_output("b")
    assert capture == "ab"


def test_flush(capfd):
    msg = "(not flushed)"
    msg2 = "(flushed)"

    with m.ostream_redirect():
        m.noisy_function(msg, flush=False)
        stdout, stderr = capfd.readouterr()
        assert stdout == ""

        m.noisy_function(msg2, flush=True)
        stdout, stderr = capfd.readouterr()
        assert stdout == msg + msg2

        m.noisy_function(msg, flush=False)

    stdout, stderr = capfd.readouterr()
    assert stdout == msg


def test_not_captured(capfd):
    msg = "Something that should not show up in log"
    stream = StringIO()
    with redirect_stdout(stream):
        m.raw_output(msg)
    stdout, stderr = capfd.readouterr()
    assert stdout == msg
    assert stderr == ""
    assert stream.getvalue() == ""

    stream = StringIO()
    with redirect_stdout(stream):
        m.captured_output(msg)
    stdout, stderr = capfd.readouterr()
    assert stdout == ""
    assert stderr == ""
    assert stream.getvalue() == msg


def test_err(capfd):
    msg = "Something that should not show up in log"
    stream = StringIO()
    with redirect_stderr(stream):
        m.raw_err(msg)
    stdout, stderr = capfd.readouterr()
    assert stdout == ""
    assert stderr == msg
    assert stream.getvalue() == ""

    stream = StringIO()
    with redirect_stderr(stream):
        m.captured_err(msg)
    stdout, stderr = capfd.readouterr()
    assert stdout == ""
    assert stderr == ""
    assert stream.getvalue() == msg


def test_multi_captured(capfd):
    stream = StringIO()
    with redirect_stdout(stream):
        m.captured_output("a")
        m.raw_output("b")
        m.captured_output("c")
        m.raw_output("d")
    stdout, stderr = capfd.readouterr()
    assert stdout == "bd"
    assert stream.getvalue() == "ac"


def test_dual(capsys):
    m.captured_dual("a", "b")
    stdout, stderr = capsys.readouterr()
    assert stdout == "a"
    assert stderr == "b"


def test_redirect(capfd):
    msg = "Should not be in log!"
    stream = StringIO()
    with redirect_stdout(stream):
        m.raw_output(msg)
    stdout, stderr = capfd.readouterr()
    assert stdout == msg
    assert stream.getvalue() == ""

    stream = StringIO()
    with redirect_stdout(stream):
        with m.ostream_redirect():
            m.raw_output(msg)
    stdout, stderr = capfd.readouterr()
    assert stdout == ""
    assert stream.getvalue() == msg

    stream = StringIO()
    with redirect_stdout(stream):
        m.raw_output(msg)
    stdout, stderr = capfd.readouterr()
    assert stdout == msg
    assert stream.getvalue() == ""


def test_redirect_err(capfd):
    msg = "StdOut"
    msg2 = "StdErr"

    stream = StringIO()
    with redirect_stderr(stream):
        with m.ostream_redirect(stdout=False):
            m.raw_output(msg)
            m.raw_err(msg2)
    stdout, stderr = capfd.readouterr()
    assert stdout == msg
    assert stderr == ""
    assert stream.getvalue() == msg2


def test_redirect_both(capfd):
    msg = "StdOut"
    msg2 = "StdErr"

    stream = StringIO()
    stream2 = StringIO()
    with redirect_stdout(stream):
        with redirect_stderr(stream2):
            with m.ostream_redirect():
                m.raw_output(msg)
                m.raw_err(msg2)
    stdout, stderr = capfd.readouterr()
    assert stdout == ""
    assert stderr == ""
    assert stream.getvalue() == msg
    assert stream2.getvalue() == msg2


def test_threading():
    with m.ostream_redirect(stdout=True, stderr=False):
        # start some threads
        threads = []

        # start some threads
        for _j in range(20):
            threads.append(m.TestThread())

        # give the threads some time to fail
        threads[0].sleep()

        # stop all the threads
        for t in threads:
            t.stop()

        for t in threads:
            t.join()

        # if a thread segfaults, we don't get here
        assert True
