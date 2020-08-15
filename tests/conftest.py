# -*- coding: utf-8 -*-
"""pytest configuration

Extends output capture as needed by pybind11: ignore constructors, optional unordered lines.
Adds docstring and exceptions message sanitizers: ignore Python 2 vs 3 differences.
"""

import contextlib
import difflib
import gc
import platform
import re
import sys
import textwrap

import pytest

# Early diagnostic for failed imports
import pybind11_tests  # noqa: F401

_unicode_marker = re.compile(r'u(\'[^\']*\')')
_long_marker = re.compile(r'([0-9])L')
_hexadecimal = re.compile(r'0x[0-9a-fA-F]+')


def _strip_and_dedent(s):
    """For triple-quote strings"""
    return textwrap.dedent(s.lstrip('\n').rstrip())


def _split_and_sort(s):
    """For output which does not require specific line order"""
    return sorted(_strip_and_dedent(s).splitlines())


def _make_explanation(a, b):
    """Explanation for a failed assert -- the a and b arguments are List[str]"""
    return ["--- actual / +++ expected"] + [line.strip('\n') for line in difflib.ndiff(a, b)]


class Output(object):
    """Basic output post-processing and comparison"""
    def __init__(self, string):
        self.string = string
        self.explanation = []

    def __str__(self):
        return self.string

    def __eq__(self, other):
        # Ignore constructor/destructor output which is prefixed with "###"
        a = [line for line in self.string.strip().splitlines() if not line.startswith("###")]
        b = _strip_and_dedent(other).splitlines()
        if a == b:
            return True
        else:
            self.explanation = _make_explanation(a, b)
            return False


class Unordered(Output):
    """Custom comparison for output without strict line ordering"""
    def __eq__(self, other):
        a = _split_and_sort(self.string)
        b = _split_and_sort(other)
        if a == b:
            return True
        else:
            self.explanation = _make_explanation(a, b)
            return False


class Capture(object):
    def __init__(self, capfd):
        self.capfd = capfd
        self.out = ""
        self.err = ""

    def __enter__(self):
        self.capfd.readouterr()
        return self

    def __exit__(self, *args):
        self.out, self.err = self.capfd.readouterr()

    def __eq__(self, other):
        a = Output(self.out)
        b = other
        if a == b:
            return True
        else:
            self.explanation = a.explanation
            return False

    def __str__(self):
        return self.out

    def __contains__(self, item):
        return item in self.out

    @property
    def unordered(self):
        return Unordered(self.out)

    @property
    def stderr(self):
        return Output(self.err)


@pytest.fixture
def capture(capsys):
    """Extended `capsys` with context manager and custom equality operators"""
    return Capture(capsys)


class SanitizedString(object):
    def __init__(self, sanitizer):
        self.sanitizer = sanitizer
        self.string = ""
        self.explanation = []

    def __call__(self, thing):
        self.string = self.sanitizer(thing)
        return self

    def __eq__(self, other):
        a = self.string
        b = _strip_and_dedent(other)
        if a == b:
            return True
        else:
            self.explanation = _make_explanation(a.splitlines(), b.splitlines())
            return False


def _sanitize_general(s):
    s = s.strip()
    s = s.replace("pybind11_tests.", "m.")
    s = s.replace("unicode", "str")
    s = _long_marker.sub(r"\1", s)
    s = _unicode_marker.sub(r"\1", s)
    return s


def _sanitize_docstring(thing):
    s = thing.__doc__
    s = _sanitize_general(s)
    return s


@pytest.fixture
def doc():
    """Sanitize docstrings and add custom failure explanation"""
    return SanitizedString(_sanitize_docstring)


def _sanitize_message(thing):
    s = str(thing)
    s = _sanitize_general(s)
    s = _hexadecimal.sub("0", s)
    return s


@pytest.fixture
def msg():
    """Sanitize messages and add custom failure explanation"""
    return SanitizedString(_sanitize_message)


# noinspection PyUnusedLocal
def pytest_assertrepr_compare(op, left, right):
    """Hook to insert custom failure explanation"""
    if hasattr(left, 'explanation'):
        return left.explanation


@contextlib.contextmanager
def suppress(exception):
    """Suppress the desired exception"""
    try:
        yield
    except exception:
        pass


def gc_collect():
    ''' Run the garbage collector twice (needed when running
    reference counting tests with PyPy) '''
    gc.collect()
    gc.collect()


def pytest_configure():
    pytest.suppress = suppress
    pytest.gc_collect = gc_collect


# Platform tools


PLAT = sys.platform
IMPL = platform.python_implementation()
PYVM = sys.version_info.major
PYN = "py{}".format(PYVM)


@pytest.fixture
def PY2():
    return PYVM == 2


# Markers


def pytest_collection_modifyitems(items):
    for item in items:
        names = set(mark.name for mark in item.iter_markers())
        if "unix" in names:
            if "linux" not in names:
                item.add_marker(pytest.mark.linux)
            if "darwin" not in names:
                item.add_marker(pytest.mark.darwin)
        if "pypy" in names:
            if "pypy2" not in names:
                item.add_marker(pytest.mark.pypy2)
            if "pypy3" not in names:
                item.add_marker(pytest.mark.pypy3)

        if "xfail_{}".format(IMPL.lower()) in names:
            item.add_marker(
                pytest.mark.xfail(reason="expected to fail on {}".format(IMPL))
            )
        if "xfail_{}".format(PLAT.lower()) in names:
            item.add_marker(
                pytest.mark.xfail(reason="expected to fail on {}".format(PLAT))
            )
        if "xfail_{}".format(PYN.lower()) in names:
            item.add_marker(
                pytest.mark.xfail(reason="expected to fail on Python {}".format(PYVM))
            )
        if "xfail_pypy{}".format(PYVM) in names and PLAT == "PyPy":
            item.add_marker(
                pytest.mark.xfail(reason="expected to fail on PyPy {}".format(PYVM))
            )


ALL_PLAT = {"darwin", "linux", "win32"}
ALL_IMPL = {"cpython", "pypy2", "pypy3"}
ALL_PY = {"py2", "py3"}


def pytest_runtest_setup(item):
    names = {mark.name for mark in item.iter_markers()}

    supported_platforms = ALL_PLAT.intersection(names)
    if supported_platforms and PLAT not in supported_platforms:
        pytest.skip("cannot run on platform {}".format(PLAT))

    supported_impls = ALL_IMPL.intersection(names)
    impl = IMPL.lower() + (str(PYVM) if IMPL == "PyPy" else "")
    if supported_impls and impl not in supported_impls:
        pytest.skip("cannot run on implementation {}".format(impl))

    supported_pythons = ALL_PY.intersection(names)
    if supported_pythons and PYN.lower() not in supported_pythons:
        pytest.skip("cannot run on Python {}".format(PYVM))
