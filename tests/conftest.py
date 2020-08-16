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


# Platform Markers

PLAT = sys.platform.lower()  # linux, osx, or win32
IMPL = platform.python_implementation().lower()  # cpython or pypy


def pytest_collection_modifyitems(items):
    """
    This will find the markers listed in pytest.ini and add
    skip or xfail as needed. The second part can be a platform
    marker (linux, osx, or win32), or python, cpython, or pypy.

    You can add also add Python version numbers - either 2 or 3.

    Keyword arguments are passed on.

    # Will skip on Python 2
    pytest.mark.skip_python(2)

    # Will xfail on pypy as long as TypeError is raised
    pytest.mark.xfail_pypy(reason="Not supported", raises=TypeError)
    """
    for item in items:
        for mark in tuple(item.iter_markers()):
            # Check for recognised name
            parts = mark.name.split("_")
            if len(parts) == 2 and parts[0] in {"xfail", "skip"}:
                marker = getattr(pytest.mark, parts[0])

                if parts[1] in {PLAT, IMPL, "python"}:
                    # args lets you remove only Py 2 or 3
                    if mark.args:
                        (ver,) = mark.args  # Only single argument supported
                        assert isinstance(ver, int), "should be a version number"
                        if ver == sys.version_info.major:
                            item.add_marker(marker(**mark.kwargs))
                    else:
                        assert parts[1] != "python", "version required (otherwise use mark.skip)"
                        item.add_marker(marker(**mark.kwargs))
