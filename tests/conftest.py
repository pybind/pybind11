"""pytest configuration

Extends output capture as needed by pybind11: ignore constructors, optional unordered lines.
Adds docstring and exceptions message sanitizers.
"""

from __future__ import annotations

import contextlib
import difflib
import gc
import importlib.metadata
import multiprocessing
import re
import sys
import sysconfig
import textwrap
import traceback
import weakref
from typing import Callable

import pytest

# Early diagnostic for failed imports
try:
    import pybind11_tests
except Exception:
    # pytest does not show the traceback without this.
    traceback.print_exc()
    raise


@pytest.fixture(scope="session", autouse=True)
def use_multiprocessing_forkserver_on_linux():
    if sys.platform != "linux" or sys.implementation.name == "graalpy":
        # The default on Windows, macOS and GraalPy is "spawn": If it's not broken, don't fix it.
        return

    # Full background: https://github.com/pybind/pybind11/issues/4105#issuecomment-1301004592
    # In a nutshell: fork() after starting threads == flakiness in the form of deadlocks.
    # It is actually a well-known pitfall, unfortunately without guard rails.
    # "forkserver" is more performant than "spawn" (~9s vs ~13s for tests/test_gil_scoped.py,
    # visit the issuecomment link above for details).
    multiprocessing.set_start_method("forkserver")


_long_marker = re.compile(r"([0-9])L")
_hexadecimal = re.compile(r"0x[0-9a-fA-F]+")

# Avoid collecting Python3 only files
collect_ignore = []


def _strip_and_dedent(s):
    """For triple-quote strings"""
    return textwrap.dedent(s.lstrip("\n").rstrip())


def _split_and_sort(s):
    """For output which does not require specific line order"""
    return sorted(_strip_and_dedent(s).splitlines())


def _make_explanation(a, b):
    """Explanation for a failed assert -- the a and b arguments are List[str]"""
    return ["--- actual / +++ expected"] + [
        line.strip("\n") for line in difflib.ndiff(a, b)
    ]


class Output:
    """Basic output post-processing and comparison"""

    def __init__(self, string):
        self.string = string
        self.explanation = []

    def __str__(self):
        return self.string

    __hash__ = None

    def __eq__(self, other):
        # Ignore constructor/destructor output which is prefixed with "###"
        a = [
            line
            for line in self.string.strip().splitlines()
            if not line.startswith("###")
        ]
        b = _strip_and_dedent(other).splitlines()
        if a == b:
            return True
        self.explanation = _make_explanation(a, b)
        return False


class Unordered(Output):
    """Custom comparison for output without strict line ordering"""

    __hash__ = None

    def __eq__(self, other):
        a = _split_and_sort(self.string)
        b = _split_and_sort(other)
        if a == b:
            return True
        self.explanation = _make_explanation(a, b)
        return False


class Capture:
    def __init__(self, capfd):
        self.capfd = capfd
        self.out = ""
        self.err = ""

    def __enter__(self):
        self.capfd.readouterr()
        return self

    def __exit__(self, *args):
        self.out, self.err = self.capfd.readouterr()

    __hash__ = None

    def __eq__(self, other):
        a = Output(self.out)
        b = other
        if a == b:
            return True
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


class SanitizedString:
    def __init__(self, sanitizer):
        self.sanitizer = sanitizer
        self.string = ""
        self.explanation = []

    def __call__(self, thing):
        self.string = self.sanitizer(thing)
        return self

    __hash__ = None

    def __eq__(self, other):
        a = self.string
        b = _strip_and_dedent(other)
        if a == b:
            return True
        self.explanation = _make_explanation(a.splitlines(), b.splitlines())
        return False


def _sanitize_general(s):
    s = s.strip()
    s = s.replace("pybind11_tests.", "m.")
    return _long_marker.sub(r"\1", s)


def _sanitize_docstring(thing):
    s = thing.__doc__
    return _sanitize_general(s)


@pytest.fixture
def doc():
    """Sanitize docstrings and add custom failure explanation"""
    return SanitizedString(_sanitize_docstring)


def _sanitize_message(thing):
    s = str(thing)
    s = _sanitize_general(s)
    return _hexadecimal.sub("0", s)


@pytest.fixture
def msg():
    """Sanitize messages and add custom failure explanation"""
    return SanitizedString(_sanitize_message)


def pytest_assertrepr_compare(op, left, right):  # noqa: ARG001
    """Hook to insert custom failure explanation"""
    if hasattr(left, "explanation"):
        return left.explanation
    return None


# Number of times we think repeatedly collecting garbage might do anything.
# The only reason to do more than once is because finalizers executed during
# one GC pass could create garbage that can't be collected until a future one.
# This quickly produces diminishing returns, and GC passes can be slow, so this
# value is a tradeoff between non-flakiness and fast tests. (It errs on the
# side of non-flakiness; many uses of this idiom only do 3 passes.)
num_gc_collect = 5


def gc_collect():
    """Run the garbage collector several times (needed when running
    reference counting tests with PyPy)"""
    for _ in range(num_gc_collect):
        gc.collect()


def delattr_and_ensure_destroyed(*specs):
    """For each of the given *specs* (a tuple of the form ``(scope, name)``),
    perform ``delattr(scope, name)``, then do enough GC collections that the
    deleted reference has actually caused the target to be destroyed. This is
    typically used to test what happens when a type object is destroyed; if you
    use it for that, you should be aware that extension types, or all types,
    are immortal on some Python versions. See ``env.TYPES_ARE_IMMORTAL``.
    """
    wrs = []
    for mod, name in specs:
        wrs.append(weakref.ref(getattr(mod, name)))
        delattr(mod, name)

    for _ in range(num_gc_collect):
        gc.collect()
        if all(wr() is None for wr in wrs):
            break
    else:
        # If this fires, most likely something is still holding a reference
        # to the object you tried to destroy - for example, it's a type that
        # still has some instances alive. Try setting a breakpoint here and
        # examining `gc.get_referrers(wrs[0]())`. It's vaguely possible that
        # num_gc_collect needs to be increased also.
        pytest.fail(
            f"Could not delete bindings such as {next(wr for wr in wrs if wr() is not None)!r}"
        )


def pytest_configure():
    pytest.suppress = contextlib.suppress
    pytest.gc_collect = gc_collect
    pytest.delattr_and_ensure_destroyed = delattr_and_ensure_destroyed


def pytest_report_header():
    assert pybind11_tests.compiler_info is not None, (
        "Please update pybind11_tests.cpp if this assert fails."
    )
    interesting_packages = ("pybind11", "numpy", "scipy", "build")
    valid = []
    for package in sorted(interesting_packages):
        with contextlib.suppress(ModuleNotFoundError):
            valid.append(f"{package}=={importlib.metadata.version(package)}")
    reqs = " ".join(valid)

    cpp_info = [
        "C++ Info:",
        f"{pybind11_tests.compiler_info}",
        f"{pybind11_tests.cpp_std}",
        f"{pybind11_tests.PYBIND11_INTERNALS_ID}",
        f"PYBIND11_SIMPLE_GIL_MANAGEMENT={pybind11_tests.PYBIND11_SIMPLE_GIL_MANAGEMENT}",
    ]
    if "__graalpython__" in sys.modules:
        cpp_info.append(
            f"GraalPy version: {sys.modules['__graalpython__'].get_graalvm_version()}"
        )
    lines = [
        f"installed packages of interest: {reqs}",
        " ".join(cpp_info),
    ]
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        lines.append("free-threaded Python build")

    return lines


@pytest.fixture
def backport_typehints() -> Callable[[SanitizedString], SanitizedString]:
    d = {}
    if sys.version_info < (3, 13):
        d["typing_extensions.TypeIs"] = "typing.TypeIs"
        d["typing_extensions.CapsuleType"] = "types.CapsuleType"
    if sys.version_info < (3, 12):
        d["typing_extensions.Buffer"] = "collections.abc.Buffer"
    if sys.version_info < (3, 11):
        d["typing_extensions.Never"] = "typing.Never"
    if sys.version_info < (3, 10):
        d["typing_extensions.TypeGuard"] = "typing.TypeGuard"

    def backport(sanatized_string: SanitizedString) -> SanitizedString:
        for old, new in d.items():
            sanatized_string.string = sanatized_string.string.replace(old, new)

        return sanatized_string

    return backport
