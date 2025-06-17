# This file will be replaced in the wheel with a hard-coded version. This only
# exists to allow running directly from source without installing (not
# recommended, but supported).

from __future__ import annotations

import re
from pathlib import Path

DIR = Path(__file__).parent.resolve()

input_file = DIR.parent / "include/pybind11/detail/common.h"
regex = re.compile(
    r"""
\#define \s+ PYBIND11_VERSION_MAJOR \s+ (?P<major>\d+) .*?
\#define \s+ PYBIND11_VERSION_MINOR \s+ (?P<minor>\d+) .*?
\#define \s+ PYBIND11_VERSION_PATCH \s+ (?P<patch>\S+)
""",
    re.MULTILINE | re.DOTALL | re.VERBOSE,
)

match = regex.search(input_file.read_text(encoding="utf-8"))
assert match, "Unable to find version in pybind11/detail/common.h"
__version__ = "{major}.{minor}.{patch}".format(**match.groupdict())


def _to_int(s: str) -> int | str:
    try:
        return int(s)
    except ValueError:
        return s


version_info = tuple(_to_int(s) for s in __version__.split("."))
