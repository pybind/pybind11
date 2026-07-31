# pylint: disable=missing-function-docstring
from __future__ import annotations

import argparse
import functools
import sys
import sysconfig
from pathlib import Path

from ._version import __version__
from .commands import (
    _quote as quote,
)
from .commands import (
    get_cflags,
    get_cmake_dir,
    get_include_dirs,
    get_ldflags,
    get_pkgconfig_dir,
)


def print_includes() -> None:
    print(" ".join(quote(f"-I{d}") for d in get_include_dirs()))


def main() -> None:
    make_parser = functools.partial(argparse.ArgumentParser, allow_abbrev=False)
    if sys.version_info >= (3, 14):
        make_parser = functools.partial(make_parser, color=True, suggest_on_error=True)
    parser = make_parser()
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print the version and exit.",
    )
    parser.add_argument(
        "--includes",
        action="store_true",
        help="Include flags for both pybind11 and Python headers.",
    )
    parser.add_argument(
        "--cmakedir",
        action="store_true",
        help="Print the CMake module directory, ideal for setting -Dpybind11_ROOT in CMake.",
    )
    parser.add_argument(
        "--pkgconfigdir",
        action="store_true",
        help="Print the pkgconfig directory, ideal for setting $PKG_CONFIG_PATH.",
    )
    parser.add_argument(
        "--extension-suffix",
        action="store_true",
        help="Print the extension for a Python module",
    )
    parser.add_argument(
        "--cflags",
        action="store_true",
        help="Print the compile flags for a simple extension (Unix-style compilers).",
    )
    parser.add_argument(
        "--ldflags",
        action="store_true",
        help="Print the link flags for a simple extension (Unix-style compilers).",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Build for embedding instead of an extension; affects --ldflags and --file.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Print a full command-line suffix for compiling the given file;"
        " the output goes next to the source file (Unix-style compilers).",
    )
    args = parser.parse_args()
    if not sys.argv[1:]:
        parser.print_help()
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
    if args.file:
        suffix = "" if args.embed else ext_suffix
        print(
            get_cflags(),
            quote(str(args.file)),
            get_ldflags(embed=args.embed),
            "-o",
            quote(str(args.file.with_suffix(suffix))),
        )
    else:
        if args.cflags:
            print(get_cflags())
        if args.ldflags:
            print(get_ldflags(embed=args.embed))
    # --cflags and --file already contain the include flags
    if args.includes and not (args.cflags or args.file):
        print_includes()
    if args.cmakedir:
        print(quote(get_cmake_dir()))
    if args.pkgconfigdir:
        print(quote(get_pkgconfig_dir()))
    if args.extension_suffix:
        print(ext_suffix)


if __name__ == "__main__":
    main()
