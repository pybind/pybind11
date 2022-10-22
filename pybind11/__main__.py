# pylint: disable=missing-function-docstring

import argparse
import sys
import sysconfig
from pathlib import Path

from .commands import (
    get_cflags,
    get_cmake_dir,
    get_extension,
    get_include,
    get_ldflags,
    get_pkgconfig_dir,
)


def print_includes() -> None:
    dirs = [
        sysconfig.get_path("include"),
        sysconfig.get_path("platinclude"),
        get_include(),
    ]

    # Make unique but preserve order
    unique_dirs = []
    for d in dirs:
        if d and d not in unique_dirs:
            unique_dirs.append(d)

    print(" ".join("-I" + d for d in unique_dirs))


def main() -> None:

    parser = argparse.ArgumentParser()
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
        "--cflags",
        action="store_true",
        help="Print the simple compile flags",
    )
    parser.add_argument(
        "--ldflags",
        action="store_true",
        help="Print the simple link flags",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Show embed version instead of extension",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Print the suggested compile line for a given file",
    )
    args = parser.parse_args()
    if not sys.argv[1:]:
        parser.print_help()
    if args.cflags or args.file:
        print(get_cflags(), end=" " if args.file else "\n")
    elif args.includes and not args.file:
        print_includes()
    if args.file:
        print(args.file.name, end=" ")
    if args.ldflags or args.file:
        print(get_ldflags(args.embed), end=" " if args.file else "\n")
    if args.file:
        if args.embed:
            print("-o", args.file.with_suffix(""))
        else:
            print("-o", args.file.with_suffix(get_extension()))

    if args.cmakedir:
        print(get_cmake_dir())
    if args.pkgconfigdir:
        print(get_pkgconfig_dir())


if __name__ == "__main__":
    main()
