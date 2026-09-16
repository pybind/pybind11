#!/bin/sh
# include-what-you-use wrapper for the "iwyu" CMake preset.
#
# Builds the --check_also list from the headers on disk, so a new header is
# covered without an edit here. Two headers make IWYU 0.26 crash while it
# analyzes them, so they are skipped:
#   include/pybind11/pybind11.h  (assertion: "There should be a redecl
#                                 specifying the default arg")
#   include/pybind11/cast.h      (segmentation fault)
set -eu
root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)

set -- -Xiwyu --no_fwd_decls "$@"
set -- -Xiwyu "--mapping_file=$root/tools/iwyu.imp" "$@"
for h in "$root"/include/pybind11/*.h "$root"/include/pybind11/*/*.h; do
    case $h in
    */pybind11/pybind11.h | */pybind11/cast.h) continue ;;
    esac
    set -- -Xiwyu "--check_also=$h" "$@"
done

exec include-what-you-use "$@"
