#!/bin/bash
set -euo pipefail

# Build and run CPython 3.14t move_subinterpreter_redux.c
#
# Usage:
#   ./build_and_run.sh /path/to/python3.14t-config
#
# Example:
#   ./build_and_run.sh "$HOME/wrk/cpython_installs/v3.14_57e0d177c26/bin/python3.14t-config"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /full/path/to/pythonX.Y[t]-config" >&2
    exit 1
fi

PYTHON_CONFIG="$1"
if [ ! -x "$PYTHON_CONFIG" ]; then
    echo "Error: $PYTHON_CONFIG is not executable" >&2
    exit 1
fi

CC="${CC:-gcc}"
CFLAGS="$($PYTHON_CONFIG --cflags)"
LDFLAGS="$($PYTHON_CONFIG --embed --ldflags)"
LIBS="$($PYTHON_CONFIG --embed --libs)"

src_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$src_dir"

rm -f move_subinterpreter_redux

echo "Building move_subinterpreter_redux with: $PYTHON_CONFIG" >&2
set -x
# shellcheck disable=SC2086  # CFLAGS/LDFLAGS/LIBS need word splitting
"$CC" -O0 -g -Wall -Wextra -o move_subinterpreter_redux move_subinterpreter_redux.c $CFLAGS $LDFLAGS $LIBS -lpthread
set -x

prefix="$($PYTHON_CONFIG --prefix)"
export LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "Running move_subinterpreter_redux..." >&2
set -x

# Temporarily disable 'exit on error' so we can inspect the exit code.
set +e
timeout 3s ./move_subinterpreter_redux
status=$?
set -e
set +x

if [ "$status" -eq 124 ]; then
    echo "move_subinterpreter_redux: TIMED OUT after 3s" >&2
elif [ "$status" -eq 0 ]; then
    echo "move_subinterpreter_redux: finished successfully (exit code 0)" >&2
else
    echo "move_subinterpreter_redux: finished with exit code $status" >&2
fi

exit "$status"
