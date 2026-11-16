# Embedding fuzzer

`pybind_fuzzer` is the existing OSS-Fuzz harness, moved here so it can be
reviewed and updated with the code it exercises. It does not define pybind11
bindings. Each input is split into strings and used to:

- start the embedded interpreter once, then run `py::exec`
- import `os` and look up an attribute
- construct `decimal.Decimal` and call a method on it
- construct a `py::str`

`py::error_already_set` is caught. That is expected for invalid Python, a
missing attribute, or a bad `Decimal` argument. A failure worth reporting is a
crash, sanitizer error, or hang in pybind11's embedding layer or in CPython
reached through that layer.

## Build and replay

The target is off unless requested. Replay needs a C++ compiler, CPython's
embedding library, and LLVM's `fuzzer/FuzzedDataProvider.h`; it does not need
libFuzzer:

```bash
cmake -S . -B build -DPYBIND11_BUILD_FUZZERS=ON
cmake --build build --target pybind_fuzzer
cmake --build build --target fuzz_replay
```

CMake finds the header through `clang -print-resource-dir`. If Clang is not
on `PATH`, set `-DPYBIND11_FUZZER_INCLUDE_DIR=/path/to/include`, where that
directory contains `fuzzer/FuzzedDataProvider.h`. GCC can use this header too.

`fuzz_replay` first checks the decoded fields of the `exec_success` seed,
then runs `LLVMFuzzerTestOneInput` on every file in `tests/fuzz/corpus`.
To reproduce one OSS-Fuzz artifact with the standalone binary:

```bash
./build/tests/fuzz/pybind_fuzzer /path/to/crash
```

OSS-Fuzz sets `LIB_FUZZING_ENGINE` and links that instead of `fuzz_replay.cpp`.
Both builds use the same callback and input decoder. When reproducing a
sanitizer finding, enable the same sanitizer in the replay build. The
libFuzzer binary can also replay one artifact with `-runs=1 <file>`.

## Input format

The decoder reads five strings, in order: Python code, an `os` attribute
name, a `Decimal` argument, a `Decimal` method name, and text for `py::str`.
It calls `FuzzedDataProvider::ConsumeRandomLengthString` once for each field.
A backslash followed by any non-backslash byte ends a field; two consecutive
backslashes encode a literal backslash. The last field can end at EOF.
For example, the `exec_success` seed contains these bytes:

```text
x = 1\!getcwd\!1\!as_tuple\!ok\!
```

This upstream harness uses five fields. The older OSS-Fuzz copy consumed an
additional, independent string for the unused tuple argument, so artifacts
from that version need the matching older harness for reproduction.

## Reducing a crash

1. Rebuild with `-DPYBIND11_BUILD_FUZZERS=ON` and rerun the artifact.
2. Reduce the decoded strings while preserving their separators, rerunning
   the input after each change. Empty fields still need separators when a
   later field is involved.
3. Turn the surviving call into a normal `tests/test_*.py` regression. If the
   crash is only reachable through embedding, add it beside
   `tests/test_with_catch` instead.

Arbitrary `py::exec` is intentionally broad and retains the operations of the
existing OSS-Fuzz target. A later harness can drive explicit `class_`, `def`,
and caster operations as a separate target.
