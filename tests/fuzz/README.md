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

The target is off unless requested, so the normal test matrix does not need
libFuzzer:

```bash
cmake -S . -B build -DPYBIND11_BUILD_FUZZERS=ON
cmake --build build --target pybind_fuzzer
cmake --build build --target fuzz_replay
```

`fuzz_replay` runs `LLVMFuzzerTestOneInput` on every file in
`tests/fuzz/corpus`. To reproduce one OSS-Fuzz artifact:

```bash
./build/tests/fuzz/pybind_fuzzer /path/to/crash
```

OSS-Fuzz sets `LIB_FUZZING_ENGINE` and links that instead of `fuzz_replay.cc`.
The callback is the same either way. Standalone replay uses the same
`ConsumeRandomLengthString` length split, so the checked-in corpus exercises
the callback without a libFuzzer toolchain. An artifact produced by OSS-Fuzz
should be rerun with libFuzzer (`-runs=1 <file>`); the standalone binary is the
maintainer path for inputs in `tests/fuzz/corpus` and for files reduced to the
same framing.

## Reducing a crash

1. Rebuild with `-DPYBIND11_BUILD_FUZZERS=ON` and rerun the artifact.
2. Delete trailing bytes until the crash disappears, then restore the last
   required byte. Repeat from the front if the first string is not involved.
3. The remaining strings are, in order: code passed to `py::exec`, the `os`
   attribute name, the `Decimal` argument, the `Decimal` method name, and the
   `py::str` argument.
4. Turn the surviving call into a normal `tests/test_*.py` regression. If the
   crash is only reachable through embedding, add it beside
   `tests/test_with_catch` instead.

Arbitrary `py::exec` is intentionally broad and is the behavior of the current
OSS-Fuzz target. A later harness can drive explicit `class_`, `def`, and caster
operations; that should be a separate target so this move does not change what
OSS-Fuzz is already running.
