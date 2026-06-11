# CLAUDE.md

This file provides guidance to agentic harnesses when working with code in this repository.

## What this is

pybind11 is a header-only C++ library (this is v3) that exposes C++ types to Python and vice versa. The "library" is entirely the headers under `include/pybind11/`; there is nothing to compile or link for a consumer. Everything else in the repo — the Python package, CMake tooling, and tests — exists to package, distribute, and verify those headers.

The version of record lives in `include/pybind11/detail/common.h` (as `PYBIND11_VERSION_*` macros); `pybind11/_version.py` parses it from there.

## Build & test

The C++ tests need a compiler; the recommended flow uses CMake presets + `uv`:

```bash
cmake --workflow venv          # set up a .venv and run all tests (config + build + check)
```

Break it apart to control Python version or target:

```bash
cmake --preset venv -DPYBIND11_CREATE_WITH_UV=3.13t   # configure (e.g. free-threaded 3.13)
cmake --build --preset venv                           # build the test extension modules
cmake --build --preset venv -t cpptest                # build/run just the C++ (Catch2) tests
```

The `default` preset uses an existing Python/venv rather than creating one. Presets default to Ninja, Debug, `PYBIND11_WERROR=ON`, `DOWNLOAD_CATCH=ON`, `DOWNLOAD_EIGEN=ON`, and export `compile_commands.json`.

Manual full setup (no presets):

```bash
cmake -S . -B build -DDOWNLOAD_CATCH=ON -DDOWNLOAD_EIGEN=ON
cmake --build build -j4
```

### Test targets

Run via `cmake --build build --target <name>`:

- `check` — everything (pytest + cpptest + cmake build tests)
- `pytest` — Python tests only
- `cpptest` — C++ Catch2 tests only
- `test_cmake_build` — install / add_subdirectory integration tests

Tests come in pairs: `tests/test_<name>.cpp` (binds C++ test fixtures into a module) and `tests/test_<name>.py` (exercises them with pytest). To build only a subset, configure with `-DPYBIND11_TEST_OVERRIDE="test_callbacks;test_pickling"` (names without extension; empty = all).

Run a single Python test directly (the `.so` is not installed into the venv, so run from the build tree's `tests/` dir):

```bash
cd build && source .venv/bin/activate && cd tests
python -m pytest test_callbacks.py -k some_test
```

Pass pytest flags via `PYTEST_ADDOPTS`, e.g. `env PYTEST_ADDOPTS="-s -x" cmake --build build --target pytest`.

### nox shortcuts (minimal setup, slower)

`nox -s lint`, `nox -s tests-3.9`, `nox -s docs -- serve`, `nox -s build`. With `pipx run nox` you don't even need nox installed.

## Linting

Use `prek -a --quiet` (the user's preferred wrapper over `pre-commit run -a`). Pre-commit handles all formatting and most lint. `gawk` is required for some hooks.

C++ formatting (`.clang-format`) is NOT auto-applied — run `clang-format -style=file -i some.cpp` manually on new/changed C++. clang-tidy support is built into CMake (`cmake --preset tidy`), normally run in CI/Docker, not locally.

## Architecture of the headers

`include/pybind11/pybind11.h` is the main entry point that most users include; it pulls in the core machinery. Key layers:

- **`detail/`** — internal implementation, not part of the public API. This is where the hard parts live:
  - `type_caster_base.h`, `cast.h` (`include/pybind11/cast.h`) — the type conversion system that maps C++ ↔ Python values. Most binding behavior ultimately routes through type casters.
  - `internals.h` — the per-interpreter global state (registered types, instances, etc.), shared across modules via a capsule. ABI compatibility is gated by an ID; changing internals layout is an ABI break.
  - `struct_smart_holder.h` / `using_smart_holder.h` — the **smart_holder** that became the default holder in v3, enabling safe passing of objects between `shared_ptr`/`unique_ptr` and Python ownership.
  - `class.h`, `init.h` — class registration and constructor (`py::init`) machinery.
- **`pytypes.h`** — C++ wrappers for Python objects (`py::object`, `py::dict`, `py::str`, …) with reference counting.
- **Feature headers** — opt-in includes: `stl.h` / `stl/` / `stl_bind.h` (STL conversions), `numpy.h` + `eigen/` (array/matrix interop), `functional.h` (std::function), `chrono.h`, `complex.h`, `eval.h`, `embed.h` (embedding the interpreter), `iostream.h`, `gil.h` / `gil_safe_call_once.h`, `subinterpreter.h`, `native_enum.h`, `typing.h`, `warnings.h`.
- **`conduit/`** — `pybind11_conduit_v1`, a stable cross-binding-framework protocol for sharing C++ objects between independently-built extension modules (even different pybind11 versions / other frameworks).

Because consumers can mix modules built against different pybind11 versions in one process, **ABI stability is a first-class concern** — be deliberate about anything touching `internals.h`, holders, or the platform ABI id (`conduit/pybind11_platform_abi_id.h`).

## The Python package (`pybind11/`)

Separate from the C++ library: it ships the headers and CMake config so downstream projects can build extensions. Notable modules:

- `setup_helpers.py` — `Pybind11Extension` / `build_ext` and the `intree_extensions` helper for setuptools-based builds; this file is intentionally standalone-copyable.
- `commands.py` / `__main__.py` — `python -m pybind11 --includes` / `--cmakedir` etc., used by build systems to locate headers and CMake files.

Packaging produces two distributions: the normal `pybind11` (headers live inside the package, found via the functions above) and `pybind11-global` (installs to `<env>/include/pybind11` and `<env>/share/cmake/pybind11` for system-wide CMake discovery). Build with `nox -s build` and `nox -s build_global`. The build backend is scikit-build-core. Packaging tests live in `tests/extra_python_package` (`nox -s tests_packaging`).

## CMake tooling (`tools/`)

`pybind11Common.cmake`, `pybind11Tools.cmake` (classic FindPythonLibs path), and `pybind11NewTools.cmake` (CMake 3.12+ FindPython path, selected with `-DPYBIND11_FINDPYTHON=ON`) implement the `pybind11_add_module()` interface and the `pybind11::*` interface targets that downstream `CMakeLists.txt` files consume.

## Conventions

- Small, self-contained PRs; this project deliberately favors minimal-code general solutions.
- Any new functionality needs a test (add to or create the paired `.cpp`/`.py` in `tests/`, and register the test in `tests/CMakeLists.txt`).
- Try to add to an existing test file if possible (more files slow down tests)
- Default C++ standard follows the consumer's toolchain; CI exercises a wide matrix (CPython 3.8+, PyPy, GraalPy; multiple compilers and C++ standards). Keep changes portable across that matrix.
