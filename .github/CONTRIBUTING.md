Thank you for your interest in this project! Please refer to the following
sections on how to contribute code and bug reports.

### Reporting bugs

Before submitting a question or bug report, please take a moment of your time
and ensure that your issue isn't already discussed in the project documentation
provided at [pybind11.readthedocs.org][] or in the [issue tracker][]. You can
also check [gitter][] to see if it came up before.

Assuming that you have identified a previously unknown problem or an important
question, it's essential that you submit a self-contained and minimal piece of
code that reproduces the problem. In other words: no external dependencies,
isolate the function(s) that cause breakage, submit matched and complete C++
and Python snippets that can be easily compiled and run in isolation; or
ideally make a small PR with a failing test case that can be used as a starting
point.

## Pull requests

Contributions are submitted, reviewed, and accepted using GitHub pull requests.
Please refer to [this article][using pull requests] for details and adhere to
the following rules to make the process as smooth as possible:

* Make a new branch for every feature you're working on.
* Make small and clean pull requests that are easy to review but make sure they
  do add value by themselves.
* Add tests for any new functionality and run the test suite (`cmake --workflow
  venv`) to ensure that no existing features break.
* Please run [`pre-commit`][pre-commit] to check your code matches the
  project style. (Note that `gawk` is required.) Use `pre-commit run
  --all-files` before committing (or use installed-mode, check pre-commit docs)
  to verify your code passes before pushing to save time.
* This project has a strong focus on providing general solutions using a
  minimal amount of code, thus small pull requests are greatly preferred.

### Licensing of contributions

pybind11 is provided under a BSD-style license that can be found in the
``LICENSE`` file. By using, distributing, or contributing to this project, you
agree to the terms and conditions of this license.

You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to the author of this software, without
imposing a separate written license agreement for such Enhancements, then you
hereby grant the following license: a non-exclusive, royalty-free perpetual
license to install, use, modify, prepare derivative works, incorporate into
other computer software, distribute, and sublicense such enhancements or
derivative works thereof, in binary and source code form.


## Development of pybind11

### Quick setup

To setup a quick development environment, use [`nox`](https://nox.thea.codes).
This will allow you to do some common tasks with minimal setup effort, but will
take more time to run and be less flexible than a full development environment.
If you use [`pipx run nox`](https://pipx.pypa.io), you don't even need to
install `nox`. Examples:

```bash
# List all available sessions
nox -l

# Run linters
nox -s lint

# Run tests on Python 3.9
nox -s tests-3.9

# Build and preview docs
nox -s docs -- serve

# Build SDists and wheels
nox -s build
```

### Full setup

To setup an ideal development environment, run the following commands on a
system with CMake 3.15+:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r tests/requirements.txt
cmake -S . -B build -DDOWNLOAD_CATCH=ON -DDOWNLOAD_EIGEN=ON
cmake --build build -j4
```

Tips:

* You can use `virtualenv` (faster, from PyPI) instead of `venv`.
* You can select any name for your environment folder; if it contains "env" it
  will be ignored by git.
* If you don't have CMake 3.15+, just add "cmake" to the pip install command.
* You can use `-DPYBIND11_FINDPYTHON=ON` to use FindPython.
* For a specific Python, you can use `-DPython_ROOT_DIR=/path/to` or
  `-DPython_EXECUTABLE=/path/to/python`.

## CMake presets

We also support CMake presets. If you have [uv](https://docs.astral.sh/uv/),
you can use:

```bash
cmake --workflow venv
```

to setup a venv and run all tests. You can break this up into components
if you want to use a specific version of Python (or any other config option) or
build only one of the valid targets (listed below).

```bash
cmake --preset venv -DPYBIND11_CREATE_WITH_UV=3.13t
cmake --build --preset venv
cmake --build --preset venv -t cpptest
```

The `default` preset will use an existing venv or Python install. If you'd like
to run pytest yourself, say to easily control the options:

```bash
cd build
source .venv/bin/activate
cd tests
python -m pytest
```

The `.so` file is not installed into the venv, so you need to run from this
directory, the local directory is included with `python -m`.

## Configuration options

In CMake, configuration options are given with "-D". Options are stored in the
build directory, in the `CMakeCache.txt` file, so they are remembered for each
build directory. Two selections are special - the generator, given with `-G`,
and the compiler, which is selected based on environment variables `CXX` and
similar, or `-DCMAKE_CXX_COMPILER=`. Unlike the others, these cannot be changed
after the initial run.

The valid options are:

* `-DCMAKE_BUILD_TYPE`: Release, Debug, MinSizeRel, RelWithDebInfo
* `-DPYBIND11_FINDPYTHON=ON`: Use CMake 3.12+'s FindPython instead of the
  classic, deprecated, custom FindPythonLibs
* `-DPYBIND11_NOPYTHON=ON`: Disable all Python searching (disables tests)
* `-DBUILD_TESTING=ON`: Enable the tests
* `-DDOWNLOAD_CATCH=ON`: Download catch to build the C++ tests
* `-DDOWNLOAD_EIGEN=ON`: Download Eigen for the NumPy tests
* `-DPYBIND11_INSTALL=ON/OFF`: Enable the install target (on by default for the
  master project)
* `-DUSE_PYTHON_INSTALL_DIR=ON`: Try to install into the python dir


<details><summary>A few standard CMake tricks: (click to expand)</summary><p>

* Use `cmake --build build -v` to see the commands used to build the files.
* Use `cmake build -LH` to list the CMake options with help.
* Use `ccmake` if available to see a curses (terminal) gui, or `cmake-gui` for
  a completely graphical interface (not present in the PyPI package).
* Use `cmake --build build -j12` to build with 12 cores (for example).
* Use `-G` and the name of a generator to use something different. `cmake
  --help` lists the generators available.
      - On Unix, setting `CMAKE_GENERATER=Ninja` in your environment will give
        you automatic multithreading on all your CMake projects!
* Open the `CMakeLists.txt` with QtCreator to generate for that IDE.
* You can use `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` to generate the `.json` file
  that some tools expect.

</p></details>


To run the tests, you can "build" the check target:

```bash
cmake --build build --target check
```

`--target` can be spelled `-t`. You can also run individual tests with these
targets:

* `pytest`: Python tests only, using the
[pytest](https://docs.pytest.org/en/stable/) framework
* `cpptest`: C++ tests only
* `test_cmake_build`: Install / subdirectory tests

If you want to build just a subset of tests, use
`-DPYBIND11_TEST_OVERRIDE="test_callbacks;test_pickling"`. If this is
empty, all tests will be built. Tests are specified without an extension if they need both a .py and
.cpp file.

You may also pass flags to the `pytest` target by editing `tests/pytest.ini` or
by using the `PYTEST_ADDOPTS` environment variable
(see [`pytest` docs](https://docs.pytest.org/en/2.7.3/customize.html#adding-default-options)). As an example:

```bash
env PYTEST_ADDOPTS="--capture=no --exitfirst" \
    cmake --build build --target pytest
# Or using abbreviated flags
env PYTEST_ADDOPTS="-s -x" cmake --build build --target pytest
```

### Formatting

All formatting is handled by pre-commit.

Install with brew (macOS) or pip (any OS):

```bash
# Any OS
python3 -m pip install pre-commit

# OR macOS with homebrew:
brew install pre-commit
```

Then, you can run it on the items you've added to your staging area, or all
files:

```bash
pre-commit run
# OR
pre-commit run --all-files
```

And, if you want to always use it, you can install it as a git hook (hence the
name, pre-commit):

```bash
pre-commit install
```

### Clang-Format

As of v2.6.2, pybind11 ships with a [`clang-format`][clang-format]
configuration file at the top level of the repo (the filename is
`.clang-format`). Currently, formatting is NOT applied automatically, but
manually using `clang-format` for newly developed files is highly encouraged.
To check if a file needs formatting:

```bash
clang-format -style=file --dry-run some.cpp
```

The output will show things to be fixed, if any. To actually format the file:

```bash
clang-format -style=file -i some.cpp
```

Note that the `-style-file` option searches the parent directories for the
`.clang-format` file, i.e. the commands above can be run in any subdirectory
of the pybind11 repo.

### Clang-Tidy

[`clang-tidy`][clang-tidy] performs deeper static code analyses and is
more complex to run, compared to `clang-format`, but support for `clang-tidy`
is built into the pybind11 CMake configuration. To run `clang-tidy`, the
following recipe should work. Run the `docker` command from the top-level
directory inside your pybind11 git clone.

```bash
docker run --rm -v $PWD:/pybind11 -w /pybind11 -it silkeh/clang:20
apt-get update && apt-get install -y git python3-dev python3-pytest ninja-build
cmake --preset tidy
cmake --build --preset tidy
```

You can add `--fix` to the options list in the preset if you want to apply fixes
(remember `-j1` to run only one thread).

### Include what you use

To run include what you use, install (`brew install include-what-you-use` on
macOS), then run:

```bash
cmake -S . -B build-iwyu -DCMAKE_CXX_INCLUDE_WHAT_YOU_USE=$(which include-what-you-use)
cmake --build build-iwyu
```

The report is sent to stderr; you can pipe it into a file if you wish.

### Build recipes

This builds with the Intel compiler (assuming it is in your path, along with a
recent CMake and Python):

```bash
python3 -m venv venv
. venv/bin/activate
pip install pytest
cmake -S . -B build-intel -DCMAKE_CXX_COMPILER=$(which icpc) -DDOWNLOAD_CATCH=ON -DDOWNLOAD_EIGEN=ON -DPYBIND11_WERROR=ON
```

This will test the PGI compilers:

```bash
docker run --rm -it -v $PWD:/pybind11 nvcr.io/hpc/pgi-compilers:ce
apt-get update && apt-get install -y python3-dev python3-pip python3-pytest
wget -qO- "https://cmake.org/files/v3.18/cmake-3.18.2-Linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local
cmake -S pybind11/ -B build
cmake --build build
```

### Explanation of the SDist/wheel building design

> These details below are _only_ for packaging the Python sources from git. The
> SDists and wheels created do not have any extra requirements at all and are
> completely normal.

The main objective of the packaging system is to create SDists (Python's source
distribution packages) and wheels (Python's binary distribution packages) that
include everything that is needed to work with pybind11, and which can be
installed without any additional dependencies. This is more complex than it
appears: in order to support CMake as a first class language even when using
the PyPI package, they must include the _generated_ CMake files (so as not to
require CMake when installing the `pybind11` package itself). They should also
provide the option to install to the "standard" location
(`<ENVROOT>/include/pybind11` and `<ENVROOT>/share/cmake/pybind11`) so they are
easy to find with CMake, but this can cause problems if you are not an
environment or using ``pyproject.toml`` requirements. This was solved by having
two packages; the "nice" pybind11 package that stores the includes and CMake
files inside the package, that you get access to via functions in the package,
and a `pybind11-global` package that can be included via `pybind11[global]` if
you want the more invasive but discoverable file locations.

If you want to package the GitHub source for the "global" package, you need
to use nox. Normal packaging will only make the normal package.


```bash
nox -s build
nox -s build_global
```


[pre-commit]: https://pre-commit.com
[clang-format]: https://clang.llvm.org/docs/ClangFormat.html
[clang-tidy]: https://clang.llvm.org/extra/clang-tidy/
[pybind11.readthedocs.org]: http://pybind11.readthedocs.org/en/latest
[issue tracker]: https://github.com/pybind/pybind11/issues
[gitter]: https://gitter.im/pybind/Lobby
[using pull requests]: https://help.github.com/articles/using-pull-requests
