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
* Add tests for any new functionality and run the test suite (`cmake --build
  build --target pytest`) to ensure that no existing features break.
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

To setup an ideal development environment, run the following commands on a
system with CMake 3.14+:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r tests/requirements.txt
cmake -S . -B build -DDOWNLOAD_CATCH=ON -DDOWNLOAD_EIGEN=ON
cmake --build build -j4
```

Tips:

* You can use `virtualenv` (from PyPI) instead of `venv` (which is Python 3
  only).
* You can select any name for your environment folder; if it contains "env" it
  will be ignored by git.
* If you don’t have CMake 3.14+, just add “cmake” to the pip install command.
* You can use `-DPYBIND11_FINDPYTHON=ON` to use FindPython on CMake 3.12+
* In classic mode, you may need to set `-DPYTHON_EXECUTABLE=/path/to/python`.
  FindPython uses `-DPython_ROOT_DIR=/path/to` or
  `-DPython_EXECUTABLE=/path/to/python`.

### Configuration options

In CMake, configuration options are given with “-D”. Options are stored in the
build directory, in the `CMakeCache.txt` file, so they are remembered for each
build directory. Two selections are special - the generator, given with `-G`,
and the compiler, which is selected based on environment variables `CXX` and
similar, or `-DCMAKE_CXX_COMPILER=`. Unlike the others, these cannot be changed
after the initial run.

The valid options are:

* `-DCMAKE_BUILD_TYPE`: Release, Debug, MinSizeRel, RelWithDebInfo
* `-DPYBIND11_FINDPYTHON=ON`: Use CMake 3.12+’s FindPython instead of the
  classic, deprecated, custom FindPythonLibs
* `-DPYBIND11_NOPYTHON=ON`: Disable all Python searching (disables tests)
* `-DBUILD_TESTING=ON`: Enable the tests
* `-DDOWNLOAD_CATCH=ON`: Download catch to build the C++ tests
* `-DOWNLOAD_EIGEN=ON`: Download Eigen for the NumPy tests
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
        you automatic mulithreading on all your CMake projects!
* Open the `CMakeLists.txt` with QtCreator to generate for that IDE.
* You can use `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` to generate the `.json` file
  that some tools expect.

</p></details>


To run the tests, you can "build" the check target:

```bash
cmake --build build --target check
```

`--target` can be spelled `-t` in CMake 3.15+. You can also run individual
tests with these targets:

* `pytest`: Python tests only
* `cpptest`: C++ tests only
* `test_cmake_build`: Install / subdirectory tests

If you want to build just a subset of tests, use
`-DPYBIND11_TEST_OVERRIDE="test_callbacks.cpp;test_pickling.cpp"`. If this is
empty, all tests will be built.

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

### Build recipes

This builds with the Intel compiler (assuming it is in your path, along with a
recent CMake and Python 3):

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


[pre-commit]: https://pre-commit.com
[pybind11.readthedocs.org]: http://pybind11.readthedocs.org/en/latest
[issue tracker]: https://github.com/pybind/pybind11/issues
[gitter]: https://gitter.im/pybind/Lobby
[using pull requests]: https://help.github.com/articles/using-pull-requests
