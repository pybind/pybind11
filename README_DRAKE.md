# RobotLocomotion Fork of `pybind11` for Drake

This fork is developed for use in [Drake](drake.mit.edu/), and provides the
following features on top of the main development:

* NumPy:
  * Incorporates https://github.com/pybind/pybind11/pull/1152 (with fixes):
  Permit creating matrices with `dtype=object`.
* Transferring ownership between C++ and Python using `py::wrapper<Class>`
  * Resolves https://github.com/pybind/pybind11/issues/1132: Casting from Python to C++ `unique_ptr<Class>`.
  * Resolves https://github.com/pybind/pybind11/issues/1145: Transfer of
  ownership of subclasse between C++ and Python with `shared_ptr<Class>`
  * Resolves https://github.com/pybind/pybind11/issues/1138: Helpful error
  message when mixing holder types at runtime.
* Other modifications:
  * Resolves https://github.com/pybind/pybind11/issues/1238: Deregister
  instances by type and pointer.

For usage in Drake, please see the following pages in Drake:

* [User: Python Bindings](https://drake.mit.edu/python_bindings.html#using-the-python-bindings)
* [Dev: Python Bindings](https://drake.mit.edu/doxygen_cxx/group__python__bindings.html)
* [User: API Reference](https://drake.mit.edu/pydrake/index.html)

## Maintenance Philosophy

This repository should be updated to synchronize with upstream at least every 3
months to ensure that we have core bugfixes and features.

When developing features or bugfixes in this fork that are relevant to upstream
features, first try to make a descriptive upstream issue to see if it is
something desirable for upstream, and make a PR for the community to see and
possibly review. Then continue developing here.

Review should happen using Reviewable.

Please avoid superfluous (non-functional) changes to the original `pybind11`
source code (e.g. no whitespace reflowing), and try to stay relatively close to
`pybind11`s style for consistency.

## Continuous Integration

For simplicity, these checks are copied from upstream's CI which uses Travis
CI as part of GitHub's Checks. They test:

* Ubuntu and macOS (Windows tests disabled)
* C++11, C++14, and C++17
* Release and debug builds
* GCC 4.8, 6, and 7
* clang 7
* Apple clang 7.3 and 9
* 64bit and 32bit
* CPython and PyPy (though PyPy is partially supported on this fork)
* Python 2.7, 3.5, 3.6, 3.7, and 3.8

To see builds, see [this fork's Travis CI page](https://travis-ci.com/RobotLocomotion/pybind11/branches).

Windows testing (with AppVeyor) is disabled for this repository.

### Quick Testing

    mkdir build && cd build
    cmake .. \
        -DPYTHON_EXECUTABLE=$(which python3) \
        -DPYBIND11_TEST_OVERRIDE='test_builtin_casters.cpp;test_class.cpp;test_eigen.cpp;test_multiple_inheritance.cpp;test_ownership_transfer.cpp;test_smart_ptr.cpp'
    make -j 4 pytest

## Local Git Setup

For development, please make your own GitHub fork of the upstream repository.

It is suggested to clone this repository with the following remotes:

    # Clone from upstream.
    git clone --origin upstream https://github.com/pybind/pybind11
    cd pybind11
    git remote set-url --push upstream no_push
    # Add robotlocomotion.
    git remote add robotlocomotion https://github.com/RobotLocomotion/pybind11
    git remote set-url --push robotlocomotion no_push
    # Add origin (your fork).
    git remote add origin <url-to-your-fork>
    # Fetch from all remotes.
    git fetch --all
    # Checkout to track robotlocomotion/drake.
    git checkout -b drake robotlocomotion/drake

## Branches

The following branches are used:

* `drake` - This is the active development branch, forked from `upstream/master`
* `no_prune` - This is now a stale branch, meant to keep track of prior
versions that were rebased. This should be kept for historical purposes.

## Submitting PRs

* Submit your PR to `RobotLocomotion/pybind11` (targeting `drake`) and request
review.
  * Ensure that the PR passes the Travis CI checks!
* Submit a PR to Drake using the latest commit from your `pybind11` PR. Ensure
experimental CI passes, and be sure to include macOS. An example of requesting
macOS testing in Drake:

        @drake-jenkins-bot mac-mojave-clang-bazel-experimental-release please
        @drake-jenkins-bot mac-catalina-clang-bazel-experimental-everything-release please

* Once your PR is reviewed, accepted, and CI passes, merge your
`RobotLocomotion/pybind11` PR, then update your Drake PR to use the latest
merge commit for the fork:

        cd pybind11
        git fetch robotlocomotion
        git rev-parse robotlocomotion/drake

* Merge the Drake PR once it passes CI and review is finished.

## Pulling Upstream Changes

This repository should be merged with upstream (the official repository) about
every 3 months. We use a merge strategy (not rebase) with Git so that updates
can be a simple fast-forward merge.

To update the repository, first checkout the branch and merge:

    git fetch robotlocomotion && git fetch upstream
    git checkout -b <new-branch-name> robotlocomotion/drake
    # Record the soon-to-be-old merge-base.
    git merge-base upstream/master robotlocomotion/drake
    # Merge.
    git merge upstream/master  # Resolve conflicts and commit
    git push --set-upstream origin <new-branch-name>

Then create a `RobotLocomotion/pybind11` PR with your branch. Title the PR as
`Merge 'upstream/master' (<sha_new>) from previous merge-base (<sha_old>)`,
where `<sha_new>` is the short-form SHA of the current `upstream/master`, and
`<sha_old>` is the short-form SHA of prior merge-base you recorded.

Afterward, follow the normal PR process as outlined above.
