========
pybind11
========

This repository is a fork of https://github.com/pybind/pybind11 and
it also contains waf build scripts that are necessary for integration with
other Steinwurf libraries.

.. contents:: Table of Contents:
   :local:

Quick Start
-----------

If you already installed a C++14 compiler, git and python on your system,
then you can clone this repository to a suitable folder::

    git clone git@github.com:steinwurf/pybind11.git

Configure and build the project::

    cd pybind11
    python waf configure
    python waf build

Run the unit tests::

    python waf --run_tests


Steinwurf Changes
-----------------

This repo was created as a fork of https://github.com/pybind/pybind11.

Unfortunately, the original repo uses tags like v2.2.4 that are not compatible
with Steinwurf's semver tags.

So as a first step, we remove all remote tags (on Github)::

    git tag -l | xargs -n 1 git push --delete origin

Then we remove all local tags::

    git tag | xargs git tag -d

After this, there should be no tags listed here: https://github.com/steinwurf/pybind11/tags

To get future changes, we add a remote called ``upstream`` to point
to the original repo::

    git remote add upstream git@github.com:pybind/pybind11.git

We should make sure that we don't fetch any future tags from this remote,
so we add the ``--no_tags`` option to upstream (see more details here:
https://stackoverflow.com/a/24917718)::

    git config remote.upstream.tagopt --no-tags

Alternatively, we can always use ``--no-tags`` option when fetching from this
remote (we don't need this if we set the option above)::

    git fetch upstream --no-tags

