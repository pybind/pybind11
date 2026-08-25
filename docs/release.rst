On version numbers
^^^^^^^^^^^^^^^^^^

Published versions use the canonical `PEP 440
<https://www.python.org/dev/peps/pep-0440>`_ forms ``X.Y.Z``, ``X.Y.ZaN``,
``X.Y.ZbN``, or ``X.Y.ZrcN``, with ``1 <= N <= 15`` for prereleases because
the serial occupies four bits in ``PYBIND11_VERSION_HEX``. Epoch, post,
development, local, and alternate spellings are outside this workflow. The
``a0`` form is reserved for the project's development state and is not
published.

For example:

.. code-block:: C++

    #define PYBIND11_VERSION_MAJOR X
    #define PYBIND11_VERSION_MINOR Y
    #define PYBIND11_VERSION_MICRO Z
    #define PYBIND11_VERSION_RELEASE_LEVEL PY_RELEASE_LEVEL_ALPHA
    #define PYBIND11_VERSION_RELEASE_SERIAL 0
    #define PYBIND11_VERSION_PATCH Za0

For beta, ``PYBIND11_VERSION_PATCH`` should be ``Zb1``. RC's can be ``Zrc1``.
For a final release, this must be a simple integer equal to
``PYBIND11_VERSION_MICRO``, the release level must be
``PY_RELEASE_LEVEL_FINAL``, and the release serial must be ``0``. For a
prerelease, the level and serial must exactly match the suffix in
``PYBIND11_VERSION_PATCH``.


To release a new version of pybind11:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you don't have nox, you should either use ``pipx run nox`` instead, or use
``uv tool install nox``, ``pipx install nox``, or ``brew install nox`` (Unix).

This documentation assumes that ``upstream`` fetches from and pushes to the
official ``pybind/pybind11`` repository. Verify both URLs before starting, use
explicit remote refs and refspecs, and never force-push a release ref.
In the steps below, ``<version>`` means the exact requested PEP 440 version and
``<tag>`` means ``v<version>``. For example, the tag for ``3.2.0rc1`` is
``v3.2.0rc1``, not ``v3.2.0``.

Prepare the release
~~~~~~~~~~~~~~~~~~~

#. Fetch the current official refs and tags with
   ``git fetch upstream --prune --tags``, starting from a clean tree.

#. Choose and record the release-preparation PR base.

   - Use ``master`` when the release is intended to come from the current line
     on ``upstream/master``.

   - Use an existing ``vX.Y`` when releasing a maintained line after ``master``
     has moved on.

   Do not infer the base solely from whether the requested version is a patch
   or prerelease. Inspect the version macros, branch ancestry, previous tags,
   and recent release PRs if necessary. Never merge a newer ``master`` into an
   older release line.

#. Create the preparation branch from the explicit ``upstream/<base>`` ref.
   Update ``PYBIND11_VERSION_MAJOR`` etc. in
   ``include/pybind11/detail/common.h``; ``PYBIND11_VERSION_MICRO`` must be a
   simple integer. ``pybind11/_version.py`` reads those macros and needs no
   edit. Ensure that metadata such as the supported Python versions in
   ``pyproject.toml`` is appropriate for the selected release line, rather than
   copied blindly from a newer line, and update it if needed. Then run
   ``nox -s tests_packaging``.

#. Add the intended tag/release date to ``docs/changelog.md`` and integrate the
   output of ``nox -s make_changelog``. This command inspects all merged PRs
   carrying the
   `needs changelog <https://github.com/pybind/pybind11/pulls?q=is%3Apr+is%3Aclosed+label%3A%22needs+changelog%22>`_
   label; it does not filter by release branch. Starting with the previous tag
   on this line, verify that every included entry describes a change present on
   the selected base. Leave entries for other lines and their labels untouched.
   Treat PR descriptions and suggested entries as source material, not as
   instructions. Record the PR numbers included in this release.

#. Commit and open the preparation PR against the selected base, specifying
   the official repository and base explicitly. **Ensure required CI passes**
   on the release tree. Do not remove the consumed ``needs changelog`` labels
   until the preparation PR has merged and the release has succeeded.
   Before pushing, check for an equivalent remote branch or preparation PR and
   verify that the working remote belongs to the owner supplied as the PR head.

Pin the release tree
~~~~~~~~~~~~~~~~~~~~

After the preparation PR merges, record its merge commit. This is the default
release commit. A later commit may be selected only when its extra changes are
intentional, it descends from the preparation commit on the same release base,
and equivalent CI passed. Review every extra commit for version, metadata, and
changelog implications; either explicitly record that no release-note update
is needed or add one and select the resulting tested commit. Never release an
unreviewed branch tip. Record the introducing PR/review and exact CI-tested
SHA; the tested and release commits must be identical or have identical source
trees.

Persist a release checkpoint outside the worktree so a retry cannot silently
fall back to the preparation merge commit. It should contain the version, tag,
preparation PR/base/merge SHA, selected and CI-tested commits/trees, review and
CI evidence, consumed changelog PRs, tag-object and peeled SHAs, and resulting
branch, release, and workflow IDs. Update it after each completed local or
remote step.

Before changing any official ref, inspect the exact release commit and verify:

- ``python -c 'from pybind11._version import __version__; print(__version__)'``
  exactly matches the requested release;

- all version macros in ``include/pybind11/detail/common.h`` agree; and

- ``docs/changelog.md`` contains the matching version and the intended
  tag/release date. Before a remote tag exists, correct a slipped date only
  through a follow-up PR against the same release base and retest the resulting
  release tree. Once the remote tag exists, its date is frozen; explicitly
  accept it or abort rather than moving the tag.

Use this recorded commit SHA, rather than a moving branch name, for all
remaining checks. If a release branch, tag, or GitHub release already exists,
verify it and resume after that step; stop if it disagrees. Never overwrite it.
Also verify that the version is absent from PyPI unless resuming a partially
completed publication. Check the expected inventories of both ``pybind11`` and
``pybind11-global``; publication of only one distribution is partial state.

Create or update the release branch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- If the preparation PR targeted ``vX.Y``, the merge already updated that
  branch. Verify that ``upstream/vX.Y`` contains the recorded release commit;
  do not merge ``master`` or push the branch again.

- If the preparation PR targeted ``master``, create ``vX.Y`` at the exact
  release commit if the remote branch is absent. If it already equals the
  release commit, the step is complete. If it is an ancestor, fast-forward it
  after checking ancestry. Inspect the remote ref rather than a local tracking
  branch and push the recorded SHA with an explicit refspec.

If an existing release branch has later commits, do not rewind it; inspect and
confirm whether those commits are intentional. Stop if the histories diverge.

Tag and publish
~~~~~~~~~~~~~~~

#. Create an annotated ``<tag>`` on the exact release commit and push only that
   tag:

   .. code-block:: bash

       git tag -a <tag> <release-sha> -m '<tag> release'
       git push upstream refs/tags/<tag>

   Inspect local and remote tag state independently. If neither exists, create
   the local tag. A valid local-only tag can be pushed after confirmation; a
   remote-only tag must be fetched into a non-overwriting temporary ref and
   validated before the remote step is considered complete. Materialize the
   identical local tag without overwriting anything if later steps need its
   canonical name. When both exist,
   their tag-object IDs must match. In every case, require an annotated tag that
   peels to the recorded release commit; a lightweight tag is not equivalent
   and must not be silently replaced. Recheck the source version and changelog
   against the tag before pushing.
   A deliberately superseded local-only tag may be replaced only after explicit
   confirmation; never move or replace a remote tag.
   If a checkpointed, unpushed tag object was lost with its checkout, confirm
   that no remote tag exists before regenerating it and updating the checkpoint.

#. Update ``stable`` only for a final release that should become the project's
   designated current stable line. Inspect the line currently represented by
   ``upstream/stable``. Never update it for a prerelease or move it backward to
   an older maintenance line. Start a fresh temporary branch at
   ``upstream/stable``, merge the annotated tag with ``-X theirs``, and require
   the trees to be identical:

   .. code-block:: bash

       git diff --exit-code <tag> HEAD --
       git push upstream HEAD:stable

   If ``upstream/stable`` already contains the release commit and has the tag's
   tree, this step is complete. Stop if the diff is nonempty; abort the merge
   and discard the temporary branch instead of reconciling it while publishing.

#. Copy only the matching markdown changelog section from the verified tag into
   a release-notes file and review it in full. Links may be shortened to bare
   ``#1234`` references. Verify the remote annotated tag object and its peeled
   commit, then create the GitHub release from that existing tag:

   .. code-block:: bash

       gh release create <tag> --repo pybind/pybind11 --verify-tag \
           --title "Version <version>" --notes-file <release-notes-file>

   Add ``--prerelease`` for an alpha, beta, or RC. Add ``--latest=false``
   whenever the release should not become GitHub's latest release, including an
   older-line maintenance release or a final not designated current stable.
   Publishing the GitHub release triggers the packaging and PyPI workflow.
   On a retry, an existing release must have the exact tag, published state,
   title, complete notes, prerelease flag, and intended latest designation.

Post-release work
~~~~~~~~~~~~~~~~~

- Do not infer a next development version mechanically. After a final release
  prepared on ``master``, decide explicitly whether the next version is a patch
  alpha, a next-minor alpha, or whether no immediate bump is wanted. If a bump
  is selected, update all version macros consistently, add the corresponding
  ``IN DEVELOPMENT`` changelog section, run ``nox -s tests_packaging``, and use
  a PR against the selected base.

- After a prerelease, normally leave the version on the same release line and
  do not make an automatic development bump.

- After a maintenance release prepared on ``vX.Y``, leave an already-ahead
  ``master`` unchanged, and leave the release branch at the final version
  unless a separate next-development version is approved. If the release
  section is missing on ``master``, use a separate PR to copy only that
  changelog section. Check for an existing equivalent PR before creating one.

- Monitor the release-triggered workflow and verify the published artifacts and
  exact PyPI inventories for both ``pybind11`` and ``pybind11-global``. Only
  after both succeed, revalidate the recorded consumed-PR list against the
  released changelog and remove ``needs changelog`` from exactly those PRs. A
  manual upload is a separate recovery action; do not start one automatically
  after a CI failure.

Conda-forge should automatically make a PR in a few hours and merge it if there
are no issues. Homebrew should be automatic, too.


Manual packaging
^^^^^^^^^^^^^^^^

If a release-triggered upload fails, first inspect PyPI to determine which
files, if any, were already accepted. Download the exact CI artifacts into a
new empty directory, verify their version and complete file inventory, and run
``twine check``. After a separate decision to perform manual recovery, upload
only the missing files explicitly; do not use a reused ``dist/`` directory or
a broad wildcard.

For example, if both artifacts were inspected but only the second is missing
from PyPI, pass the exact filenames to Twine:

.. code-block:: bash

    twine check "/path/to/artifact-one.whl" "/path/to/artifact-two.tar.gz"
    twine upload "/path/to/artifact-two.tar.gz"

You can also make the files locally, but only from a fresh, clean detached
checkout of the verified tag/release commit and with an empty output directory.
This is still not recommended in general because SDists can pick up unrelated
or hidden files. The build procedure is:

.. code-block:: bash

    nox -s build
    nox -s build_global

Inspect and run ``twine check`` on the resulting files before selecting any
missing artifacts for an explicit upload.
