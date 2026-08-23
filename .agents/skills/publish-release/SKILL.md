---
name: publish-release
description: Publish a pybind11 release after the prepare-release PR merged — release branch, tag, stable, GitHub release, and post-release bump. Follows docs/release.rst.
---

# pybind11 release publication

Source of truth: `docs/release.rst`. If this skill and that file disagree,
follow `docs/release.rst` and update this skill.

Require both the exact version and the merged release-preparation PR URL or
number. Define the tag as `v` followed by that exact version (for example,
`v3.2.0rc1`, never `v3.2.0` for that RC). Do not infer either input from the
current checkout. On a retry, also require any previously recorded release
checkpoint, especially a nondefault release commit.

**Pause and get explicit confirmation before every push, label mutation, and
the GitHub release.** Show the exact repository, refs, commit SHA, and release
notes involved. Everything else can proceed autonomously.

## 1. Preflight

- Require a clean tree. Verify that both fetch and push URLs for `upstream`
  identify the official `pybind/pybind11` repository, run
  `git fetch upstream --prune --tags`, and check `gh auth status` and the
  account's repository release permissions.
- Inspect the preparation PR in the official repository. Require that it is
  merged, identify its base (`master` or `vX.Y`), record its merge commit, and
  require `gh pr checks <PR> --repo pybind/pybind11` to show the complete
  expected release matrix finished successfully. Investigate skipped or
  cancelled coverage rather than checking only the required subset. Record the
  tested SHA; require it to be the release SHA or prove that their source trees
  are identical.
- The release commit defaults to that merge commit. A later commit may be used
  only if the user explicitly identifies and approves it, it is descended from
  the preparation commit on the same release base, its extra changes are
  intentional, and equivalent CI passed. Review every extra commit for version,
  metadata, and changelog implications. Either record an explicit decision that
  no release-note change is needed or add the needed changelog entry, update the
  consumed-PR list, select the new commit, and run the complete matrix on that
  exact commit/tree. Record the introducing PR/review and CI evidence. Never
  release an unreviewed branch tip.
- Inspect the exact release commit, preferably in a detached checkout. Before
  any remote mutation, require all of the following:
  - `python -c 'from pybind11._version import __version__; print(__version__)'`
    exactly equals the requested version and is not a development version.
  - `include/pybind11/detail/common.h` has internally consistent version
    macros, including release level and serial.
  - `docs/changelog.md` at the release commit has the matching version and the
    intended tag/release date. Before a remote tag exists, a slipped date must
    either be accepted explicitly or corrected through a follow-up PR to the
    same release base, with the resulting merge selected and tested as the new
    release commit. Once the remote tag exists, its changelog date is frozen;
    accept it explicitly or abort publication, but never move the tag.
  - The release commit is contained in the preparation PR's base ref.
- Verify the exact expected file inventories for both `pybind11` and
  `pybind11-global` are absent from PyPI unless this is an intentional resume
  of a partially completed publication. Treat publication of only one
  distribution as partial state, not success.
- Record a release checkpoint containing the exact version, tag, preparation
  PR/base/merge commit, selected and CI-tested commits/trees, review/CI
  evidence, consumed changelog PRs, annotated tag-object and peeled SHAs, and
  resulting branch, GitHub release, and workflow IDs. Persist it at a
  user-approved location outside the worktree, show and update it after every
  completed local or remote step, and reuse it on every retry. Verify existing
  state as described below; stop on any mismatch and never force or overwrite
  remote state.

## 2. Release branch

- If the preparation PR targeted `vX.Y`, verify that `upstream/vX.Y` contains
  the release commit. The merge already updated the branch; do not merge
  `master` or push it again.
- If the preparation PR targeted `master`, create `vX.Y` at the exact release
  commit if the remote ref is absent. If `upstream/vX.Y` equals the release
  commit, record this step as complete. If it is an ancestor, fast-forward it
  by pushing the recorded SHA directly to `refs/heads/vX.Y` after confirmation.
  Inspect the remote ref, not a local tracking branch.
- If an existing `vX.Y` contains later commits, do not rewind it. Stop and ask
  whether those commits are intentional before proceeding; stop on divergence.

## 3. Tag

- Inspect the exact tag independently in the local and `upstream` namespaces.
  In every existing state, require an annotated tag object that peels to the
  recorded release commit; a lightweight tag is not equivalent and must not be
  silently replaced.
  - Neither exists: create the local annotated tag on the explicit commit with
    `git tag -a <tag> <release-sha> -m '<tag> release'`.
  - Local only: validate it, then treat its push as the pending step.
  - Remote only: fetch it into a non-overwriting temporary ref, validate it,
    record the remote push as complete, and materialize the identical local tag
    without overwriting anything if later steps need its canonical name.
  - Both: require matching local and remote tag-object IDs as well as matching
    peeled commits.
  - A deliberately superseded local-only tag may be deleted and recreated only
    after showing the mismatch and obtaining explicit confirmation. Never move
    or replace a remote tag.
  - If the checkpoint records an unpushed local tag whose object was lost with
    its checkout, require that the remote tag is still absent and obtain
    explicit confirmation before regenerating the annotated tag and updating
    its checkpointed object ID.
- Re-run the version and changelog consistency checks against the tag, show the
  tag and target SHA, then, if it is not already remote, push only that tag to
  `upstream` after confirmation.

## 4. Update stable when appropriate

- Inspect the line currently represented by `upstream/stable`. Never update it
  for a prerelease, and never move it backward to an older maintenance line.
  A final release on the current or a newer line updates it only when the user
  confirms that the release should become the project's designated stable.
- Work from a fresh temporary branch based on `upstream/stable`, merge the
  annotated tag with `-X theirs`, and enforce tree equality with
  `git diff --exit-code <tag> HEAD --`. If `upstream/stable` already contains
  the release commit and has that tree, record this step as complete. Stop and
  ask if the trees differ; abort any in-progress merge and discard the
  temporary branch rather than reconciling it autonomously.
- Show the resulting commit and push it with an explicit refspec such as
  `git push upstream HEAD:stable` after confirmation. Never force-push.

## 5. GitHub release

- Extract only the matching markdown changelog section from the verified tag
  into a temporary notes file. Links may be reduced to bare `#1234`. Show the
  complete file to the user and verify once more that the remote annotated tag
  object and peeled commit match the recorded values.
- After confirmation, run:
  `gh release create <tag> --repo pybind/pybind11 --verify-tag --title
  "Version <version>" --notes-file <file>`.
  Add `--prerelease` for an alpha, beta, or RC. Add `--latest=false` whenever
  this release should not become GitHub's latest release, including an
  older-line maintenance release or a final release that was not designated
  current stable.
- If the GitHub release already exists, require the exact tag, published (not
  draft) state, title, complete notes, prerelease flag, and intended latest
  designation instead of recreating it. This release triggers the
  packaging/PyPI workflow.

## 6. Post-release bump ("get back to work")

- Do not infer the next development version arithmetically. Propose the exact
  version and target branch, explain the alternatives, and require explicit
  confirmation before editing.
- After a prerelease, normally leave the version on the same release line and
  make no automatic development bump.
- After a maintenance release prepared on `vX.Y`, leave an already-ahead
  `master` unchanged, and leave the maintenance branch at the final version
  unless a separate next-development version is explicitly approved. If
  `master` lacks this release, open a separate PR against `master` that copies
  only the released changelog section.
- After a final release prepared on `master`, the project may choose a next
  patch alpha, a next-minor alpha, or no immediate bump. Once confirmed, create
  a fresh branch from the explicit remote ref, update all version macros and
  the `IN DEVELOPMENT` changelog section consistently, and run
  `nox -s tests_packaging`.
- Show the diff, head SHA, repository, and target base. Confirm before pushing
  and opening the post-release PR. Check for an existing equivalent PR first so
  retries do not create duplicates.

## Afterwards

- Monitor the release-triggered packaging workflow and verify the published
  artifacts and exact PyPI inventories for both `pybind11` and
  `pybind11-global`. Report failures and stop. Only after both succeed,
  revalidate the consumed-PR list against the released changelog and remove
  `needs changelog` from exactly those PRs, after confirmation.
- A manual `twine` upload is a separate, high-impact recovery action and
  requires new explicit confirmation; `docs/release.rst` describes the
  artifact-based procedure.
- Conda-forge and Homebrew update automatically; no action is normally needed.
