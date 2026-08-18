---
name: publish-release
description: Publish a pybind11 release after the prepare-release PR merged — release branch, tag, stable, GitHub release, and post-release bump. Follows docs/release.rst.
---

# pybind11 release publication

Source of truth: `docs/release.rst`. If this skill and that file disagree,
follow `docs/release.rst` and update this skill. One difference: the
post-release bump goes through a PR rather than a direct push to master.

**Pause and get explicit confirmation before every push, tag push, and the
GitHub release.** Everything else can proceed autonomously.

## 1. Preflight

- The release-preparation PR (version bump + changelog) must be merged.
- `git checkout master && git fetch upstream` and be up to date, clean tree.
- Get the version from `git grep PYBIND11_VERSION include/pybind11/detail/common.h`
  Stop if this is a development version and not a release version.
- Check `gh auth status` works.

## 2. Release branch

- New MINOR version: `git checkout -b vX.Y && git push -u upstream vX.Y`.
- Patch release: `git checkout vX.Y && git merge master && git push`.

## 3. Tag

- `git tag -a vX.Y.Z -m 'vX.Y.Z release'`
- Last-minute consistency check before pushing:
  `git grep PYBIND11_VERSION include/pybind11/detail/common.h` — must match
  the tag.
- `git push upstream vX.Y.Z` (confirm first).

## 4. Update stable (final releases only, not pre-releases)

- `git checkout stable && git merge -X theirs vX.Y.Z`
- `git diff vX.Y.Z` — review; the diff must be empty. Reconcile if not.
- `git push` (confirm first).

## 5. GitHub release

- `gh release create vX.Y.Z -t "Version X.Y.Z"` with the markdown changelog
  section as the notes (links may be reduced to bare `#1234`); add `-p` for a
  pre-release. This triggers the PyPI upload from CI.
- Confirm with the user before running; show them the release notes first.

## 6. Post-release bump ("get back to work")

- `git checkout master`, then a working branch, e.g. `chore/back-to-work`.
- In `common.h`: increment MINOR (only after a final MINOR release), set
  MICRO to the next value, `PYBIND11_VERSION_PATCH` to `<micro>a0` (e.g.
  `0a0`), level to `PY_RELEASE_LEVEL_ALPHA`, serial to `0`. If a version
  branch was updated instead, use PATCH `1a0`.
- `nox -s tests_packaging` to verify.
- New MINOR only: add a new `IN DEVELOPMENT` section at the top of
  `docs/changelog.md`.
- Commit, push, and open a PR (confirm first).

## Afterwards

Conda-forge and Homebrew update automatically; no action needed. If the CI
upload fails, `docs/release.rst` describes manual `twine` upload from the
job artifacts.
