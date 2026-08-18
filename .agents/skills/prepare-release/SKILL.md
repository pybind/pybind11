---
name: prepare-release
description: Open the pybind11 release-preparation PR — version bump in common.h and changelog integration. Follows docs/release.rst. After the PR merges, use publish-release.
---

# pybind11 release preparation PR

Source of truth: `docs/release.rst`. If this skill and that file disagree,
follow `docs/release.rst` and update this skill. One difference: this skill
makes a PR rather than pushing directly to master.

The argument is the version to release (PEP 440: `X.Y.Z`, or a pre-release
like `X.Y.Zrc1` / `X.Y.Zb1` / `X.Y.Za1`). If no argument is given, propose
the next version from the current `PYBIND11_VERSION_*` macros and confirm
with the user before you start.

**Pause and get explicit confirmation before pushing or opening the PR.**
Everything else can proceed autonomously.

## 1. Preflight

- Start from an up-to-date `master` with a clean tree; `git fetch upstream`
  (the upstream remote must be `https://github.com/pybind/pybind11.git`).
- Create a working branch, e.g. `chore/prepare-X.Y.Z`.
- Check `gh auth status` works; `nox` (or `uvx nox`) is available.

## 2. Version bump

Edit `include/pybind11/detail/common.h` only — `pybind11/_version.py` parses
it, so it needs no edit:

- `PYBIND11_VERSION_MAJOR` / `MINOR` / `MICRO`: plain integers.
- `PYBIND11_VERSION_RELEASE_LEVEL`: `PY_RELEASE_LEVEL_FINAL` for a final
  release; `PY_RELEASE_LEVEL_ALPHA` / `PY_RELEASE_LEVEL_BETA` /
  `PY_RELEASE_LEVEL_GAMMA` (rc) for pre-releases, with
  `PYBIND11_VERSION_RELEASE_SERIAL` set to the pre-release number.
- `PYBIND11_VERSION_PATCH`: `Z` for final, `Zrc1` / `Zb1` / `Za1` for
  pre-releases (must agree with the level/serial above).

Verify: `nox -s tests_packaging`.

Also confirm `pyproject.toml` metadata is current (e.g. supported Python
versions).

## 3. Changelog

- Run `nox -s make_changelog`. It reads merged PRs labeled
  [needs changelog](https://github.com/pybind/pybind11/pulls?q=is%3Apr+is%3Aclosed+label%3A%22needs+changelog%22).
- Integrate the output into `docs/changelog.md` under the section for this
  version, and add the release date (today) to the section header.
- Do not paste generated or suggested entries verbatim without review. Normalize
  them to match the surrounding changelog style:
  - Use concise, user-facing entries; avoid PR-description detail, rationale,
    implementation history, and long caveats unless needed to understand the
    user-visible change.
  - Use reporting/past-tense style consistently (`Fixed`, `Added`, `Updated`,
    `Improved`, `Removed`, etc.), converting imperative suggestions like
    "Fix ..." or "Add ...". Prefer wording like "was updated to ..." when it
    preserves the meaning better than "now ..."; use "now" only when it is the
    clearest way to avoid ambiguity.
  - Preserve the technical meaning of the PR suggestion. If shortening risks
    changing the meaning, inspect the PR description and commits before
    rewriting.
  - Keep the standard entry shape: bullet text, then the PR link on the next
    indented line. Flatten accidental code fences or deeply nested bullets
    unless they are genuinely needed.
  - Categorize using the nearby release pattern (`New Features`, `Bug fixes`,
    `Internal`, `Documentation`, `Tests`, `CI`, etc.). Put non-breaking
    production-code maintenance that is not user-facing under `Internal`.
- Proofread the resulting section for consistent tense, category placement,
  duplicate/missing PR links, and overly long entries. Inform the user if any
  wording or categorization still needs human review.
- Remind the user to clear the `needs changelog` labels in the GitHub web UI
  (or offer to do it with `gh pr edit <n> --remove-label "needs changelog"`).

## 4. Commit and PR

- `git add -u`, commit (conventional commits, e.g.
  `chore: prepare X.Y.Z release`, with the `Assisted-by:` trailer).
- Push the branch and open a PR against `master` with `gh pr create`
  (confirm first). Keep the description short; no changelog entry is needed
  for the prep PR itself.

## Handing off

When the PR is approved and merged, invoke the `publish-release` skill for
the branch, tag, stable update, GitHub release, and post-release bump.
