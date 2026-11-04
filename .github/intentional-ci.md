# Intentional CI development pilot

The `ci:intentional` label opts a PR out of automatic GitHub Actions test jobs,
independently of draft or ready-for-review state. Other PRs retain their current
behavior. Scheduled, branch-push, release, and existing manual runs retain their
current conditions. Skipped workflow entries can still appear in GitHub.

Apply the label before pushing updates. Adding it does not cancel existing runs;
automatic labeling after PR creation can be too late to suppress the first run.
Existing stack branches may need rebasing onto the bootstrap before their
PR workflows contain the opt-out guards. AppVeyor and Read the Docs are separate
services and do not honor this GitHub Actions opt-out.

## Experimental commands

After this bootstrap merges, create `intentional-ci-experiment` in
`pybind/pybind11` from `master`. Implement the worker there, retaining
`.github/workflows/intentional-ci.yml` and its `workflow_dispatch` string input
named `request`. The bootstrap worker only reports that no tests were executed;
it does not implement either test profile or publish a CI result for the PR.

Any PR author is welcome to join the pilot. On their own open PR labeled
`ci:intentional`, they can post:

* `/ci full`
* `/ci quick`
* `/ci help` (usage appears in the listener's Actions job summary)

The listener runs trusted code from `master`, fetches the current PR state, and
sends the PR number, profile, head SHA, base SHA, and requesting comment ID to
the fixed development branch. Each command requests one captured revision.
Draft status and commit messages do not select the profile. A missing worker
branch produces an actionable failure in the listener; no PR tests run.

Keep experimental workflow/check names, concurrency groups, and cache keys
separate from ordinary CI. Test the captured revisions with read-only
permissions and no secrets; keep result publishing in a separate trusted job.
The listener itself never checks out PR code. It accepts requests only from
the PR author; the label is the explicit opt-in to the evolving experiment.

Implementation changes can stay on the development branch while the listener
and ordinary workflows on `master` remain stable. Submit the completed design
and any proposed project-wide rollout together for review.
