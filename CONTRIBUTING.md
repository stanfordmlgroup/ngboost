# Contributing to NGBoost

Thank you for your help. This page tells you how we handle pull requests (PRs) and releases. We keep the rules short on purpose.

## When you open a PR

1. Add one line about your change to `RELEASE_NOTES.md`, under `## Unreleased`, in the section that fits. Write it for users: say what changed and what they must do, if anything. End the line with the issue and PR numbers, for example `(issue #123, PR #456)`.
2. In the PR description, say if your change is breaking.
3. Make sure that `make lint` and `make test` pass. CI runs them on every PR.

## Breaking changes

A change is breaking if code that works today stops working, or if it gives different results and the old results were not wrong. Some examples: you remove or rename a function, argument or attribute, or you change a default.

A bug fix that corrects wrong results is not breaking. A deprecation is not breaking if the old name still works and gives a `FutureWarning`. If you are not sure, treat the change as breaking.

## For maintainers

**Sort each PR when it comes in.** If it is breaking, add the `breaking` label.

**Merge PRs without the `breaking` label when they are ready.** Do not hold them for a release or for other PRs. Open PRs conflict with each other, and the number of conflicts increases quickly as more PRs stay open. Merge small PRs first. If a PR conflicts with master, ask the author to update it. If the only conflict is in `RELEASE_NOTES.md`, resolve it in the GitHub web editor and keep both lines.

**Hold PRs with the `breaking` label, then merge them together right after a release.** Users then get all the breaking changes in one release.

**Release when you want to.** In the release PR, change `## Unreleased` to the new version number, and add a new `## Unreleased` section above it. To choose the number:

- If no PR with the `breaking` label was merged since the last release, increase the last number, for example from 0.6.0 to 0.6.1.
- If one was merged, increase the middle number and set the last number to 0, for example from 0.6.1 to 0.7.0.

**Stacked PRs.** If PR B is built on the branch of PR A, merge A first. GitHub then changes the base of B to master.
