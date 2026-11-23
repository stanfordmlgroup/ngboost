# Instructions for coding agents

This file tells AI coding agents how to handle pull requests (PRs) in this repository. People who contribute can read CONTRIBUTING.md only.

Read CONTRIBUTING.md before you review, write or merge a PR. It defines a breaking change and gives the rules for merges and releases. This file tells you how to apply those rules.

## Actions on GitHub

Merge, close, label, approve or comment on a PR only when the user asks for that action in this session. Push only to branches that you made, unless the user tells you to push somewhere else. When you finish, tell the user each action that you did on GitHub.

## Review a PR

Give a result for each of these items in your review.

1. **Release note.** The PR adds one line to `RELEASE_NOTES.md` under `## Unreleased`, in the section that fits. A line under an older `## Version` heading is in the wrong place, even if git merges it without a conflict.
2. **Breaking or not.** Use the definition in CONTRIBUTING.md. If the PR is breaking and does not have the `breaking` label, say so. If you are not sure, treat it as breaking and give your reason.
3. **Tests.** A bug fix adds a test that fails on master and passes with the fix. CI passes.
4. **Conflicts.** Say if the PR conflicts with master, and list the files.

## Merge a PR

1. Merge only when CI passes.
2. Do not merge a PR with the `breaking` label unless the user tells you that a release was just made. Then merge all the `breaking` PRs together.
3. Before you merge, find the PR's release-note line. If it is not under `## Unreleased`, move it there first: push the fix to the PR branch, or ask the author to do it.
4. If the only conflict is in `RELEASE_NOTES.md`, resolve it and keep both lines. If other files conflict, stop and tell the user. Usually the author fixes these conflicts.
5. When several PRs are ready, merge the small ones first. Merge a stacked PR only after the PR that it is built on.

## Write a PR

- Fill in the template in `.github/pull_request_template.md`.
- Add your release-note line under `## Unreleased`.
- Do not change `version` in `pyproject.toml`. Only the release PR changes it.

## Prepare a release

Do this only when the user asks. A maintainer (usually ryan-wolbeck) publishes each release.

1. Choose the version number with the rule in CONTRIBUTING.md.
2. Change `## Unreleased` to `## Version X.Y.Z`. Above it, add a new `## Unreleased` heading with the same HTML comment.
3. Set `version` in `pyproject.toml` to `X.Y.Zdev`. The maintainer removes `dev` before publishing.

Do not make a tag, a GitHub release or a PyPI upload unless the user asks.

## Commands

- `make install` installs the package and the development tools with poetry.
- `make lint` runs the pre-commit hooks: black, isort, flake8 and pylint.
- `make test` runs all the tests, including the slow ones.
