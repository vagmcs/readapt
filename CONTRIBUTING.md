# Overview

Thanks for taking the time to contribute! We appreciate all contributions, from reporting bugs to implementing new features. If you're unclear on how to proceed after reading this guide, please contact the author.

# Reporting bugs or suggesting enhancements

We use GitHub issues to track both bugs and suggested enhancements. You can report a bug or suggest an enhancement by opening a new issue.

Before creating an issue, please:

- Check that your bug or enhancement has not already been reported.
- Ensure your bug exists on the latest version of readapt (for bug reports).
- If you find a closed issue that seems relevant, open a new issue and include a link to the original in your description.

For bug reports, include as many details as possible to help maintainers resolve the issue faster.

For enhancement suggestions, describe the desired behavior and rationale, and provide examples of how the feature should be used. Clear, detailed reports and suggestions help improve the project more efficiently.

# Contributing to the codebase

## Picking an issue

Pick an issue by going through the issue tracker and finding an issue you would like to work on. Feel free to pick any issue that is not already assigned. If you decide to take on an issue, please comment on the issue to let others know. You may use the issue to discuss possible solutions.

## Configuring Git

For contributing to readapt you need a free GitHub account and have git installed on your machine. Start by forking the readapt repository, then clone your forked repository using git:

```bash
git clone https://github.com/<username>/readapt.git
cd readapt
```

Optionally you may also set the upstream remote to be able to sync your fork with the readapt repository in the future:

```bash
git remote add upstream https://github.com/vagmcs/readapt.git
git fetch upstream
```

## Updating the development environment

Dependencies are updated regularly - at least once per month. If you do not keep your environment up-to-date, you may notice tests or CI checks failing, or you may not be able to build readapt at all.

To update your environment, first make sure your fork is in sync with the readapt repository:

```bash
git checkout main
git fetch upstream
git rebase upstream/main
git push origin main
```

If the Rust toolchain version has been updated, you should update your Rust toolchain. Follow it up by running cargo clean to make sure your Cargo folder does not grow too large:

```bash
rustup update
cargo clean
```

## Pull requests

When you have resolved your issue, open a pull request in the readapt repository. Please adhere to the following guidelines:

- **Title**:

  - Start your pull request title with a [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) tag. This ensures that your contribution appears to the right section of the changelog.
  - Use a descriptive title starting with an uppercase letter. This text ends up in the changelog, so make sure the text is meaningful to the user. Use single backticks to annotate code snippets. Use active language and do not end your title with punctuation.
  - Example: fix: Fix `MDP` trait not handling terminal states correctly

- **Description**: Add any relevant information that you think may help the maintainers review your code.

- Make sure your branch is rebased against the main branch.
- Make sure all GitHub Actions checks pass.

After you have opened your pull request, a maintainer will review it and possibly leave some comments. Once all issues are resolved, the maintainer will merge your pull request, and your work will be part of the next release!

> Keep in mind that your work does not have to be perfect right away! If you are stuck or unsure about your solution, feel free to open a draft pull request and ask for help.

# Contributing to documentation

Readapt uses `cargo doc` to build its documentation. Contributions to improve or clarify the API reference are welcome.

# Release flow

This section is intended for readapt maintainers. Readapt releases Rust crates to crates.io. New releases are marked by an official GitHub release and an associated git tag. We use [release drafter](https://github.com/release-drafter/release-drafter) to automatically draft GitHub release notes.

# License

Any contributions you make to this project fall under the MIT License that covers the readapt project.