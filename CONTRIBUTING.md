# Contributing to rocm-aic

Thank you for contributing! This guide outlines the development workflow,
contribution standards, and best practices when working on rocm-aic.

## Development Setup

Follow instructions listed in [README.md](README.md) and build the
project.

## Branching Model

The main development trunk of rocm-aic is the `main` branch.

rocm-aic generally uses trunk-based development, where feature branches are
intended to be relatively short-lived. When necessary, feature branches will
be created and prefixed with `feature/`. These feature branches will be
deleted after the feature has been merged to `main`. Any feature branches
which are not merged to `main`, but should be kept around for posterity,
will be renamed `feature` --> `inactive`.

External developers must use forks for development. You will sometimes see
branches from AMD staff named `<category>/<user>/<description>`. These will
be very short-lived.

### Release branches

Releases are tagged `vX.Y.Z`, where `X`, `Y`, and `Z` are the major, minor,
and patch versions of the release.

Releases occur from `main`. Upon release, the tag is created and the minor
version number is bumped. The major version number will only be bumped on
`main` when making an API/ABI-breaking change.

Release branches are created retroactively and only when it is necessary to
bugfix supported versions of the library. Bugfixing should take place on
`main` and be cherry-picked to any branches that are being maintained.

### Pre-release packages from `main`

Each push to `main` rebuilds the Debian packages and updates the
GitHub pre-release attached to tag `rocm-aic-latest`. That is the only
rolling package channel maintained by this repository's CI. An older GitHub
release or tag named `latest` is not produced by the current workflows; if
it still appears on the releases page, it is legacy and may be removed to
avoid confusion.

## Pull Requests

We welcome pull requests from outside contributors. Pull requests must pass
our CI and be approved by at least one code owner. Outside contributors
should fully fill out the PR template for non-trivial PRs.


## Issue Reporting

* Issues should be reported as GitHub issues
* Feature requests should be made using GitHub discussions