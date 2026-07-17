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

## Pull Requests

We welcome pull requests from outside contributors. Pull requests must pass
our CI and be approved by at least one code owner. Outside contributors
should fully fill out the PR template for non-trivial PRs.


## Issue Reporting

* Issues should be reported as GitHub issues
* Feature requests should be made using GitHub discussions