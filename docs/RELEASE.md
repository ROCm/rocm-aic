# Stable Release Process

Stable releases are published from immutable git tags and are handled by
`.github/workflows/aic-release.yml`.

## What the release workflow publishes

Each tagged release publishes:

- versioned GitHub Release notes
- `vllm`, `lmcache`, and `nixl_rocm` ROCm wheels from the Dockerfile `wheels`
  stage
- a source tarball from `make export`
- a `SHA256SUMS` manifest for the attached assets
- a Docker image pushed as `rocm-aic:vX.Y.Z` and `rocm-aic:latest`

## Prerequisites

- the release commit is already merged to `main`
- required CI is green on the release commit
- `VERSION` contains the release version without the leading `v`
- repository secrets are configured:
  - `DOCKERHUB_USERNAME`
  - `DOCKERHUB_TOKEN`
  - `CORP_CA_CERT` when the build must trust the AMD corporate proxy CA

## Release steps

1. Update `VERSION` to the new release number and land any related documentation
   updates in `main`.
2. Confirm the release commit is the one you want to publish.
3. Create an annotated tag whose name exactly matches `v$(<VERSION)`.
4. Push the tag to GitHub.
5. Wait for the `AIC Release` workflow to finish.

The workflow rejects tags that do not match `VERSION`, so the release tag and
the version file stay in sync.

## Post-release checks

After the workflow completes, verify:

- the GitHub Release exists for the tag and contains the wheels, tarball, and
  `SHA256SUMS`
- the release notes mention the correct source SHA and component refs
- `docker pull rocm-aic:vX.Y.Z` succeeds
- `docker pull rocm-aic:latest` resolves to the same release build

## Reruns and recovery

The workflow is safe to rerun for the same tag. If the GitHub Release already
exists, the workflow refreshes the notes and replaces assets in place.
