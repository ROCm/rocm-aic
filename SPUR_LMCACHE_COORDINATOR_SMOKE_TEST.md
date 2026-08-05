# Manually test the LMCache coordinator CPU smoke workflow on SPUR

This procedure tests the coordinator-only workflow introduced on branch
`lmcache-coordinator-smoketest`. It builds the container for the exact Git
commit, starts only the LMCache coordinator in a CPU-only container, and runs
the HTTP API smoke suite.

The procedure uses three locations:

1. `mi355-29`, where the working branch is committed and pushed.
2. The SPUR head node, reached with `ssh spur`, where the image build is
   submitted.
3. A workstation checkout where `ssh spur` works, from which the workflow
   launcher is run.

## Important storage detail

Do not use `/shared_nfs` for new artifacts on this SPUR deployment. It is
mounted read-only even though its directory mode bits appear writable:

```text
/shared_nfs  nfs  ro
```

Use `/home/ivaganev/spur-shared` instead. `/home` is writable NFS, is visible
from the SPUR compute nodes, and has enough free space for the image artifact.
The same storage root must be supplied to both the build and the smoke-test
launcher.

The corrected setting used throughout this guide is:

```bash
AIC_SHARED_NFS=/home/ivaganev/spur-shared
```

## 1. Push the branch from `mi355-29`

```bash
ssh mi355-29
cd /home/ivaganev/rocm-aic

git branch --show-current
git status --short
git push -u origin lmcache-coordinator-smoketest

git rev-parse HEAD
exit
```

Pushing is required because the launcher clones the requested commit from
GitHub. An unpushed SHA cannot be tested through the outer launcher.

## 2. Clone the pushed branch on the SPUR head node

```bash
ssh spur
source /etc/profile.d/spur.sh

mkdir -p "$HOME/Projects"
WORKDIR="$(mktemp -d "$HOME/Projects/rocm-aic-coordinator-smoke.XXXXXX")"

git clone \
  --branch lmcache-coordinator-smoketest \
  --single-branch \
  https://github.com/ROCm/rocm-aic.git \
  "$WORKDIR"

cd "$WORKDIR"

SHA="$(git rev-parse HEAD)"
SHORT="${SHA:0:7}"

printf 'SPUR checkout: %s\n' "$SHA"
```

The printed SHA must match the SHA printed on `mi355-29`.

Always source `/etc/profile.d/spur.sh` after logging into SPUR. Noninteractive
SSH sessions do not automatically set `SPUR_CONTROLLER_ADDR`.

## 3. Select the writable artifact location

Still on the SPUR head node:

```bash
AIC_SHARED_NFS="/home/ivaganev/spur-shared"
TARBALL_DIR="$AIC_SHARED_NFS/rocm-aic/images/aic-ci-${SHORT}"

mkdir -p "$TARBALL_DIR"

printf 'Image:    rocm-aic-ci-%s:latest\n' "$SHORT"
printf 'Artifact: %s\n' "$TARBALL_DIR"
```

Check whether the exact artifact already exists:

```bash
find "$TARBALL_DIR" -maxdepth 1 -type f \
  -name 'rocm-aic-ci-*-*.tar*' -ls
```

If a complete artifact for this SHA and `gfx950` is present, skip the build in
the next section.

## 4. Build the matching image

Still in the SPUR checkout:

```bash
cd "$WORKDIR"

AIC_SPUR_CLUSTER=1 \
AIC_SHARED_NFS="/home/ivaganev/spur-shared" \
AIC_SPUR_CONTROLLER="$SPUR_CONTROLLER_ADDR" \
AIC_IMAGE="rocm-aic-ci-${SHORT}:latest" \
AIC_IMAGE_DIR="$TARBALL_DIR" \
make dist-build-fast
```

This submits a CPU build job to the `amd-spur` partition. It is a full image
build and can take substantial time. The compressed image is commonly around
10--15 GB, while the shared BuildKit cache may consume additional space.

After it completes, verify the artifact:

```bash
find "$TARBALL_DIR" -maxdepth 1 -type f \
  -name 'rocm-aic-ci-*-*.tar*' -ls
```

The expected path and filename are derived from the checked-out commit:

```text
/home/ivaganev/spur-shared/rocm-aic/images/aic-ci-${SHORT}/
rocm-aic-ci-${SHORT}-latest-gfx950.tar.zst
```

Exit the SPUR session after the artifact is ready:

```bash
exit
```

## 5. Prepare a launcher checkout on the workstation

Run this on a machine where `ssh spur` succeeds. If a checkout of this branch
already exists, update it with a fast-forward pull:

```bash
cd /path/to/rocm-aic
git switch lmcache-coordinator-smoketest
git pull --ff-only
```

Otherwise, make a temporary launcher checkout:

```bash
cd /path/where/you/want/the/checkout

git clone \
  --branch lmcache-coordinator-smoketest \
  --single-branch \
  https://github.com/ROCm/rocm-aic.git \
  rocm-aic-coordinator-smoke-driver

cd rocm-aic-coordinator-smoke-driver
```

Confirm that the launcher checkout uses the same SHA as the image build:

```bash
SHA="$(git rev-parse HEAD)"
SHORT="${SHA:0:7}"
printf 'Launcher checkout: %s\n' "$SHA"
```

## 6. Run the workflow launcher manually

First read the controller address from the SPUR profile:

```bash
SPUR_CONTROLLER="$(
  ssh spur \
    'source /etc/profile.d/spur.sh; printf "%s" "$SPUR_CONTROLLER_ADDR"'
)"

test -n "$SPUR_CONTROLLER"
printf 'SPUR controller: %s\n' "$SPUR_CONTROLLER"
```

Then run the same launcher used by the GitHub Actions workflow:

```bash
AIC_SPUR_HOST=spur \
AIC_SHARED_NFS=/home/ivaganev/spur-shared \
AIC_SPUR_CONTROLLER="$SPUR_CONTROLLER" \
GITHUB_RUN_ID="manual-$(date +%s)" \
GITHUB_RUN_ATTEMPT=1 \
bash .github/scripts/spur-lmcache-coordinator-cpu-smoke.sh "$SHA"
```

The launcher will:

1. Clone the exact pushed SHA on the SPUR head node.
2. Submit an eight-CPU, 16 GB, CPU-only Slurm job.
3. Load the matching image tarball on the allocated compute node.
4. Start only the LMCache coordinator on a random localhost port.
5. Verify that the container has no device mappings and cannot see `/dev/kfd`
   or `/dev/dri`.
6. Run the coordinator HTTP API suite, including the relevant PR #4275 tests
   and the HTTP-reachable PR #4292 behavior.
7. Print coordinator logs on failure and remove the container during cleanup.

A successful run ends with output similar to:

```text
Coordinator HTTP smoke suite passed: 14 cases
Coordinator CPU smoke test passed
```

## Troubleshooting

### The build reports `Error 141`

Inspect the actual SPUR job log before treating `141` as the root cause. Status
141 means `SIGPIPE` and can be a secondary status from the submission wrapper's
`squeue | awk | grep -q` polling pipeline under `pipefail`.

For a job ID such as `38844`:

```bash
ssh spur
source /etc/profile.d/spur.sh

scontrol show job 38844
tail -n 200 \
  /home/ivaganev/Projects/rocm-aic-coordinator-smoke.*/logs/38844/build.out
```

Job `38844` actually failed because it tried to create
`/shared_nfs/rocm-aic`, which is inaccessible and on a read-only mount. Its
authoritative build exit marker was `1`, not `141`.

### The launcher cannot find the tarball

Check all three values together:

```bash
printf 'SHA=%s\nSHORT=%s\nTARBALL_DIR=%s\n' \
  "$SHA" "$SHORT" "$TARBALL_DIR"
```

The Git SHA used by the launcher, image name, and artifact directory must match
exactly. Both build and launcher must use:

```text
AIC_SHARED_NFS=/home/ivaganev/spur-shared
```

### The allocated host contains GPU device nodes

A SPUR compute host may physically contain `/dev/kfd` or `/dev/dri` even when
the Slurm job did not request a GPU. The workflow's CPU-only guarantee is at the
container boundary: it passes no GPU device mappings and asserts that those
device nodes are absent inside the coordinator container.

### GitHub Actions manual dispatch is unavailable

Pushing this feature branch is sufficient for the manual launcher because it
can clone the SHA from GitHub. It may not make the workflow dispatchable in the
GitHub Actions UI: a `workflow_dispatch` workflow generally must also exist on
the repository's default branch.
