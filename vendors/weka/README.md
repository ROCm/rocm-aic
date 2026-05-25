<!--
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# AMD WEKA-FS proof of concept

Scripts and a minimal SSH-enabled container image for exploring WEKA-backed
storage on the AMD Arad Slurm pool. See [NOTES.md](NOTES.md) for informal
cluster notes from vendor calls.

Part of [rocm-aic](../../README.md).

## Slurm and pool commands

From the login node, list nodes in the pool:

```bash
sinfo -N -l -p amd-arad
```

Interactive session on a chosen node:

```bash
srun -w <node-name> -A amd-arad -p amd-arad -t 01:00:00 --pty bash
```

Run a command inside a container:

```bash
srun --container-id=docker://ubuntu:noble -A amd-arad \
  -p amd-arad -t 01:00:00 grep PRETTY_NAME /etc/os-release
```

## SSH container launcher

`docker/docker-ssh-setup.py` runs a Docker container with SSH enabled, a
created user, and optional bind-mounted home. Tested with Debian/Ubuntu-based
images.

### Setup (venv)

From the **repository root**:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure the Docker daemon is running (and reachable via `DOCKER_HOST` if
remote).

### Usage

From the repository root:

```bash
python vendors/weka/docker/docker-ssh-setup.py \
    --image IMAGE --ssh-port PORT [OPTIONS]
```

Required arguments:

- **--image**: Docker image (e.g. `ubuntu:22.04`).
- **--ssh-port**: Host port to publish container SSH (e.g. `2222`).

Optional arguments:

- **--user**: Username to create in the container (default: current user).
- **--home**: Host path to bind as user's home (default: current `$HOME`).
- **--uid**: UID for the new user (default: current process UID).
- **--gid**: GID for the new user (default: current process GID).
- **--container-name**: Container name (default: Docker assigns a random name).
- **--hostname**: Container hostname (shown in shell prompt; default: container ID).
- **--authorized-keys**: Host path to file to copy as the user's
  `~/.ssh/authorized_keys` (for key-based login without bind-mounted home).
- **--network-host**: Use host network mode so the container sees all host
  network interfaces (e.g. for RDMA). SSH listens on `--ssh-port` on the host.
- **--device**: Pass a host device into the container (e.g. `/dev/infiniband/
  uverbs0`). Can be repeated.
- **--expose-nvme**: Pass all host `/dev/nvme*` devices so nvme-cli works
  on NVMe SSDs inside the container.

Example:

```bash
python vendors/weka/docker/docker-ssh-setup.py \
    --image ubuntu:22.04 --ssh-port 2222
```

Then connect with: `ssh -p 2222 $USER@localhost` (use the host's IP when
connecting remotely).

### Pre-built image with OpenSSH

A minimal Ubuntu 24.04 image with OpenSSH server pre-installed avoids
`apt-get install` on each start. Build from `vendors/weka/docker/`:

```bash
docker build -t amd-weka-fs/ssh-base:24.04 vendors/weka/docker/
```

Then:

```bash
python vendors/weka/docker/docker-ssh-setup.py \
    --image amd-weka-fs/ssh-base:24.04 --ssh-port 2222
```
