# AMD WEKA-FS Proof of Concept

## Stephen - Notes from Weka Call (March 18 2026)

Stephen to add more compute containers to the cluster. Use the same steps as
per the drive based container 

```
$ creation --cores 2 --flag 
$ weka local setup container --name default6 --net $NIC --cores 2 \
  --cores-ids 12,13 --drives-dedicated-cores 1 \
  --compute-dedicated-cores 1 --no-frontends --base-port 17000 \
  --memory 16GB --failure-domain fd7
```
Only the first container needs a frontend to access the filesystem. Add to the
existing containers. Add a frontend to the first container. Be sure to update
the --core-ids.

weka cluster process
CPU is the core id.

weka status
weka cloud enable (this will send heuristics to Weka)




## Introduction

## Useful Commands

To get a list of all the nodes in the pool. Run from the login node:
```
sinfo -N -l -p amd-arad
```
To start an interactive session on one of the machines identified via the
`sinfo` command above:
```
srun -w <node-name> -A amd-arad -p amd-arad -t 01:00:00 --pty bash
```

To run a job inside a container:
```
srun --container-id=docker://ubuntu:noble -A amd-arad \
  -p amd-arad -t 01:00:00 grep PRETTY_NAME /etc/os-release
```

## SSH container launcher

A Python script runs a Docker container with SSH enabled, a created user, and
optional bind-mounted home. Tested with Debian/Ubuntu-based images.

### Setup (venv)

Create and activate a virtual environment, then install dependencies:

```
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Ensure the Docker daemon is running (and reachable via `DOCKER_HOST` if
remote).

### Usage

Run from the repo root (so the venv and requirements apply):

```
python docker/docker-ssh-setup.py --image IMAGE --ssh-port PORT [OPTIONS]
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

```
python docker/docker-ssh-setup.py --image ubuntu:22.04 --ssh-port 2222
```

Then connect with: `ssh -p 2222 $USER@localhost` (use the host's IP when
connecting remotely).

### Pre-built image with OpenSSH

A minimal image based on Ubuntu 24.04 with OpenSSH server pre-installed is
provided so container startup skips `apt-get install` and is faster. Build and
tag it from the repo root:

```
docker build -t amd-weka-fs/ssh-base:24.04 docker/
```

Then run the script with that image:

```
python docker/docker-ssh-setup.py --image amd-weka-fs/ssh-base:24.04 --ssh-port 2222
```
