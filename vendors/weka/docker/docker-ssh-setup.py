#!/usr/bin/env python3
"""
Run a Docker container with SSH enabled, a created user, and optional
bind-mounted home. Uses argparse and the Docker SDK.
"""
import argparse
import getpass
import os
import sys
import tarfile
from io import BytesIO
from pathlib import Path

import docker


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a container with SSH, created user, and optional "
        "bind-mounted home.",
    )
    p.add_argument("--image", required=True, help="Docker image (e.g. ubuntu:22.04)")
    p.add_argument(
        "--user",
        default=None,
        help="Username to create in the container (SSH login); default: "
        "current user ($USER / whoami).",
    )
    p.add_argument(
        "--ssh-port",
        required=True,
        type=int,
        help="Host port to publish container port 22 (e.g. 2222).",
    )
    p.add_argument(
        "--home",
        default=None,
        help="Host path to bind as user's home; default: current $HOME.",
    )
    p.add_argument(
        "--uid",
        type=int,
        default=None,
        help="UID for the new user; default: current process UID.",
    )
    p.add_argument(
        "--gid",
        type=int,
        default=None,
        help="GID for the new user; default: current process GID.",
    )
    p.add_argument(
        "--container-name",
        default=None,
        help="Container name; if omitted, Docker assigns a random name.",
    )
    p.add_argument(
        "--hostname",
        default=None,
        help="Container hostname (shown in prompt as \\h); if omitted, "
        "Docker uses the container ID.",
    )
    p.add_argument(
        "--authorized-keys",
        default=None,
        metavar="FILE",
        help="Host path to file to copy as user's ~/.ssh/authorized_keys.",
    )
    p.add_argument(
        "--network-host",
        action="store_true",
        help="Use host network mode so the container sees all host network "
        "interfaces (e.g. for RDMA). SSH will listen on --ssh-port.",
    )
    p.add_argument(
        "--device",
        action="append",
        default=None,
        metavar="DEVICE",
        dest="devices",
        help="Pass a host device into the container (e.g. /dev/infiniband/"
        "uverbs0). Can be repeated.",
    )
    p.add_argument(
        "--expose-nvme",
        action="store_true",
        help="Pass all host /dev/nvme* devices into the container so "
        "nvme-cli can see NVMe SSDs.",
    )
    return p.parse_args()


def build_run_kwargs(args):
    uid = args.uid if args.uid is not None else os.getuid()
    gid = args.gid if args.gid is not None else os.getgid()
    home = args.home or str(Path.home())

    kwargs = {
        "image": args.image,
        "detach": True,
        "volumes": {home: {"bind": f"/home/{args.user}", "mode": "rw"}},
        "user": f"{uid}:{gid}",
        "command": ["sleep", "infinity"],
    }
    if getattr(args, "network_host", False):
        kwargs["network_mode"] = "host"
    else:
        kwargs["ports"] = {"22/tcp": args.ssh_port}
    if args.container_name:
        kwargs["name"] = args.container_name
    if args.hostname:
        kwargs["hostname"] = args.hostname
    devices = []
    if getattr(args, "devices", None):
        devices.extend(d if ":" in d else f"{d}:{d}" for d in args.devices)
    if getattr(args, "expose_nvme", False):
        for d in sorted(Path("/dev").glob("nvme*")):
            if d.is_block_device() or d.is_char_device():
                devices.append(f"{d}:{d}")
    if devices:
        kwargs["devices"] = devices
    return kwargs, uid, gid, home


def copy_authorized_keys(container, user, host_path, uid, gid):
    """Copy host file into container as user's ~/.ssh/authorized_keys."""
    path = Path(host_path)
    if not path.is_file():
        raise FileNotFoundError(f"Authorized keys file not found: {host_path}")
    content = path.read_bytes()

    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        ti = tarfile.TarInfo(name=".ssh/authorized_keys")
        ti.size = len(content)
        tar.addfile(ti, BytesIO(content))
    buf.seek(0)
    container.put_archive(f"/home/{user}", buf)
    # Fix ownership and permissions (container may have created .ssh as root)
    result = container.exec_run(
        [
            "sh", "-c",
            f"chown -R {uid}:{gid} /home/{user}/.ssh "
            f"&& chmod 700 /home/{user}/.ssh "
            f"&& chmod 600 /home/{user}/.ssh/authorized_keys",
        ],
        user="root",
    )
    if result.exit_code != 0:
        out = result.output.decode() if isinstance(result.output, bytes) else result.output
        raise RuntimeError(f"chown/chmod failed (exit {result.exit_code}): {out}")


def setup_sshd_and_user(container, args, uid, gid):
    """Create user, configure sshd, optionally copy keys, start sshd."""
    user = args.user
    port_line = ""
    if getattr(args, "network_host", False):
        port_line = f"echo 'Port {args.ssh_port}' >> /etc/ssh/sshd_config; "
    script = (
        "set -e; "
        "mkdir -p /run/sshd; "
        "%s"
        "echo 'PermitRootLogin no' >> /etc/ssh/sshd_config; "
        f"groupadd -o -g {gid} {user} 2>/dev/null || true; "
        f"useradd -M -u {uid} -g {gid} -d /home/{user} -s /bin/bash {user}; "
        "chown -R %s:%s /home/%s 2>/dev/null || true; "
        "echo '%s ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/%s; "
        "chmod 440 /etc/sudoers.d/%s; "
        "pgrep -x sshd >/dev/null 2>&1 || "
        "nohup /usr/sbin/sshd </dev/null >/dev/null 2>&1 &"
    ) % (port_line, uid, gid, user, user, user, user)
    result = container.exec_run(["bash", "-c", script], user="root")
    if result.exit_code != 0:
        out = result.output.decode() if isinstance(result.output, bytes) else result.output
        raise RuntimeError(
            f"Setup script failed (exit {result.exit_code}): {out}"
        )

    if args.authorized_keys:
        print("Copying authorized keys into container...")
        copy_authorized_keys(
            container, user, args.authorized_keys, uid, gid
        )


def main():
    args = parse_args()
    if args.user is None:
        args.user = getpass.getuser()
    kwargs, uid, gid, home = build_run_kwargs(args)

    if args.authorized_keys:
        path = Path(args.authorized_keys)
        if not path.is_file():
            print(f"Error: authorized-keys file not found: {path}", file=sys.stderr)
            sys.exit(1)

    print("Connecting to Docker...")
    try:
        client = docker.from_env()
    except docker.errors.DockerException as e:
        print(f"Error: cannot connect to Docker: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting container from image {args.image} "
          "(pulling image if needed, may take a while)...")
    try:
        container = client.containers.run(**kwargs)
    except docker.errors.ImageNotFound:
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)
    except docker.errors.APIError as e:
        print(f"Error running container: {e}", file=sys.stderr)
        sys.exit(1)

    print("Creating user and starting SSH inside container...")
    try:
        setup_sshd_and_user(container, args, uid, gid)
    except Exception as e:
        print(f"Error setting up SSH/user: {e}", file=sys.stderr)
        container.stop()
        sys.exit(1)

    print("Done.")
    print(f"Container running. Connect with:")
    print(f"  ssh -p {args.ssh_port} {args.user}@localhost")
    print("  (Use the host's IP instead of localhost when connecting remotely.)")


if __name__ == "__main__":
    main()
