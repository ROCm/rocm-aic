#!/usr/bin/env bash
# Liveness probe for an in-flight run-rocjitsu-cpu.sbatch job.
#
# [6d] buffers its output, so a run that is slowly making progress and one that
# is wedged look identical from the job log.  This asks the host and the guest
# what they are actually doing right now.  Run it on the node holding the job
# (e.g. `srun --jobid=<id> --pty .slurm/probe-rocjitsu.sh`), not from a login
# node -- the containers and the SSH port are node-local.
#
# Defaults match mk/bench.mk's RJ_WORK_DIR and RJ_SSH_PORT.
set -uo pipefail

WORK_DIR="${RJ_WORK_DIR:-/tmp/aic-rocjitsu-test}"
SSH_PORT="${RJ_SSH_PORT:-12222}"
KEY="${WORK_DIR}/vm/id_rsa"
VMINFO="${WORK_DIR}/vm/vm-info.json"

echo "host containers:"
docker ps --format '  {{.Names}}  {{.Status}}' 2>/dev/null | head

echo
echo "rocjitsu container CPU (emulated GPU work shows up here, not on a GPU):"
docker stats --no-stream --format '  {{.Name}}  cpu={{.CPUPerc}}  mem={{.MemUsage}}' \
    aic-rj-rocjitsu aic-rj-qemu 2>/dev/null

echo
echo "rocjitsu emulator log:"
docker logs --tail 10 aic-rj-rocjitsu 2>&1 | sed 's/^/  /'

# -I: the work dir is job-controlled data, so do not let the interpreter pick
# up anything from its directory.
VM_USER=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['username'])" "${VMINFO}" 2>/dev/null)

if [ -z "${VM_USER}" ] || [ ! -f "${KEY}" ]; then
    echo
    echo "(no vm-info.json / key under ${WORK_DIR} yet — VM not started)"
    exit 0
fi

SSH_FLAGS=(-i "${KEY}" -o StrictHostKeyChecking=no -o ConnectTimeout=10
           -o BatchMode=yes -p "${SSH_PORT}")

echo
echo "guest containers:"
ssh "${SSH_FLAGS[@]}" "${VM_USER}@localhost" \
    "sudo docker ps --format '  {{.Names}}  {{.Status}}'" 2>&1 | head

echo
echo "guest vllm log tail:"
ssh "${SSH_FLAGS[@]}" "${VM_USER}@localhost" \
    "sudo docker logs --tail 12 aic-vllm-gpu0 2>&1 || echo '  (no vllm container — probe-only run, or not started yet)'" \
    2>&1 | head -15
