#! /bin/bash
WEKAIP=10.158.213.59
WEKAIP2=10.158.215.67
NIC="eth0"

# ── Helper: poll until a command succeeds or times out ──────────────────────
wait_for() {
    local cmd="$1" label="$2" retries="${3:-30}" delay="${4:-2}"
    echo "Waiting for: $label"
    for ((i=1; i<=retries; i++)); do
        eval "$cmd" && return 0
        echo "  [$i/$retries] not ready, retrying in ${delay}s..."
        sleep "$delay"
    done
    echo "ERROR: Timed out waiting for: $label" >&2
    return 1
}

echo "System 1 IP: $WEKAIP  |  System 2 IP: $WEKAIP2"

# ── Stop containers on both hosts IN PARALLEL, then wait ────────────────────
echo "Stopping existing containers..."
ssh $WEKAIP  sudo weka local stop -f &
ssh $WEKAIP2 sudo weka local stop -f &
wait

# ── Remove containers on both hosts IN PARALLEL, then wait ──────────────────
echo "Removing existing containers..."
ssh $WEKAIP  sudo weka local rm --all --force &
ssh $WEKAIP2 sudo weka local rm --all --force &
wait

# ── Create containers — WEKAIP and WEKAIP2 run fully in parallel ─────────────
echo "Creating containers..."
(
    ssh $WEKAIP sudo weka local setup container --name default  --net $NIC --cores 5 --cores-ids 1,2,3,4,5   --drives-dedicated-cores 1 --compute-dedicated-cores 2 --frontend-dedicated-cores 2 --memory 20GB --failure-domain fd1
    ssh $WEKAIP sudo weka local setup container --name default1 --net $NIC --cores 3 --cores-ids 6,7,8       --drives-dedicated-cores 1 --compute-dedicated-cores 2 --no-frontends --base-port 15000 --memory 16GB --failure-domain fd2
    ssh $WEKAIP sudo weka local setup container --name default2 --net $NIC --cores 3 --cores-ids 9,10,11     --drives-dedicated-cores 1 --compute-dedicated-cores 2 --no-frontends --base-port 16000 --memory 16GB --failure-domain fd3
) &
(
    ssh $WEKAIP2 sudo weka local setup container --name default  --net $NIC --cores 5 --cores-ids 1,2,3,4,5  --drives-dedicated-cores 1 --compute-dedicated-cores 2 --frontend-dedicated-cores 2 --memory 20GB --failure-domain fd4
    ssh $WEKAIP2 sudo weka local setup container --name default1 --net $NIC --cores 3 --cores-ids 6,7,8      --drives-dedicated-cores 1 --compute-dedicated-cores 2 --no-frontends --base-port 15000 --memory 16GB --failure-domain fd5
    ssh $WEKAIP2 sudo weka local setup container --name default2 --net $NIC --cores 3 --cores-ids 9,10,11    --drives-dedicated-cores 1 --compute-dedicated-cores 2 --no-frontends --base-port 16000 --memory 16GB --failure-domain fd6
) &
wait   # both host subshells must finish before proceeding

# ── Poll until all 6 containers are Running ─────────────
echo "Waiting for containers to reach Running state..."
wait_for \
    "[ \$(sudo weka local ps 2>/dev/null | grep -c Running) -ge 3 ]" \
    "containers Running on $WEKAIP" 30 3
wait_for \
    "[ \$(ssh $WEKAIP2 sudo weka local ps 2>/dev/null | grep -c Running) -ge 3 ]" \
    "containers Running on $WEKAIP2" 30 3

# ── Form cluster ──────────────────────────────────────────────────────────────
echo "Forming cluster..."
sudo weka cluster create weka{1..6} \
    --host-ips="$WEKAIP":14000,"$WEKAIP":15000,"$WEKAIP":16000,"$WEKAIP2":14000,"$WEKAIP2":15000,"$WEKAIP2":16000

# Poll until cluster shows INITIALIZING or better
wait_for "sudo weka status &>/dev/null" "cluster API reachable" 20 3

# ── Add drives IN PARALLEL ────────────────────────
#echo "Adding drives to containers..."
#for i in 0 1 2; do sudo weka cluster drive add $i /dev/nvme$((i+1))n1 & done
#for i in 3 4 5; do sudo weka cluster drive add $i /dev/nvme$((i-2))n1 & done
#wait
echo "Adding drives to containers..."
for i in {0..5}; do
    base=$(( (i % 3) * 2 ))
    sudo weka cluster drive add "$i" /dev/nvme${base}n1 /dev/nvme$((base+1))n1 &
done
wait


# ── Cluster config ───────
echo "Setting cluster name and stripe..."
sudo weka cluster update --cluster-name weka-two-server --data-drives=3 --parity-drives=2
sudo weka cluster hot-spare 0

echo "Starting IO..."
sudo weka cluster start-io

# Poll until IO is confirmed STARTED
wait_for \
    "sudo weka status -J 2>/dev/null | jq -e '.io_status == \"STARTED\"' &>/dev/null" \
    "IO STARTED" 20 3

# ── Create filesystem group and FS ───────────────────────────────────────────
echo "Creating default FS group and default FS..."
sudo weka fs group create default
sudo weka fs create default default \
    "$(sudo weka status -J | jq .capacity.unprovisioned_bytes)"

echo "Done."