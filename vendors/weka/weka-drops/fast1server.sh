#! /bin/bash -x
WEKAIP=10.158.213.59
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

echo "System IP is: $WEKAIP"

# ── Stop and remove existing containers ─────────────────────────────────────
echo "Removing existing containers..."
sudo weka local stop -f
sudo weka local rm --all --force

wait

# ── Create containers ────────────────────────────────────────────────────────
echo "Creating containers..."
sudo weka local setup container --name default  --net $NIC --cores 5 --cores-ids 1,2,3,4,5   --drives-dedicated-cores 1 --compute-dedicated-cores 2 --frontend-dedicated-cores 2 --memory 20GB --failure-domain fd1
sudo weka local setup container --name default1 --net $NIC --cores 3 --cores-ids 6,7,8        --drives-dedicated-cores 1 --compute-dedicated-cores 2 --no-frontends --base-port 15000 --memory 16GB --failure-domain fd2
sudo weka local setup container --name default2 --net $NIC --cores 3 --cores-ids 9,10,11      --drives-dedicated-cores 1 --compute-dedicated-cores 2 --no-frontends --base-port 16000 --memory 16GB --failure-domain fd3
sudo weka local setup container --name default3 --net $NIC --cores 3 --cores-ids 12,13,14     --drives-dedicated-cores 1 --compute-dedicated-cores 2 --no-frontends --base-port 17000 --memory 16GB --failure-domain fd4
sudo weka local setup container --name default4 --net $NIC --cores 3 --cores-ids 15,16,17     --drives-dedicated-cores 1 --compute-dedicated-cores 2 --no-frontends --base-port 18000 --memory 16GB --failure-domain fd5
sudo weka local setup container --name default5 --net $NIC --cores 3 --cores-ids 18,19,20     --drives-dedicated-cores 1 --compute-dedicated-cores 2 --no-frontends --base-port 19000 --memory 16GB --failure-domain fd6

# Poll until all 6 containers are Running (replaces: sleep 15)
wait_for \
    "[ \$(sudo weka local ps 2>/dev/null | grep -c Running) -ge 6 ]" \
    "all 6 containers Running" 30 3

# ── Form cluster ─────────────────────────────────────────────────────────────
echo "Forming cluster..."
sudo weka cluster create weka{1..6} \
    --host-ips="$WEKAIP":14000,"$WEKAIP":15000,"$WEKAIP":16000,"$WEKAIP":17000,"$WEKAIP":18000,"$WEKAIP":19000

# Poll until cluster API is reachable (replaces: sleep 5)
wait_for "sudo weka status &>/dev/null" "cluster API reachable" 20 3

# ── Add drives IN PARALLEL ───────────────────────────────────────────────────
echo "Adding drives to containers..."
for i in {0..5}; do sudo weka cluster drive add $i /dev/nvme$((i+1))n1 & done
wait

# ── Cluster config — synchronous, no waits needed ───────────────────────────
echo "Setting cluster name and stripe..."
sudo weka cluster update --cluster-name weka-one-server --data-drives=3 --parity-drives=2
sudo weka cluster hot-spare 0

# ── Start IO ─────────────────────────────────────────────────────────────────
echo "Starting IO..."
sudo weka cluster start-io

# Poll until IO is confirmed STARTED (replaces: sleep 5)
wait_for \
    "sudo weka status -J 2>/dev/null | jq -e '.io_status == \"STARTED\"' &>/dev/null" \
    "IO STARTED" 30 3

# ── Create filesystems ───────────────────────────────────────────────────────
echo "Creating default FS group and default FS..."
sudo weka fs group create default
sudo weka fs create default default 5TiB