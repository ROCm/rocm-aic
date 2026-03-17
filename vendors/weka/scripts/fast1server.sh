#!/bin/bash -x

NIC=${NIC:-udp}
NIC_QUEUES=${NIC_QUEUES:-3}

# Auto-detect the management IP from the default route
# if the caller did not supply one explicitly.
if [ -n "$WEKAIP" ]; then
    :
else
    WEKAIP=$(ip -4 route get 1.1.1.1 \
             | awk '{for(i=1;i<=NF;i++)
                       if($i=="src"){print $(i+1);exit}}')
fi
if [ -z "$WEKAIP" ]; then
    echo "ERROR: could not detect management IP"
    exit 1
fi
if [ "$NIC_QUEUES" -lt 1 ] 2>/dev/null; then
    echo "NIC_QUEUES must be >= 1 (got $NIC_QUEUES)"
    exit 1
fi

NUM_CONTAINERS=6

# Weka minimum per container is ~1.4 GB.  Keep
# defaults low for dev/test VMs; override with
# PRIMARY_MEM / SECONDARY_MEM env vars if needed.
PRIMARY_MEM="${PRIMARY_MEM:-4GB}"
SECONDARY_MEM="${SECONDARY_MEM:-1536MB}"

# Derive per-container core/role counts from NIC_QUEUES.
# Each container gets at most NIC_QUEUES cores since
# each core needs its own NIC queue.
#
# Primary container (default): drive + compute + frontend
# Secondary containers (default1-5): drive + compute only
compute_role_split() {
    local q=$1

    P_DRIVE=1
    local p_remaining=$((q - P_DRIVE))
    if [ $p_remaining -ge 2 ]; then
        P_FRONTEND=$(( p_remaining / 2 ))
        P_COMPUTE=$(( p_remaining - P_FRONTEND ))
    elif [ $p_remaining -eq 1 ]; then
        P_FRONTEND=1
        P_COMPUTE=0
    else
        P_FRONTEND=0
        P_COMPUTE=0
    fi
    P_CORES=$q

    S_DRIVE=1
    S_COMPUTE=$(( q - S_DRIVE ))
    S_CORES=$q
}

# Build a comma-separated list of core IDs:
#   core_id_list <start> <count>
core_id_list() {
    local start=$1 count=$2 ids=""
    for ((c = start; c < start + count; c++)); do
        ids="${ids:+$ids,}$c"
    done
    echo "$ids"
}

compute_role_split "$NIC_QUEUES"

# ── Helper: poll until a command succeeds ──────
wait_for() {
    local cmd="$1" label="$2"
    local retries="${3:-30}" delay="${4:-2}"
    echo "Waiting for: $label"
    for ((i=1; i<=retries; i++)); do
        eval "$cmd" && return 0
        echo "  [$i/$retries] not ready," \
             "retrying in ${delay}s..."
        sleep "$delay"
    done
    echo "ERROR: Timed out waiting for: $label" >&2
    return 1
}

# ── Stop and remove existing containers ────────
echo "Removing existing containers..."
sudo weka local stop -f
sudo weka local rm --all --force
wait

# Resolve the --net argument for container setup.
#   NIC=udp          → kernel UDP networking (default)
#   NIC=<iface>      → DPDK via PCI address
#   NIC=0000:xx:yy.z → DPDK with explicit PCI address
if [ "$NIC" == "udp" ]; then
    NET_ARG="udp"
    echo "Data NIC   : UDP (kernel networking)"
elif [[ "$NIC" =~ ^[0-9a-fA-F]{4}: ]]; then
    NET_ARG="$NIC"
    echo "Data NIC   : $NIC (PCI address)"
else
    wait_for \
        "ethtool -i $NIC &>/dev/null" \
        "NIC $NIC available" 10 2
    NET_ARG=$(ethtool -i "$NIC" 2>/dev/null \
              | awk '/bus-info:/{print $2}')
    if [ -z "$NET_ARG" ]; then
        echo "ERROR: cannot resolve PCI address" \
             "for $NIC"
        exit 1
    fi
    echo "Data NIC   : $NIC -> $NET_ARG (DPDK)"
fi

echo "System IP is: $WEKAIP"
echo "NIC queues : $NIC_QUEUES"
echo "Primary    : $P_CORES cores, $PRIMARY_MEM" \
     "($P_DRIVE drv, $P_COMPUTE cmp, $P_FRONTEND fe)"
echo "Secondary  : $S_CORES cores, $SECONDARY_MEM" \
     "($S_DRIVE drv, $S_COMPUTE cmp)"

# ── Create containers ─────────────────────────
echo "Creating containers..."

next_core=1

# Primary container (default) — has frontends
p_ids=$(core_id_list "$next_core" "$P_CORES")
next_core=$(( next_core + P_CORES ))

frontend_flags="--frontend-dedicated-cores $P_FRONTEND"
if [ "$P_FRONTEND" -eq 0 ]; then
    frontend_flags="--no-frontends"
fi

# shellcheck disable=SC2086
sudo weka local setup container \
    --name default \
    --net "$NET_ARG" \
    --cores "$P_CORES" \
    --cores-ids "$p_ids" \
    --drives-dedicated-cores "$P_DRIVE" \
    --compute-dedicated-cores "$P_COMPUTE" \
    $frontend_flags \
    --memory "$PRIMARY_MEM" \
    --failure-domain fd1

# Secondary containers (default1–default5)
for idx in $(seq 1 $(( NUM_CONTAINERS - 1 ))); do
    s_ids=$(core_id_list "$next_core" "$S_CORES")
    next_core=$(( next_core + S_CORES ))
    port=$(( 14000 + idx * 1000 ))

    compute_flag=""
    if [ "$S_COMPUTE" -gt 0 ]; then
        compute_flag="--compute-dedicated-cores $S_COMPUTE"
    fi

    # shellcheck disable=SC2086
    sudo weka local setup container \
        --name "default${idx}" \
        --net "$NET_ARG" \
        --cores "$S_CORES" \
        --cores-ids "$s_ids" \
        --drives-dedicated-cores "$S_DRIVE" \
        $compute_flag \
        --no-frontends \
        --base-port "$port" \
        --memory "$SECONDARY_MEM" \
        --failure-domain "fd$(( idx + 1 ))"
done

# Poll until all containers are Running
wait_for \
    "[ \$(sudo weka local ps 2>/dev/null \
       | grep -c Running) -ge $NUM_CONTAINERS ]" \
    "all $NUM_CONTAINERS containers Running" 30 3

# ── Form cluster ──────────────────────────────
echo "Forming cluster..."
host_ips=""
for idx in $(seq 0 $(( NUM_CONTAINERS - 1 ))); do
    port=$(( 14000 + idx * 1000 ))
    host_ips="${host_ips:+$host_ips,}${WEKAIP}:${port}"
done

sudo weka cluster create weka{1..6} \
    --host-ips="$host_ips"

# Poll until cluster API is reachable
wait_for \
    "sudo weka status &>/dev/null" \
    "cluster API reachable" 20 3

# ── Add drives in parallel ────────────────────
echo "Adding drives to containers..."
for i in $(seq 0 $(( NUM_CONTAINERS - 1 ))); do
    sudo weka cluster drive add \
        "$i" "/dev/nvme$((i+1))n1" &
done
wait

# ── Cluster config ────────────────────────────
echo "Setting cluster name and stripe..."
sudo weka cluster update \
    --cluster-name weka-one-server \
    --data-drives=3 \
    --parity-drives=2
sudo weka cluster hot-spare 0

# ── Start IO ──────────────────────────────────
echo "Starting IO..."
sudo weka cluster start-io

# Poll until IO is confirmed STARTED
wait_for \
    "sudo weka status -J 2>/dev/null \
     | jq -e '.io_status == \"STARTED\"' \
       &>/dev/null" \
    "IO STARTED" 30 3

# ── Create filesystems ────────────────────────
echo "Creating default FS group and default FS..."
sudo weka fs group create default
sudo weka fs create default default 5TiB
