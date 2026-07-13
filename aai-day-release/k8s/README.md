# aai-day-release Kubernetes

Kustomize-based Kubernetes deployment of the aai-day inference stack
(vLLM + LMCache + hipFile + NIXL) on ROCm AMD GPUs.

## Architecture

Each Pod runs two containers that mirror the `docker-compose.yml` stack:

```
Pod: aai-day-vllm
├── lmcache-server  (native sidecar, starts first)
│     vLLM-compatible KV cache server
│     Listens on tcp://127.0.0.1:6555 (ZMQ)
│     Exposes /metrics on :8080
│
└── vllm  (main container)
      vLLM OpenAI API server
      Connects to lmcache-server via LMCacheMPConnector
      Exposes API + /metrics on :8000
```

Scaling is **data-parallel**: add replicas to run multiple independent
inference Pods. Each Pod has its own LMCache KV tier. For prefix-cache-aware
routing across Pods (so cache-warm Pods are preferred), see
`recipies/llm-d/` in this repository for a full llm-d + InferencePool setup.

## Prerequisites

Install tools:

```bash
bash prereqs.sh
```

Verify cluster access:

```bash
kubectl get nodes
kubectl describe nodes | grep -i amd.com/gpu
```

## Quick start (TP=2, DRAM L1)

```bash
# 1. Register HuggingFace token
export HF_TOKEN=hf_...
just register-hf-token

# 2. Set the model (required — patch the deployment or set env var)
# Edit manifests/base/deployment.yaml and change the `--model` arg,
# or apply a patch via your flavor's kustomization.yaml.

# 3. Set your image reference
export AAI_DAY_IMAGE_REGISTRY=ghcr.io/your-org
export AAI_DAY_IMAGE_TAG=latest

# 4. Deploy
just deploy-tp2

# 5. Wait for the Pod to be ready
just status

# 6. Forward the API port and test
just port-forward &
just health-check
```

## Flavors

| Flavor | GPUs | L1 | L2 | Storage needed |
|--------|------|----|----|----------------|
| `tp2-dram-l1` | 2 | DRAM 20 GiB | none | none |
| `tp2-nfs-l2` | 2 | DRAM 20 GiB | NFS POSIX | ReadWriteMany PVC |
| `tp4-nvme-l2` | 4 | DRAM 20 GiB | NVMe AIS_MT | ReadWriteOnce PVC |
| `tp8-gds-l1` | 8 | NVMe GDS slab | none | ReadWriteOnce PVC |

Deploy a specific flavor:

```bash
just deploy-tp2-nfs   # DRAM L1 + NFS L2
just deploy-tp4       # TP=4 + NVMe L2
just deploy-tp8       # TP=8 + GDS NVMe slab L1
```

## RDMA NIC library injection (optional)

When using NIXL AIS_MT or POSIX L2 adapters with RDMA, the lmcache-server
sidecar needs access to the host's RDMA user-space library.

Apply the appropriate host-config component from your flavor's kustomization:

```yaml
# In your flavor kustomization.yaml:
components:
  - ../../host-configs/mlx5    # Mellanox ConnectX / NVIDIA NICs
  # or
  - ../../host-configs/bnxt-re  # Broadcom RoCE NICs
```

The patches mount the NIC library from the host into the container. Adjust
the host path in the patch file if your distribution places the library
elsewhere.

## NFS L2 tier (`tp2-nfs-l2`)

The `tp2-nfs-l2` flavor adds a `ReadWriteMany` PVC for the POSIX L2 pool.
This enables multiple Pods (replicas) to share a common NFS KV cache.

Options for the PVC:

1. **Ansible provisioned**: run `ansible-playbook site.yml --tags nfs-rdma`
   from `aai-day-release/ansible/` to set up NFS-over-RDMA, then create a
   PV pointing to that mount.

2. **StorageClass**: install an NFS StorageClass such as
   `nfs-subdir-external-provisioner` and set `storageClassName` in
   `manifests/flavors/tp2-nfs-l2/pvc-nfs.yaml`.

## Multi-node considerations

This Kustomize deployment is designed for single-node multi-GPU inference
(tensor-parallel within one node). Multi-node deployment considerations:

- **Data-parallel scale-out** (multiple replicas, each on a different node):
  works with any flavor. Each Pod serves requests independently.
  Use a LoadBalancer Service or Ingress in front.

- **Cross-node tensor-parallelism** (`--tensor-parallel-size` > node GPU count):
  requires distributed inference support beyond the aai-day stack.
  See `recipies/llm-d/inference-scheduling/` for a helmfile-based approach
  with `multinode: true`.

- **Shared NFS L2 cache across nodes**: deploy the `tp2-nfs-l2` flavor with
  multiple replicas and a `ReadWriteMany` NFS PVC. All replicas read/write
  the same KV cache files. The NFS-over-RDMA ansible play provisions a
  low-latency RDMA-backed NFS server for this purpose.

## Teardown

```bash
just teardown
```

This deletes the namespace and all resources within it.

## Upgrading the image

Update the image reference and roll the Deployment:

```bash
export AAI_DAY_IMAGE_TAG=v2
just deploy-tp2
kubectl rollout status deployment/aai-day-vllm --namespace aai-day
```

## Step up: llm-d with InferencePool

For prefix-cache-aware routing, KV cache scoring, and multi-node inference
scheduling, see `recipies/llm-d/` in this repository. That recipe uses
llm-d v0.5.1 with Helm and Kustomize and is the production-grade path for
cluster-scale deployments.
