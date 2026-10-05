# LMCache P2P KV-cache sharing — local VM test

AIC includes a local end-to-end functional test for LMCache
**peer-to-peer KV-cache sharing** that requires no physical GPU or RDMA
hardware. Two QEMU KVM virtual machines are booted with emulated AMD
Pensando ionic RDMA NICs (via [rocm-ernic](https://github.com/ROCm/rocm-ernic))
and LMCache is run in P2P mode inside each VM using the in-repo Docker
Compose stack.

## What it validates

LMCache P2P allows multiple server instances to share KV-cache blocks
directly over RDMA rather than recomputing prefixes. When VM2 receives
the same prefix that VM1 already cached, it fetches the KV blocks from
VM1's L1 buffer via a one-sided RDMA read — without central storage in
the hot path.

The test sends the same 512-token prompt to VM2's vLLM twice and asserts
`lmcache_mp_p2p_load_count_total > 0` on VM2's LMCache metrics endpoint.

## Architecture

```
Host
├── qemu-tool compose (vfio-user-ernic-2vm)
│     ├── ernic-hub     (rocm-ernic TCP manager, socket ernic-1.sock)
│     ├── ernic-worker  (rocm-ernic TCP worker,  socket ernic-2.sock)
│     ├── qemu-1        (VM1, ionic NIC via ernic-1.sock, SSH :12230)
│     └── qemu-2        (VM2, ionic NIC via ernic-2.sock, SSH :12231)
│
├── VM1 (primary)  192.168.200.10
│     ├── lmcache coordinator   :9300
│     └── lmcache server        :6555  --p2p-advertise-url VM1:18200
│
└── VM2 (secondary)  192.168.200.20
      ├── lmcache server        :6555  --p2p-advertise-url VM2:18200
      └── vllm serve (llm-emu)  :8000  → lmcache on 127.0.0.1:6555
```

The ernic TCP mesh (`tcp:manager/worker`) carries both the Ethernet
frames and RDMA verbs traffic between the two ionic NICs, so the full
RDMA data path is exercised with no physical RDMA hardware.

## Prerequisites

| Requirement | Notes |
|---|---|
| `/dev/kvm` | KVM acceleration; 4+ vCPUs recommended per VM |
| Docker with Compose v2 | `make ensure-compose` installs if missing |
| `qemu-tool` | Auto-installed from PyPI if not present |
| `HF_TOKEN` | HuggingFace token for model download in VM2 |
| `AIC_LMCACHE_P2P_LMCACHE_IMAGE_REF` | LMCache image ref (or `LMCACHE_IMAGE_REF`) |
| `AIC_LMCACHE_P2P_VLLM_IMAGE_REF` | vLLM image ref (or `IMAGE_REF`) |

The ionic-flavour guest disk is pulled automatically from
`batesste-ci-images-ubuntu-qcow2-gen-ionic` on first run and cached in
`AIC_LMCACHE_P2P_VM_IMAGES_DIR`. Subsequent runs skip the pull.

## Running the test

```bash
make test-lmcache-p2p-local \
  HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  AIC_LMCACHE_P2P_LMCACHE_IMAGE_REF=<lmcache-image> \
  AIC_LMCACHE_P2P_VLLM_IMAGE_REF=<vllm-image>
```

The default model (`HuggingFaceTB/SmolLM2-135M-Instruct`, ~270 MB fp16)
is deliberately small so it fits within the 4 GiB VM memory budget and
downloads quickly.

## Configuration reference

All variables have the `AIC_LMCACHE_P2P_` prefix. Pass them on the
`make` command line or export them before running.

| Variable | Default | Description |
|---|---|---|
| `AIC_LMCACHE_P2P_QCOW2_IMAGE` | `batesste-ci-images-ubuntu-qcow2-gen-ionic:…` | Pinned ionic-flavour guest disk image |
| `AIC_LMCACHE_P2P_VM_IMAGES_DIR` | `/var/lib/qemu-tool/images` | Directory where qcow2 disks are placed |
| `AIC_LMCACHE_P2P_VM1_NAME` | `qemu-minimal` | VM1 image basename |
| `AIC_LMCACHE_P2P_VM2_NAME` | `qemu-minimal-2` | VM2 image basename |
| `AIC_LMCACHE_P2P_VM1_SSH_PORT` | `12230` | Host port for VM1 SSH |
| `AIC_LMCACHE_P2P_VM2_SSH_PORT` | `12231` | Host port for VM2 SSH |
| `AIC_LMCACHE_P2P_VM_VCPUS` | `4` | vCPUs per VM (4 is the minimum for `ionic_rdma`) |
| `AIC_LMCACHE_P2P_VM_MEM_MB` | `4096` | RAM per VM in MiB |
| `AIC_LMCACHE_P2P_READY_S` | `300` | SSH / vLLM readiness timeout in seconds |
| `AIC_LMCACHE_P2P_MODEL` | `HuggingFaceTB/SmolLM2-135M-Instruct` | Model served by vLLM in VM2 |
| `AIC_LMCACHE_P2P_LMCACHE_IMAGE_REF` | `$(LMCACHE_IMAGE_REF)` | LMCache container image |
| `AIC_LMCACHE_P2P_VLLM_IMAGE_REF` | `$(IMAGE_REF)` | vLLM container image |
| `AIC_LMCACHE_P2P_LMCACHE_L1_SIZE_GB` | `0.5` | L1 DRAM cap per server in GiB |
| `AIC_LMCACHE_P2P_LMCACHE_PORT` | `6555` | LMCache HTTP API port |
| `AIC_LMCACHE_P2P_COORD_PORT` | `9300` | Coordinator port |
| `AIC_LMCACHE_P2P_PORT` | `18200` | P2P RDMA transfer-channel port |
| `AIC_LMCACHE_P2P_VLLM_PORT` | `8000` | vLLM OpenAI API port |
| `AIC_LMCACHE_P2P_VM1_IP` | `192.168.200.10` | VM1 ionic interface IP |
| `AIC_LMCACHE_P2P_VM2_IP` | `192.168.200.20` | VM2 ionic interface IP |
| `AIC_LMCACHE_P2P_WORK_DIR` | `/tmp/aic-lmcache-p2p-test` | Scratch directory on the host |

## Test steps (what `make test-lmcache-p2p-local` does)

1. **Install qemu-tool** from PyPI if not present.
2. **Pull and extract** the ionic guest disk from `AIC_LMCACHE_P2P_QCOW2_IMAGE`
   into `AIC_LMCACHE_P2P_VM_IMAGES_DIR`. Skipped on warm runs.
3. **Start the ernic-2vm compose stack** via `qemu-tool compose
   --stack vfio-user-ernic-2vm up -d`. This starts four containers:
   `ernic-hub`, `ernic-worker`, `qemu-1`, `qemu-2`.
4. **Wait for SSH** on both VMs (up to `AIC_LMCACHE_P2P_READY_S`).
5. **Push** `docker/compose/lmcache-p2p/docker-compose.yml` and
   `scripts/lmcache-p2p-guest-setup.sh` into both VMs over SCP.
6. **Configure and start** each VM's LMCache compose stack via SSH:
   - VM1 (`primary`): loads `ionic`/`ionic_rdma`, configures the ionic
     interface as `192.168.200.10/24`, starts coordinator + lmcache server
     with P2P flags.
   - VM2 (`secondary`): same NIC setup at `192.168.200.20/24`, starts
     lmcache server peered to VM1's coordinator, then starts vLLM
     (llm-emu executor).
7. **Wait for vLLM** on VM2 (up to `AIC_LMCACHE_P2P_READY_S`).
8. **Run P2P assertion**: send the same prompt twice to VM2's vLLM,
   check `lmcache_mp_p2p_load_count_total > 0` on VM2's LMCache metrics.

## In-VM Compose file

`docker/compose/lmcache-p2p/docker-compose.yml` runs inside each VM.
It has two Compose profiles:

| Profile | Services |
|---|---|
| `primary` | `coordinator` + `lmcache` |
| `secondary` | `lmcache` + `vllm` |

Both `lmcache` services pass `--p2p-transfer-engine nixl` and
`--l1-align-bytes 65536`. NIXL is built with UCX `--with-verbs` in the
AIC image, so it can use the ionic RDMA verbs device once the
`ionic_rdma` module is loaded in the guest.

## CI

The test runs on GitHub Actions via
`.github/workflows/rocm-aic-lmcache-p2p-test.yml`:

- **PR gate**: triggers on changes to `docker/compose/lmcache-p2p/**`,
  `scripts/lmcache-p2p-guest-setup.sh`, `mk/bench.mk`, and the workflow
  file itself.
- **Nightly schedule**: `cron: '13 2 * * *'` (off-peak UTC).
- **Manual**: `workflow_dispatch` with optional image override.

The job runs inside the `batesste-ci-images-ubuntu-qemu-libvfio-user`
container image (which has qemu-tool pre-installed) with `/dev/kvm` and
the Docker socket mounted, matching the pattern used by qemu-minimal's
own CI.

The ionic guest qcow2 (~3.5 GB compressed) is cached with
`actions/cache` keyed on the image tag extracted from `AIC_LMCACHE_P2P_QCOW2_IMAGE`.

## Troubleshooting

**VM does not become SSH-reachable**
: Check that `/dev/kvm` is accessible. The `vfio-user-ernic-2vm` compose
  stack logs (`qemu-tool compose --stack vfio-user-ernic-2vm logs`) show
  QEMU boot output. Increase `AIC_LMCACHE_P2P_READY_S` if the host is slow.

**`ionic_rdma` fails to load in guest**
: `ionic_rdma` requires at least 4 vCPUs online in the guest
  (`ionic_lif_size()` sets `neqs = min(rdma.eq_qtype.qid_count, num_online_cpus())`
  and rejects anything below `IONIC_EQ_COUNT_MIN = 4`). Do not set
  `AIC_LMCACHE_P2P_VM_VCPUS` below 4.

**P2P hit count = 0 after both requests**
: The ionic interface may not have come up in time. Check VM2's lmcache
  log (`docker compose -f /tmp/lmcache-p2p/docker-compose.yml logs lmcache`
  via SSH) for NIXL/UCX peer connection errors. Confirm that
  `ibv_devices` shows a device in both VMs.

**"LMCACHE_IMAGE_REF not set" error**
: Pass `AIC_LMCACHE_P2P_LMCACHE_IMAGE_REF=<image>` on the make command
  line, or export `LMCACHE_IMAGE_REF` in the environment.
