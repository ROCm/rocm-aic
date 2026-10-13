# vLLM + LMCache (MP mode) + NIXL on ROCm, for AMD Instinct GPUs.
#
#   docker buildx bake                  # both images, local cache, loaded into docker
#   docker buildx bake vllm             # just the serving image
#   GPU_ARCH=gfx942 docker buildx bake  # retarget to MI300X
#   REGISTRY=ghcr.io/sbates130272/ CACHE=registry docker buildx bake --push

variable "REGISTRY" {
  default     = ""
  description = "Image prefix, with trailing slash. Empty = local-only images."
}

variable "TAG" {
  default = "dev"
}

# MI210/MI250 = gfx90a, MI300X/MI325X = gfx942, MI350X = gfx950.
# ROCm 10.1 ships per-GFX wheels, so one arch is also the slimming lever:
# all 25 targets is 24 GB of site-packages, one target is 6.7 GB.
variable "GPU_ARCH" {
  default = "gfx90a"
}

variable "RUNTIME_BASE" { default = "ubuntu:26.04" }
variable "PYTHON_VERSION" { default = "3.14" }
variable "ROCM_VERSION" { default = "10.1.0" }
variable "TORCH_VERSION" { default = "2.13.0+rocm10.1.0" }
variable "TORCHVISION_VERSION" { default = "0.28.0+rocm10.1.0" }
variable "ROCM_INDEX" { default = "https://stable.repo.amd.com/rocm/whl-next/" }

variable "VLLM_REF" { default = "v0.31.0" }
variable "LMCACHE_REF" { default = "v0.5.5" }
variable "NIXL_REF" { default = "v1.5.0" }
variable "UCX_REF" { default = "v1.22.0" }

variable "MAX_JOBS" { default = "64" }

# local | registry | none. "local" and "registry" both need a docker-container
# builder; the default docker driver cannot export cache.
variable "CACHE" { default = "local" }

# Where CACHE=local writes. Deliberately NOT inside the repo: the local cache
# exporter is containerd's content store, which needs atomic renames and
# startedat lock files. On NFS those semantics do not hold, so ingest entries
# are orphaned and every build ends in "failed to resume the status from path
# ... will recreate them" and "ERROR: failed to remove ingest directory".
# The images still build -- it is the export tail that fails -- but the
# exported cache is incomplete, so cache-from stops paying for itself.
# Keep this on a local filesystem. Nothing here is precious; delete freely.
variable "CACHE_DIR" { default = "/var/tmp/buildx-cache" }

function "cache_from" {
  params = [name]
  result = (
    CACHE == "none" ? [] :
    CACHE == "registry" ? ["type=registry,ref=${REGISTRY}buildcache:${name}-${GPU_ARCH}"] :
    ["type=local,src=${CACHE_DIR}/${name}-${GPU_ARCH}"]
  )
}

function "cache_to" {
  params = [name]
  result = (
    CACHE == "none" ? [] :
    CACHE == "registry" ? ["type=registry,ref=${REGISTRY}buildcache:${name}-${GPU_ARCH},mode=max"] :
    ["type=local,dest=${CACHE_DIR}/${name}-${GPU_ARCH},mode=max"]
  )
}

target "_common" {
  context    = "."
  dockerfile = "Dockerfile"
  platforms  = ["linux/amd64"]
  # Load into the local daemon unless a registry is configured.
  output = REGISTRY == "" ? ["type=docker"] : ["type=registry"]
  args = {
    RUNTIME_BASE   = RUNTIME_BASE
    PYTHON_VERSION = PYTHON_VERSION
    GPU_ARCH      = GPU_ARCH
    ROCM_VERSION  = ROCM_VERSION
    TORCH_VERSION = TORCH_VERSION
    TORCHVISION_VERSION = TORCHVISION_VERSION
    ROCM_INDEX    = ROCM_INDEX
    VLLM_REF      = VLLM_REF
    LMCACHE_REF   = LMCACHE_REF
    NIXL_REF      = NIXL_REF
    UCX_REF       = UCX_REF
    MAX_JOBS      = MAX_JOBS
  }
  labels = {
    "org.opencontainers.image.description" = "vLLM ${VLLM_REF} + LMCache ${LMCACHE_REF} (MP) + NIXL ${NIXL_REF}, ROCm ${ROCM_VERSION}, ${GPU_ARCH}"
  }
}

# The vLLM server, with the LMCache MP connector and NIXL available.
target "vllm" {
  inherits   = ["_common"]
  target     = "vllm"
  tags       = ["${REGISTRY}vllm-lmcache-rocm:${TAG}-${GPU_ARCH}"]
  cache-from = cache_from("vllm")
  cache-to   = cache_to("vllm")
}

# The standalone LMCache MP cache server. Same wheels, no vLLM.
target "lmcache-server" {
  inherits   = ["_common"]
  target     = "lmcache-server"
  tags       = ["${REGISTRY}lmcache-server-rocm:${TAG}-${GPU_ARCH}"]
  cache-from = cache_from("lmcache-server")
  cache-to   = cache_to("lmcache-server")
}

group "default" {
  targets = ["vllm", "lmcache-server"]
}
