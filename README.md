# rocm-icms

AMD internal exploration of storage infrastructure for
ROCm-based GPU clusters, inspired by NVIDIA's
[Inference Context Memory Storage (ICMS)][icms] platform
(recently rebranded **CMX**). ICMS uses BlueField-4 DPUs
and disaggregated NVMe flash to create a shared KV-cache
tier for large-scale AI inference; this repo investigates
analogous approaches on AMD hardware.

## Repository Layout

```
vendors/
  weka/       WEKA-FS proof-of-concept (Ansible, Docker,
              shell scripts for cluster deployment)
```

## References

- [NVIDIA ICMS technical blog][icms]
- [WEKA blog on BlueField-4 and ICMS][weka]

[icms]: https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/
[weka]: https://www.weka.io/blog/ai-ml/demystifying-the-bluefield-4-inference-context-memory-storage-announcement/
