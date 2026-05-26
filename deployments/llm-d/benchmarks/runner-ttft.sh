#!/bin/bash

# Runs LMCache TTFT latency bench for gpu prefill, cpu cache hit, nfs cache hit.
# TP=1,8 with scaled chunk size to transfer granularity constant.

export OUTDIR=results-runner-ttft
mkdir -p ${OUTDIR}

export NAME=ttft-nfs-only; rm -Rf results/sweeps/$NAME; just sweep sweep-configs/ttft-latency/bench-ttft-lmcache-nfs-only.yaml ${NAME} | tee output_${NAME}
just results-aggregate ${NAME} -o ${OUTDIR}/lmcache_nfs_aggregated_results.json 

export NAME=ttft-cpu-only; rm -Rf results/sweeps/$NAME; just sweep sweep-configs/ttft-latency/bench-ttft-lmcache-cpu-only.yaml ${NAME} | tee output_${NAME}
just results-aggregate ${NAME} -o ${OUTDIR}/lmcache_cpu_aggregated_results.json

# Generate plot
export PYTHONPATH=$PWD/scripts;
python -m plots.plot_config sweep-configs/ttft-latency/plot-config-ttft-lmcache-prefill-cpu-nfs.yaml;
