#!/bin/bash

set -x

export EXP="cliff-mi300-gpt-oss-120b"
export OUTDIR=results/cliff
mkdir -p ${OUTDIR}

export NAME=lmcache-gpu-tp1-isl20k-${EXP}
export UNAME=`echo ${NAME} | tr '-' '_'`
rm -Rf results/sweeps/$NAME; just sweep sweep-configs/cliff/cliff-gpu.yaml ${NAME} | tee output_${UNAME}
echo "UNAME=${UNAME}"
just results-aggregate ${NAME} -o ${OUTDIR}/${UNAME}.json 

export NAME=lmcache-gpu-cpu-tp1-isl20k-${EXP}
export UNAME=`echo ${NAME} | tr '-' '_'`
rm -Rf results/sweeps/$NAME; just sweep sweep-configs/cliff/cliff-gpu-cpu.yaml ${NAME} | tee output_${UNAME}
echo "UNAME=${UNAME}"
just results-aggregate ${NAME} -o ${OUTDIR}/${UNAME}.json 

export NAME=lmcache-gpu-aic-tp1-isl20k-${EXP}
export UNAME=`echo ${NAME} | tr '-' '_'`
rm -Rf results/sweeps/$NAME; just sweep sweep-configs/cliff/cliff-gpu-aic.yaml ${NAME} | tee output_${UNAME}
just results-aggregate ${NAME} -o ${OUTDIR}/${UNAME}.json 

echo "Results dir: ${OUTDIR}"

# Assuming we are in `rocm-aic/recipies/llm-d/benchmarks`
# Assuming OUTDIR matches the plot-config
# Warning! requires python packages to output plots
export PYTHONPATH=$PWD/scripts;
python -m plots.plot_config ./sweep-configs/cliff/plot-config-cliff-gpu-cpu-aic.yaml
