#!/bin/bash
# Per-node bring-up for WESTPA + MPS.
# Adapted from anugrahat/ParGamD-in-OpenMM-MPS (TACC variant) for UCD HPC.
set -x
umask g+r

RUN_DIR="$1"; shift
cd "$RUN_DIR" || exit 1
export WEST_JOBID="$1"; shift
export SLURM_NODENAME="$1"; shift
export CUDA_VISIBLE_DEVICES_ALLOCATED="$1"; shift

source env.sh

##############################################################################
# MPS: one nvidia-cuda-mps-control -d daemon per GPU, with isolated pipe dirs
##############################################################################
IFS=',' read -ra NODE_GPUS <<< "$CUDA_VISIBLE_DEVICES_ALLOCATED"
for gpuid in "${NODE_GPUS[@]}"; do
    export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-$WEST_JOBID-$gpuid"
    export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log-$WEST_JOBID-$gpuid"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    CUDA_VISIBLE_DEVICES="$gpuid" nvidia-cuda-mps-control -d
done

export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_ALLOCATED"
echo "[node.sh] $SLURM_NODENAME using GPUs $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

##############################################################################
# Launch w_run worker; it inherits CUDA_VISIBLE_DEVICES + MPS env
##############################################################################
w_run "$@" &> "west-${WEST_JOBID}-node-${SLURM_NODENAME}.log"

##############################################################################
# Cleanup MPS daemons on exit
##############################################################################
for gpuid in "${NODE_GPUS[@]}"; do
    export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-$WEST_JOBID-$gpuid"
    echo quit | nvidia-cuda-mps-control || true
done

exit $?
