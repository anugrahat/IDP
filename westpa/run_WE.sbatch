#!/bin/bash
#SBATCH --job-name=asyn_fas_WE
#SBATCH --account=ahnlab
#SBATCH --partition=gpu-ahn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100_80:4
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/anugraha/IDP/westpa/logs/we-%j.out
#SBATCH --error=/home/anugraha/IDP/westpa/logs/we-%j.err

set -x
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source env.sh
export WEST_SIM_ROOT="$SLURM_SUBMIT_DIR"
export WORKERS_PER_GPU=12
export OMP_NUM_THREADS=1
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=70

IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
echo "Got GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

############################################################
# MPS: ONE daemon for all GPUs (the working ParGamD pattern).
# All workers point to the same CUDA_MPS_PIPE_DIRECTORY and
# pass physical GPU index via CUDA_VISIBLE_DEVICES.
############################################################
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

# Restore all GPUs visible for the daemon
export CUDA_VISIBLE_DEVICES="$(IFS=','; echo "${DEVICES[*]}")"
nvidia-cuda-mps-control -d
sleep 5
echo set_active_thread_percentage 70 | nvidia-cuda-mps-control

# WESTPA master + workers via ZMQ
export SERVER_INFO="$WEST_SIM_ROOT/west_zmq_info.json"
rm -f "$SERVER_INFO"

w_run --work-manager=zmq --n-workers=0 --zmq-mode=master \
      --zmq-write-host-info="$SERVER_INFO" --zmq-comm-mode=ipc \
      &> "logs/west-${SLURM_JOB_ID}-master.log" &

for ((n=0; n<60; n++)); do
    [ -e "$SERVER_INFO" ] && { cat "$SERVER_INFO"; break; }
    sleep 1
done
[ -e "$SERVER_INFO" ] || { echo "ZMQ master failed to start"; exit 1; }

# Spawn $WORKERS_PER_GPU client workers per GPU
total_workers=$(( ${#DEVICES[@]} * WORKERS_PER_GPU ))
echo "Spawning $total_workers workers (${WORKERS_PER_GPU} × ${#DEVICES[@]} GPUs)"

for gpuid in "${DEVICES[@]}"; do
    for ((w=1; w<=WORKERS_PER_GPU; w++)); do
        (
            export CUDA_VISIBLE_DEVICES="$gpuid"
            w_run --work-manager=zmq --n-workers=1 --zmq-mode=client \
                  --zmq-read-host-info="$SERVER_INFO" --zmq-comm-mode=ipc \
                  &> "logs/west-${SLURM_JOB_ID}-worker-gpu${gpuid}-w${w}.log"
        ) &
        sleep 0.15   # stagger startup to avoid IPC socket / port races
    done
done

wait

# Tear down MPS
echo quit | nvidia-cuda-mps-control || true

echo "WE run complete at $(date)"
