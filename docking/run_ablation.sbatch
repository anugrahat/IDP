#!/bin/bash
#SBATCH --job-name=dock_ablation
#SBATCH --account=ahnlab
#SBATCH --partition=gpu-ahn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100_80:0
#SBATCH --time=00:30:00
#SBATCH --output=/home/anugraha/IDP/docking/logs/ablation-%j.out
#SBATCH --error=/home/anugraha/IDP/docking/logs/ablation-%j.err

set -x
cd /home/anugraha/IDP/docking
mkdir -p logs

source /software/conda3/4.X/etc/profile.d/conda.sh
conda activate openmm_env

export ABLATION_WORKERS=32     # 32 parallel Vina × 2 CPU each = 64 CPUs

python3 ablation_parallel.py
echo "[done] $(date)"
