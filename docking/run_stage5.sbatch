#!/bin/bash
#SBATCH --job-name=stage5
#SBATCH --account=ahnlab
#SBATCH --partition=gpu-ahn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100_80:0
#SBATCH --time=00:30:00
#SBATCH --output=/home/anugraha/IDP/docking/logs/stage5-%j.out
#SBATCH --error=/home/anugraha/IDP/docking/logs/stage5-%j.err

set -x
cd /home/anugraha/IDP/docking
mkdir -p logs

source /software/conda3/4.X/etc/profile.d/conda.sh
conda activate openmm_env

export STAGE5_WORKERS=32

python3 stage5_full_ablation.py
echo "[done] $(date)"
