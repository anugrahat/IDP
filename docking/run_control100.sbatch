#!/bin/bash
#SBATCH --job-name=ctrl100
#SBATCH --account=ahnlab
#SBATCH --partition=gpu-ahn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100_80:0
#SBATCH --time=00:30:00
#SBATCH --output=/home/anugraha/IDP/docking/logs/ctrl100-%j.out
#SBATCH --error=/home/anugraha/IDP/docking/logs/ctrl100-%j.err

set -x
cd /home/anugraha/IDP/docking
mkdir -p logs

source /software/conda3/4.X/etc/profile.d/conda.sh
conda activate openmm_env

export CONTROL_WORKERS=32

python3 control_100seed.py
echo "[done] $(date)"
