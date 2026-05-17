#!/bin/bash
#SBATCH --job-name=c09_abla
#SBATCH --account=ahnlab
#SBATCH --partition=gpu-ahn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100_80:0
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/anugraha/IDP/docking/logs/c09-%j.out
#SBATCH --error=/home/anugraha/IDP/docking/logs/c09-%j.err

set -x
cd /home/anugraha/IDP/docking
source /software/conda3/4.X/etc/profile.d/conda.sh
conda activate openmm_env
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PLIF_WORKERS=32
python3 -u stage5_v2_c09_only.py
echo "[done] $(date)"
