#!/bin/bash
#SBATCH --job-name=plif_v4
#SBATCH --account=ahnlab
#SBATCH --partition=gpu-ahn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100_80:0
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/anugraha/IDP/docking/logs/plif-v4-%j.out
#SBATCH --error=/home/anugraha/IDP/docking/logs/plif-v4-%j.err

set -x
cd /home/anugraha/IDP/docking
source /software/conda3/4.X/etc/profile.d/conda.sh
conda activate openmm_env
# Disable inner-library threading at SLURM level too
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PLIF_WORKERS=32
python3 -u stage5_v2_plif_rescue_v4.py
echo "[done] $(date)"
