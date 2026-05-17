#!/bin/bash
#SBATCH --job-name=pca_landscape
#SBATCH --account=ahnlab
#SBATCH --partition=gpu-ahn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100_80:0
#SBATCH --time=00:30:00
#SBATCH --output=/home/anugraha/IDP/conformational_ensemble/pca-%j.out
#SBATCH --error=/home/anugraha/IDP/conformational_ensemble/pca-%j.err

set -x
cd /home/anugraha/IDP/conformational_ensemble
source /software/conda3/4.X/etc/profile.d/conda.sh
conda activate openmm_env
export PCA_WORKERS=32
python3 -u pca_analysis_parallel.py
echo "[done] $(date)"
