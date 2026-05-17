#!/bin/bash
#SBATCH --job-name=md_cand4
#SBATCH --account=ahnlab
#SBATCH --partition=gpu-ahn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=06:00:00
#SBATCH --output=/home/anugraha/IDP/md_cand4/md-%j.out
#SBATCH --error=/home/anugraha/IDP/md_cand4/md-%j.err

set -x
cd /home/anugraha/IDP/md_cand4
module load amber/24
export PMEMD=$(command -v pmemd.cuda || echo /software/amber/24/ucdhpc-20.04/pmemd24/bin/pmemd.cuda)
echo "using PMEMD=$PMEMD"
nvidia-smi -L
TOP=complex_cand4.prmtop
INI=complex_cand4.rst7

# Minimize stage 1 (restraints on protein+ligand)
$PMEMD -O -i min1.in -p $TOP -c $INI -ref $INI -o min1.out -r min1.rst7 -inf min1.mdinfo

# Minimize stage 2 (no restraints)
$PMEMD -O -i min2.in -p $TOP -c min1.rst7 -o min2.out -r min2.rst7 -inf min2.mdinfo

# Heat 0 → 300 K NVT (50 ps)
$PMEMD -O -i heat.in -p $TOP -c min2.rst7 -ref min2.rst7 -o heat.out -r heat.rst7 -x heat.nc -inf heat.mdinfo

# NPT equilibration (100 ps)
$PMEMD -O -i equil_npt.in -p $TOP -c heat.rst7 -ref heat.rst7 -o equil.out -r equil.rst7 -x equil.nc -inf equil.mdinfo

# Production NPT (10 ns)
$PMEMD -O -i prod.in -p $TOP -c equil.rst7 -o prod.out -r prod.rst7 -x prod.nc -inf prod.mdinfo

echo "[done] $(date)"
