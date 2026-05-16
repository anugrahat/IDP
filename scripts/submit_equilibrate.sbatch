#!/bin/bash
#SBATCH --job-name=asyn_fas_equil
#SBATCH --account=ahnlab
#SBATCH --partition=gpu-ahn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/anugraha/IDP/logs/equil-%j.out
#SBATCH --error=/home/anugraha/IDP/logs/equil-%j.err

set -euo pipefail
echo "Equil job ${SLURM_JOB_ID} on $(hostname) at $(date)"
nvidia-smi -L

module load amber/24
cd /home/anugraha/IDP/equilibrate

PRM=/home/anugraha/IDP/build/complex.prmtop
INI=/home/anugraha/IDP/build/complex.rst7

echo "=== 1) Explicit-solvent minimization (heavy-atom restraints) ==="
pmemd.cuda -O \
  -i min_explicit.in -p $PRM -c $INI -ref $INI \
  -o min.out -r min.rst7 -inf min.mdinfo

echo "=== 2) NVT heating 100→300 K (100 ps, restraints) ==="
pmemd.cuda -O \
  -i heat_nvt.in -p $PRM -c min.rst7 -ref min.rst7 \
  -o heat.out -r heat.rst7 -x heat.nc -inf heat.mdinfo

echo "=== 3) NPT restrained density equilibration (500 ps, CA restraints) ==="
pmemd.cuda -O \
  -i equil_npt_restrained.in -p $PRM -c heat.rst7 -ref heat.rst7 \
  -o equil_npt_r.out -r equil_npt_r.rst7 -x equil_npt_r.nc -inf equil_npt_r.mdinfo

echo "=== 4) NPT free equilibration (5 ns, no restraints) ==="
pmemd.cuda -O \
  -i equil_npt_free.in -p $PRM -c equil_npt_r.rst7 \
  -o equil_npt_f.out -r equil_npt_f.rst7 -x equil_npt_f.nc -inf equil_npt_f.mdinfo

echo "=== 5) Stage WESTPA basis state ==="
cp equil_npt_f.rst7 /home/anugraha/IDP/westpa/bstates/bstate.rst7
cp $PRM            /home/anugraha/IDP/westpa/common_files/complex.prmtop

echo "Equilibration complete at $(date). bstate.rst7 staged."
