#!/bin/bash
#SBATCH --job-name=asyn_fas_equil
#SBATCH --account=ahnlab
#SBATCH --partition=gpu-ahn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/anugraha/IDP/logs/equil-%j.out
#SBATCH --error=/home/anugraha/IDP/logs/equil-%j.err

set -euo pipefail
echo "Equil job ${SLURM_JOB_ID} on $(hostname) at $(date)"
nvidia-smi -L

module load amber/24
cd /home/anugraha/IDP/equilibrate
PRM=/home/anugraha/IDP/build/complex.prmtop
INI=/home/anugraha/IDP/build/complex.rst7

[ -f "$PRM" ] && [ -f "$INI" ] || { echo "ERROR: complex.prmtop/rst7 missing in build/"; exit 1; }

echo "=== 1) Min1 (10 kcal heavy-atom restrained) ==="
pmemd.cuda -O -i min1.in -p $PRM -c $INI -ref $INI \
  -o min1.out -r min1.rst7 -inf min1.mdinfo

echo "=== 2) Min2 (unrestrained) ==="
pmemd.cuda -O -i min2.in -p $PRM -c min1.rst7 \
  -o min2.out -r min2.rst7 -inf min2.mdinfo

echo "=== 3) Heat1 (NVT 0→150 K, 50 ps) ==="
pmemd.cuda -O -i heat1.in -p $PRM -c min2.rst7 -ref min2.rst7 \
  -o heat1.out -r heat1.rst7 -x heat1.nc -inf heat1.mdinfo

echo "=== 4) Heat2 (NVT 150→300 K, 50 ps) ==="
pmemd.cuda -O -i heat2.in -p $PRM -c heat1.rst7 -ref heat1.rst7 \
  -o heat2.out -r heat2.rst7 -x heat2.nc -inf heat2.mdinfo

echo "=== 5) NPT 10 ns @ 300 K, 0.1 kcal backbone restraint, Berendsen ==="
pmemd.cuda -O -i equil_npt.in -p $PRM -c heat2.rst7 -ref heat2.rst7 \
  -o equil_npt.out -r equil_npt.rst7 -x equil_npt.nc -inf equil_npt.mdinfo

echo "=== Stage WESTPA basis state ==="
mkdir -p /home/anugraha/IDP/westpa/common_files /home/anugraha/IDP/westpa/bstates
cp equil_npt.rst7 /home/anugraha/IDP/westpa/bstates/bstate.rst7
cp $PRM           /home/anugraha/IDP/westpa/common_files/complex.prmtop

echo "=== Sanity-check ==="
cat > rg_check.cpptraj <<EOF
parm $PRM
trajin equil_npt.nc
radgyr :1-140@CA out rg_asyn_equil.dat mass
nativecontacts :FAS :125-140 distance 5.0 out lig_cterm_contacts.dat
go
EOF
cpptraj -i rg_check.cpptraj > rg_check.log 2>&1
echo "Final 5 Rg values:"
tail -5 rg_asyn_equil.dat
echo "Equil complete at $(date)"
