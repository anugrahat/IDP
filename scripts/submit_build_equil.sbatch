#!/bin/bash
#SBATCH --job-name=asyn_fas_build_equil
#SBATCH --account=ahnlab
#SBATCH --partition=gpu-ahn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/anugraha/IDP/logs/build_equil-%j.out
#SBATCH --error=/home/anugraha/IDP/logs/build_equil-%j.err

set -euo pipefail
echo "Build+equil job ${SLURM_JOB_ID} on $(hostname) at $(date)"
nvidia-smi -L

module load amber/24
source /software/conda3/4.X/etc/profile.d/conda.sh

##############################################################################
# 1) Place fasudil ~10 Å from α-syn COM (matching paper)
##############################################################################
echo "=== Place fasudil ==="
conda activate openmm_env
python /home/anugraha/IDP/scripts/place_fasudil.py
conda deactivate

##############################################################################
# 2) Build solvated complex (tleap)
##############################################################################
echo "=== tleap: 1XQ8 + fasudil + TIP4P-D + 0.1 M NaCl ==="
cd /home/anugraha/IDP/build
rm -f complex.prmtop complex.rst7 complex.pdb leap.log
tleap -s -f /home/anugraha/IDP/scripts/build_complex.tleap | tee /home/anugraha/IDP/logs/tleap_complex.log
[ -f complex.prmtop ] && [ -f complex.rst7 ] || { echo "ERROR: tleap did not produce complex.{prmtop,rst7}"; exit 1; }

##############################################################################
# 3) Equilibrate per the paper's protocol:
#    (a) 10 kcal restrained min, (b) unrestrained min,
#    (c) NVT heat 0→150 K (50 ps, restr 10→6), (d) NVT heat 150→300 K (50 ps, restr 6→2),
#    (e) NPT 10 ns @ 300 K, Berendsen, 0.1 kcal backbone restraint
##############################################################################
cd /home/anugraha/IDP/equilibrate
PRM=/home/anugraha/IDP/build/complex.prmtop
INI=/home/anugraha/IDP/build/complex.rst7

echo "=== 3a) Min1 (10 kcal restrained) ==="
pmemd.cuda -O -i min1.in -p $PRM -c $INI -ref $INI \
  -o min1.out -r min1.rst7 -inf min1.mdinfo

echo "=== 3b) Min2 (unrestrained) ==="
pmemd.cuda -O -i min2.in -p $PRM -c min1.rst7 \
  -o min2.out -r min2.rst7 -inf min2.mdinfo

echo "=== 3c) Heat1 (0→150 K, 50 ps) ==="
pmemd.cuda -O -i heat1.in -p $PRM -c min2.rst7 -ref min2.rst7 \
  -o heat1.out -r heat1.rst7 -x heat1.nc -inf heat1.mdinfo

echo "=== 3d) Heat2 (150→300 K, 50 ps) ==="
pmemd.cuda -O -i heat2.in -p $PRM -c heat1.rst7 -ref heat1.rst7 \
  -o heat2.out -r heat2.rst7 -x heat2.nc -inf heat2.mdinfo

echo "=== 3e) NPT equilibration (10 ns @ 300 K, 0.1 kcal backbone restraint, Berendsen) ==="
pmemd.cuda -O -i equil_npt.in -p $PRM -c heat2.rst7 -ref heat2.rst7 \
  -o equil_npt.out -r equil_npt.rst7 -x equil_npt.nc -inf equil_npt.mdinfo

##############################################################################
# 4) Stage WESTPA basis state + topology
##############################################################################
echo "=== Stage WESTPA basis state ==="
mkdir -p /home/anugraha/IDP/westpa/common_files /home/anugraha/IDP/westpa/bstates
cp equil_npt.rst7 /home/anugraha/IDP/westpa/bstates/bstate.rst7
cp $PRM           /home/anugraha/IDP/westpa/common_files/complex.prmtop

##############################################################################
# 5) Sanity-check the equilibration trajectory
##############################################################################
echo "=== sanity check: Rg of α-syn over the NPT trajectory ==="
cat > rg_check.cpptraj <<EOF
parm $PRM
trajin equil_npt.nc
radgyr :1-140@CA out rg_asyn_equil.dat mass
distance :FAS :125-140&!@H= out lig_cterm_dist.dat
go
EOF
cpptraj -i rg_check.cpptraj > rg_check.log 2>&1
echo "Last 5 lines of α-syn Rg (CA):"
tail -5 rg_asyn_equil.dat
echo
echo "Last 5 lines of fasudil ↔ C-term min distance:"
tail -5 lig_cterm_dist.dat

echo "Build+equil complete at $(date)"
