#!/bin/bash
#SBATCH --job-name=asyn_fas_equil
#SBATCH --account=ahnlab
#SBATCH --partition=gpu-ahn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/anugraha/IDP/logs/equil_full-%j.out
#SBATCH --error=/home/anugraha/IDP/logs/equil_full-%j.err

# Full equilibration with cascaded NPT to avoid the pmemd.cuda density-crash issue.
#   min1 (10 kcal heavy restr) → min2 (unrestr) → heat1 (NVT 0→150 K) → heat2 (NVT 150→300 K)
# → NPT × 22 chunks × 250 ps = 5.5 ns total NPT (each chunk restart rebuilds GPU grid)

set -euo pipefail
echo "Equil job ${SLURM_JOB_ID} on $(hostname) at $(date)"
nvidia-smi -L

module load amber/24
cd /home/anugraha/IDP/equilibrate

PRM=/home/anugraha/IDP/build/complex.prmtop
INI=/home/anugraha/IDP/build/complex.rst7
[ -f "$PRM" ] && [ -f "$INI" ] || { echo "missing complex.prmtop/rst7"; exit 1; }

# clean any prior failed-run state but keep the input files (*.in)
rm -rf chunks
mkdir -p chunks
for stage in min1 min2 heat1 heat2 equil_npt; do
    rm -f ${stage}.out ${stage}.rst7 ${stage}.nc ${stage}.mdinfo
done

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

echo "=== 5) Cascade NPT: 25 × 200 ps = 5 ns (starting density 0.83 → 0.997 in 1-2 chunks, rest is equilibration) ==="
CUR=heat2.rst7
N_CHUNKS=25
for ((k=1; k<=N_CHUNKS; k++)); do
    KID=$(printf "%02d" $k)
    OUT=chunks/npt_$KID
    echo "--- chunk $k of $N_CHUNKS — restart from $CUR ---"
    pmemd.cuda -O -i equil_npt_chunk.in -p $PRM -c $CUR -ref $CUR \
        -o ${OUT}.out -r ${OUT}.rst7 -x ${OUT}.nc -inf ${OUT}.mdinfo \
        || { echo "ERROR: chunk $k failed — last rst7: $CUR"; exit 1; }
    grep -E "Density|VOLUME" ${OUT}.out 2>&1 | tail -2
    CUR=${OUT}.rst7
done

echo "=== Stage WESTPA basis state ==="
mkdir -p /home/anugraha/IDP/westpa/common_files /home/anugraha/IDP/westpa/bstates
cp $CUR /home/anugraha/IDP/westpa/bstates/bstate.rst7
cp $PRM /home/anugraha/IDP/westpa/common_files/complex.prmtop

echo "=== Sanity check: trajectory analyses ==="
cat > /tmp/concat.cpptraj <<EOF
parm $PRM
$(for ((k=1; k<=N_CHUNKS; k++)); do printf "trajin chunks/npt_%02d.nc\n" $k; done)
radgyr :1-140@CA out rg_asyn_equil.dat mass
distance lig_cterm :FAS  :125-140&!@H= out lig_cterm_min_dist.dat
density out density.dat
go
EOF
cpptraj -i /tmp/concat.cpptraj > /tmp/concat.log 2>&1
echo "α-syn Rg trace (last 5):"
tail -5 rg_asyn_equil.dat
echo
echo "fasudil → C-term min distance trace (last 5):"
tail -5 lig_cterm_min_dist.dat
echo "Equil complete at $(date)"
