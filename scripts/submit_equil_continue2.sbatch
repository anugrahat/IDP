#!/bin/bash
#SBATCH --job-name=asyn_fas_npt_polish
#SBATCH --account=ahnlab
#SBATCH --partition=gpu-ahn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/anugraha/IDP/logs/npt_polish-%j.out
#SBATCH --error=/home/anugraha/IDP/logs/npt_polish-%j.err

# Continue from npt_01.rst7 (density 0.9825) — system is at near-equilibrium now,
# so 250 ps chunks should run cleanly (box hardly changes any more).
set -euo pipefail
echo "Polish job ${SLURM_JOB_ID} on $(hostname) at $(date)"
nvidia-smi -L

module load amber/24
cd /home/anugraha/IDP/equilibrate

PRM=/home/anugraha/IDP/build/complex.prmtop
[ -f "$PRM" ] && [ -f chunks/npt_01.rst7 ] || { echo "missing inputs"; exit 1; }

CUR=chunks/npt_01.rst7
N_CHUNKS=20      # 20 × 250 ps = 5 ns additional NPT polish
START=02         # we already did chunk 01

# Build a longer 250 ps chunk spec
cat > equil_npt_polish.in <<'EOF'
NPT polish chunk: 250 ps, 300 K, 0.1 kcal backbone restraint, MC barostat
 &cntrl
  imin   = 0,
  irest  = 1, ntx = 5,
  nstlim = 125000, dt = 0.002,
  ntc    = 2, ntf = 2,
  cut    = 10.0,
  ntb    = 2,
  ntp    = 1, barostat = 2, pres0 = 1.0, mcbarint = 100,
  ntt    = 3, gamma_ln = 2.0, temp0 = 300.0,
  ntr    = 1,
  restraint_wt = 0.1,
  restraintmask = "(:1-140) & @N,CA,C,O",
  ig     = -1,
  ntpr   = 2500, ntwx = 2500, ntwr = 25000,
  iwrap  = 1, ioutfm = 1,
 /
EOF

for ((k=2; k<=N_CHUNKS+1; k++)); do
    KID=$(printf "%02d" $k)
    OUT=chunks/npt_$KID
    echo "--- chunk $k restart from $CUR ---"
    pmemd.cuda -O -i equil_npt_polish.in -p $PRM -c $CUR -ref $CUR \
        -o ${OUT}.out -r ${OUT}.rst7 -x ${OUT}.nc -inf ${OUT}.mdinfo \
        || { echo "chunk $k failed — last good: $CUR"; break; }
    grep -E "Density|VOLUME" ${OUT}.out 2>&1 | tail -2
    CUR=${OUT}.rst7
done

echo "=== final rst7: $CUR ==="
cp $CUR /home/anugraha/IDP/westpa/bstates/bstate.rst7
echo "Polish complete at $(date) — bstate.rst7 updated to final"
