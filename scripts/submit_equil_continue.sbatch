#!/bin/bash
#SBATCH --job-name=asyn_fas_equil_cont
#SBATCH --account=ahnlab
#SBATCH --partition=gpu-ahn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/anugraha/IDP/logs/equil_cont-%j.out
#SBATCH --error=/home/anugraha/IDP/logs/equil_cont-%j.err

# Resumes from equil_npt.rst7 (which we already have at TIME=500 ps).
# Runs NPT in 250-ps chunks, each restart rebuilds GPU neighbor grid so box-shrink
# doesn't trip pmemd.cuda's "box dimensions changed too much" check.
# We do 20 chunks × 250 ps = 5 ns additional NPT. Then total NPT = 5.5 ns ≈ paper's protocol.

set -euo pipefail
echo "Equil-continue job ${SLURM_JOB_ID} on $(hostname) at $(date)"
nvidia-smi -L

module load amber/24
cd /home/anugraha/IDP/equilibrate

PRM=/home/anugraha/IDP/build/complex.prmtop
IN=equil_npt_chunk.in
[ -f "$PRM" ] || { echo "missing prmtop"; exit 1; }
[ -f equil_npt.rst7 ] || { echo "missing starting rst7"; exit 1; }

# move the failed long-NPT artefacts aside (keep for forensics)
mkdir -p chunks
mv equil_npt.nc   chunks/equil_npt_failed.nc 2>/dev/null || true
mv equil_npt.out  chunks/equil_npt_failed.out 2>/dev/null || true
mv equil_npt.mdinfo chunks/equil_npt_failed.mdinfo 2>/dev/null || true

CUR=equil_npt.rst7    # 500 ps in
N_CHUNKS=20            # 20 × 250 ps = 5 ns more

for ((k=1; k<=N_CHUNKS; k++)); do
    KID=$(printf "%02d" $k)
    OUT=chunks/npt_$KID
    echo "=== chunk $k of $N_CHUNKS — restart from $CUR ==="
    pmemd.cuda -O \
        -i $IN -p $PRM -c $CUR -ref $CUR \
        -o ${OUT}.out -r ${OUT}.rst7 -x ${OUT}.nc -inf ${OUT}.mdinfo \
        || { echo "ERROR: chunk $k failed — last rst7: $CUR"; exit 1; }
    # Print density + volume per chunk for monitoring
    grep -E "Density|VOLUME" ${OUT}.out 2>&1 | tail -2
    CUR=${OUT}.rst7
done

echo "=== final state: $CUR ==="

##############################################################################
# Stage WESTPA basis state + topology
##############################################################################
echo "=== Stage WESTPA basis state ==="
mkdir -p /home/anugraha/IDP/westpa/common_files /home/anugraha/IDP/westpa/bstates
cp $CUR /home/anugraha/IDP/westpa/bstates/bstate.rst7
cp $PRM /home/anugraha/IDP/westpa/common_files/complex.prmtop

##############################################################################
# Sanity check: Rg of α-syn + lig-Cterm distance over the cascaded NPT
##############################################################################
echo "=== sanity check ==="
# concatenate the chunk trajectories
cat > /tmp/concat.cpptraj <<EOF
parm $PRM
$(for ((k=1; k<=N_CHUNKS; k++)); do printf "trajin chunks/npt_%02d.nc\n" $k; done)
radgyr :1-140@CA out rg_asyn_equil.dat mass
distance lig_cterm :FAS  :125-140&!@H= out lig_cterm_min_dist.dat
go
EOF
cpptraj -i /tmp/concat.cpptraj > /tmp/concat.log 2>&1
echo "Final Rg:"
tail -5 rg_asyn_equil.dat
echo
echo "Final fasudil → C-term distance:"
tail -5 lig_cterm_min_dist.dat
echo "Equil-continue complete at $(date)"
