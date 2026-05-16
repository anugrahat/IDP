#!/bin/bash
# Sanity-test the WESTPA scaffold once bstate.rst7 + complex.prmtop are staged.
# Run this on a GPU node (login node lacks pmemd.cuda):
#   srun -p gpu-ahn -A ahnlab --gres=gpu:a100_80:1 -t 0:30:00 --pty bash test_basis_state.sh
set -e

source /home/anugraha/IDP/westpa/env.sh
cd /home/anugraha/IDP/westpa

##############################################################################
# 1) Check inputs
##############################################################################
for f in bstates/bstate.rst7 common_files/complex.prmtop common_files/md.in; do
    [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
done
echo "[ok] bstate + topology + md.in present"

##############################################################################
# 2) Compute pcoord on the basis state via get_pcoord.sh
##############################################################################
export WEST_STRUCT_DATA_REF=$WEST_SIM_ROOT/bstates/bstate.rst7
export WEST_PCOORD_RETURN=/tmp/test_pcoord_bstate_$$.dat
bash westpa_scripts/get_pcoord.sh
echo "[ok] get_pcoord.sh ran; basis-state pcoord:"
cat "$WEST_PCOORD_RETURN"
echo

##############################################################################
# 3) Run one test segment (100 ps pmemd.cuda) to validate runseg.sh
##############################################################################
TEST=/tmp/westpa_test_seg_$$
mkdir -p "$TEST"
cd "$TEST"
ln -sf $WEST_SIM_ROOT/common_files/complex.prmtop complex.prmtop
ln -sf $WEST_SIM_ROOT/common_files/md.in          md.in
cp $WEST_SIM_ROOT/bstates/bstate.rst7             parent.rst7

echo "Running 100 ps test segment with pmemd.cuda..."
"$PMEMD" -O \
    -i md.in -p complex.prmtop -c parent.rst7 \
    -o seg.out -r seg.rst7 -x seg.nc -inf seg.mdinfo

python $WEST_SIM_ROOT/westpa_scripts/calc_pcoord.py complex.prmtop seg.nc
echo "[ok] segment ran; pcoord trace:"
head -5 pcoord.dat
echo "..."
tail -3 pcoord.dat
echo "lines in pcoord.dat: $(wc -l < pcoord.dat)  (expected: 100 frames + maybe header)"
echo "aux files: $(ls -la *.dat)"
cd -

##############################################################################
# 4) Try w_init (tests west.cfg validity)
##############################################################################
echo "Running w_init dry-run (will be redone for real production launch)..."
( cd $WEST_SIM_ROOT && bash init.sh ) || { echo "w_init failed"; exit 1; }
echo "[ok] w_init succeeded; west.h5 created"
ls -la $WEST_SIM_ROOT/west.h5
echo "Iteration 1 segments:"
python -c "
import h5py
f = h5py.File('$WEST_SIM_ROOT/west.h5','r')
print('iters:', list(f['iterations'].keys())[:5])
print('iter_00000001 seg pcoords shape:', f['iterations/iter_00000001/pcoord'].shape)
"

echo
echo "[OK] WESTPA scaffold validated end-to-end."
echo "Clean state for production launch:"
echo "  cd $WEST_SIM_ROOT && rm -rf traj_segs seg_logs istates west.h5"
echo "Then: sbatch run_WE.sbatch"
rm -rf "$TEST"
