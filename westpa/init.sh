#!/bin/bash
# WIPES west.h5, traj_segs/, seg_logs/, istates/ — ONLY USE FOR A FRESH RUN.
#
# To RESTART/RESUME a stuck WE run, DO NOT re-init.  Instead just run:
#   sbatch run_WE.sbatch
# WESTPA picks up from the last completed iteration in west.h5 automatically.
# You only lose the in-progress (stuck) iteration, not all prior work.
#
# This safeguard is fail-CLOSED: if west.h5 exists and we can't prove it's
# brand-new (≤1 iter), we refuse to wipe.  Bypass with FORCE_INIT=1.

set -e
source ./env.sh

if [ -f west.h5 ] && [ "${FORCE_INIT:-0}" != "1" ]; then
    # CRITICAL: a WE master may have west.h5 locked. h5py read will fail with
    # BlockingIOError. Use file-size as a coarse proxy: a fresh-init west.h5 is
    # ~30-40 KB. Anything larger means iteration data is present.
    size=$(stat -c%s west.h5)
    if [ "$size" -gt 60000 ]; then
        echo
        echo "REFUSING TO WIPE west.h5 (size $size bytes — looks like an active run)."
        echo "If you really want to start over, run:    FORCE_INIT=1 bash init.sh"
        echo "If you want to RESTART a stuck run, DON'T re-init — just resubmit:"
        echo "    sbatch run_WE.sbatch"
        echo "WESTPA resumes from the last completed iteration automatically."
        exit 1
    fi
    # Also try the lighter h5py read; if it succeeds AND has >1 iter, refuse.
    n_iters=$(python3 -c "
import h5py, sys
try:
    f = h5py.File('west.h5','r', swmr=True)
    print(len(f['iterations'].keys()))
except Exception:
    print('LOCKED', file=sys.stderr)
    print(-1)
" 2>/dev/null)
    if [ "$n_iters" = "-1" ]; then
        echo
        echo "REFUSING TO WIPE west.h5 — it is locked (another WESTPA master is running)."
        echo "Stop the running job first, or bypass with FORCE_INIT=1."
        exit 1
    fi
    if [ "$n_iters" -gt 1 ] 2>/dev/null; then
        echo
        echo "REFUSING TO WIPE west.h5 — it has $n_iters iterations of progress."
        echo "If you really want to start over (discard $n_iters iters of WE data):"
        echo "    FORCE_INIT=1 bash init.sh"
        echo "Otherwise just: sbatch run_WE.sbatch (resumes from west.h5)."
        exit 1
    fi
fi

rm -rf traj_segs seg_logs istates west.h5
mkdir  seg_logs traj_segs istates

# Phase 1: equilibrium WE (no recycling) — just sample binding/unbinding.
w_init \
    --bstate-file "$WEST_SIM_ROOT/bstates/bstates.txt" \
    --segs-per-state 48 \
    --work-manager=threads "$@"
