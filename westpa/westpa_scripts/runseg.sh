#!/bin/bash
# WESTPA runseg.sh
# Propagator: Amber pmemd.cuda (single-GPU, shared via MPS on multi-walker node)
# Each invocation is one walker for one iteration.

set -e

# --- 0) Set up paths and load env (env.sh is already sourced by master but workers need it too)
source "$WEST_SIM_ROOT/env.sh"

# --- 1) Move into this segment's working dir
mkdir -p "$WEST_CURRENT_SEG_DATA_REF"
cd "$WEST_CURRENT_SEG_DATA_REF"

# --- 2) Stage topology + MD input
ln -sf "$WEST_SIM_ROOT/common_files/complex.prmtop" complex.prmtop
ln -sf "$WEST_SIM_ROOT/common_files/md.in"          md.in

# --- 3) Stage parent restart (basis state for first iter, prior segment rst7 thereafter)
case "$WEST_CURRENT_SEG_INITPOINT_TYPE" in
    SEG_INITPOINT_CONTINUES)
        cp "$WEST_PARENT_DATA_REF/seg.rst7"      parent.rst7 ;;
    SEG_INITPOINT_NEWTRAJ)
        cp "$WEST_PARENT_DATA_REF"               parent.rst7 ;;
    *)
        echo "Unknown WEST_CURRENT_SEG_INITPOINT_TYPE=$WEST_CURRENT_SEG_INITPOINT_TYPE" >&2
        exit 1 ;;
esac

# --- 4) Run one segment of pmemd.cuda
"$PMEMD" -O \
    -i md.in \
    -p complex.prmtop \
    -c parent.rst7 \
    -o seg.out \
    -r seg.rst7 \
    -x seg.nc \
    -inf seg.mdinfo \
  || { echo "pmemd.cuda failed for seg $WEST_CURRENT_SEG_ID iter $WEST_CURRENT_ITER" >&2; exit 1; }

# --- 5) Compute pcoord + aux data from the segment trajectory
# Concatenate parent's last frame onto seg.nc so pcoord_len matches expected (= ntwx_per_seg + 1)
python "$WEST_SIM_ROOT/westpa_scripts/calc_pcoord.py" complex.prmtop seg.nc

# --- 6) Hand pcoord + aux files to WESTPA via return paths
cp pcoord.dat                "$WEST_PCOORD_RETURN"
[ -n "${WEST_CONTACT_MAP_RETURN:-}" ]      && cp contact_map.dat       "$WEST_CONTACT_MAP_RETURN"
[ -n "${WEST_RING_DISTS_RETURN:-}" ]       && cp ring_dists.dat        "$WEST_RING_DISTS_RETURN"
[ -n "${WEST_D135_CHARGE_DIST_RETURN:-}" ] && cp d135_charge_dist.dat  "$WEST_D135_CHARGE_DIST_RETURN"

# --- 7) Cleanup huge intermediate but keep restart + traj + log
rm -f mdinfo
