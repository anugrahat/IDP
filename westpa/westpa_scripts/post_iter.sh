#!/bin/bash
# Optional per-iteration cleanup: tar the iteration's intermediate prmtop links to save inodes.
set -e
source "$WEST_SIM_ROOT/env.sh"

ITER_DIR="$WEST_SIM_ROOT/traj_segs/$(printf '%06d' $WEST_CURRENT_ITER)"
if [ -d "$ITER_DIR" ]; then
    # remove symlinks to prmtop / md.in inside each seg dir (originals live in common_files)
    find "$ITER_DIR" -maxdepth 2 -type l -delete
fi
