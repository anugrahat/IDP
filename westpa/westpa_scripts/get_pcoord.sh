#!/bin/bash
# Compute pcoord for a basis/initial state (no propagation, just analysis).
set -e
source "$WEST_SIM_ROOT/env.sh"

cd "$(dirname "$WEST_STRUCT_DATA_REF")"
PRM="$WEST_SIM_ROOT/common_files/complex.prmtop"

# Convert rst7 → single-frame nc via cpptraj so calc_pcoord works the same way
TMPDIR_LOCAL=$(mktemp -d)
cat > "$TMPDIR_LOCAL/conv.cpptraj" <<EOF
parm $PRM
trajin $WEST_STRUCT_DATA_REF
trajout $TMPDIR_LOCAL/single.nc netcdf
go
EOF
"$CPPTRAJ" -i "$TMPDIR_LOCAL/conv.cpptraj" > "$TMPDIR_LOCAL/conv.log" 2>&1

python "$WEST_SIM_ROOT/westpa_scripts/calc_pcoord.py" "$PRM" "$TMPDIR_LOCAL/single.nc"

# Only the first column-pair (pcoord) goes to WESTPA at the basis-state stage.
cp pcoord.dat "$WEST_PCOORD_RETURN"
rm -rf "$TMPDIR_LOCAL"
