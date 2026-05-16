#!/bin/bash
# WESTPA environment for IDP fasudil association
# Sourced by every shell that touches the simulation.

# --- HPC modules (UC Davis hpc2) ---
module load amber/24       # also pulls cuda/11.8.0; gives pmemd.cuda + AmberTools
module load slurm/23.02.7  # default
export PATH="$AMBERHOME/bin:$PATH"

# --- Conda env with westpa + mdanalysis ---
source /software/conda3/4.X/etc/profile.d/conda.sh
conda activate openmm_env   # has westpa 2022.11, mdanalysis 2.2.0, parmed 4.3.0

# --- Sim root ---
if [[ -z "${WEST_SIM_ROOT:-}" ]]; then
    export WEST_SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
export SIM_NAME=$(basename "$WEST_SIM_ROOT")

# --- WESTPA runtime tuning ---
export USE_LOCAL_SCRATCH=0
export WM_ZMQ_MASTER_HEARTBEAT=100
export WM_ZMQ_WORKER_HEARTBEAT=100
export WM_ZMQ_TIMEOUT_FACTOR=300

# --- Amber binaries — IMPORTANT: pmemd.cuda lives in pmemd24/, NOT in ambertools25/bin/ ---
# Use which lookup since module load already prepended pmemd24/bin to PATH.
export PMEMD="$(command -v pmemd.cuda 2>/dev/null || echo /software/amber/24/ucdhpc-20.04/pmemd24/bin/pmemd.cuda)"
export CPPTRAJ="$(command -v cpptraj 2>/dev/null || echo $AMBERHOME/bin/cpptraj)"
[ -x "$PMEMD" ]   || { echo "[env.sh] ERROR: pmemd.cuda not found at $PMEMD" >&2; }
[ -x "$CPPTRAJ" ] || { echo "[env.sh] ERROR: cpptraj not found at $CPPTRAJ" >&2; }

echo "[env.sh] WEST_SIM_ROOT=$WEST_SIM_ROOT  AMBERHOME=$AMBERHOME"
