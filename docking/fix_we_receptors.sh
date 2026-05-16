#!/bin/bash
# Fix WE receptor PDBs so ProLIF / MDAnalysis can load them.
# Root cause: cpptraj kept AMBER-protonated H atoms with naming that confuses
# RDKit's bond perception. Strip Hs first, then re-protonate with obabel at pH 7.4
# (the same path that works for 1XQ8).
set -euo pipefail
cd /home/anugraha/IDP/docking/receptor_ensemble
for pdb in receptor_*.pdb; do
    # Skip already-fixed ones (would have _orig.pdb backup)
    if [ -f "${pdb%.pdb}_orig.pdb" ]; then
        echo "skip ${pdb} (already fixed)"; continue
    fi
    cp "$pdb" "${pdb%.pdb}_orig.pdb"
    # strip all H atoms with grep then obabel-protonate at pH 7.4
    grep -v -E "^(ATOM|HETATM).{12}.H" "${pdb%.pdb}_orig.pdb" > "${pdb%.pdb}_noH.pdb"
    obabel "${pdb%.pdb}_noH.pdb" -O "$pdb" -h -p 7.4 2>/dev/null
    echo "fixed $pdb"
done
