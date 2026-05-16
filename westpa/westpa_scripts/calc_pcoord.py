#!/usr/bin/env python3
"""
Compute pcoord + aux datasets for a single WESTPA segment.

pcoord columns (used for binning):
    0  min heavy-atom distance from fasudil heavy atoms to α-syn residues 125-140 heavy atoms (Å)
    1  Rg of α-syn CA atoms (Å)

Aux datasets (stored, not used for binning):
    contact_map      (n_frames, 140)  per-residue min heavy-atom distance from fasudil to that residue
    ring_dists       (n_frames, 3)    fasudil isoquinoline centroid distance to ring centroids of Y125, Y133, Y136
    d135_charge_dist (n_frames,)      fasudil cationic-N to D135 carboxylate-O minimum distance

Invocation:
    calc_pcoord.py <topology.parm7> <trajectory.nc>  [--reference <ref.pdb>]
Writes:
    pcoord.dat                 (n_frames lines, 2 cols)
    contact_map.dat            (n_frames lines, 140 cols)
    ring_dists.dat             (n_frames lines, 3 cols)
    d135_charge_dist.dat       (n_frames lines, 1 col)
"""
import argparse, sys, numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array

# Residue numbering matches Amber: contiguous 1..140, no chain breaks.
CTERM_RESIDS  = list(range(125, 141))  # 125..140 inclusive
TYR_RESIDS    = [125, 133, 136]
ASP135_RESID  = 135

LIGAND_RESN   = "FAS"

def select_ligand(u):
    sel = u.select_atoms(f"resname {LIGAND_RESN} and not name H*")
    if len(sel) == 0:
        raise SystemExit(f"no atoms with resname {LIGAND_RESN} in topology")
    return sel

def select_ligand_isoquinoline_centroid(u):
    # Isoquinoline aromatic carbons + pyridine N; GAFF2 atom names from antechamber output are C6..C14, N3
    sel = u.select_atoms(f"resname {LIGAND_RESN} and (name C6 C7 C8 C9 C10 C11 C12 C13 C14 N3)")
    return sel

def select_ligand_cation_N(u):
    # The protonated diazepane N — antechamber labelled it N1 (atom index 3 in mol2).
    sel = u.select_atoms(f"resname {LIGAND_RESN} and name N1")
    if len(sel) == 0:
        sel = u.select_atoms(f"resname {LIGAND_RESN} and type ny")
    return sel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topology")
    ap.add_argument("trajectory")
    args = ap.parse_args()

    u = mda.Universe(args.topology, args.trajectory)

    lig_heavy = select_ligand(u)
    lig_ring  = select_ligand_isoquinoline_centroid(u)
    lig_cat_N = select_ligand_cation_N(u)

    cterm_heavy = u.select_atoms(
        "protein and not name H* and (resid " + " or resid ".join(str(r) for r in CTERM_RESIDS) + ")"
    )
    asyn_ca = u.select_atoms("protein and name CA")

    tyr_rings = {
        r: u.select_atoms(f"protein and resid {r} and name CG CD1 CD2 CE1 CE2 CZ")
        for r in TYR_RESIDS
    }
    d135_oxy = u.select_atoms(f"protein and resid {ASP135_RESID} and (name OD1 OD2)")

    n_frames = len(u.trajectory)
    pcoord = np.zeros((n_frames, 2), dtype=np.float32)
    contact_map = np.full((n_frames, 140), np.nan, dtype=np.float32)
    ring_dists = np.zeros((n_frames, 3), dtype=np.float32)
    d135_dist = np.zeros((n_frames,), dtype=np.float32)

    # Per-residue heavy atom groups (precompute selections)
    per_res_heavy = [u.select_atoms(f"protein and not name H* and resid {r}") for r in range(1, 141)]

    for i, ts in enumerate(u.trajectory):
        # Min distance ligand → C-term residues 125-140
        d = distance_array(lig_heavy.positions, cterm_heavy.positions)
        pcoord[i, 0] = float(d.min())
        # Rg of α-syn CA
        pcoord[i, 1] = float(asyn_ca.radius_of_gyration())

        # Per-residue min distance (contact_map aux)
        for k, group in enumerate(per_res_heavy):
            if len(group) == 0:
                continue
            d_res = distance_array(lig_heavy.positions, group.positions)
            contact_map[i, k] = float(d_res.min())

        # Aromatic stacking distances (Y125, Y133, Y136 ring centroid ↔ fasudil isoquinoline centroid)
        if len(lig_ring) > 0:
            lig_ring_c = lig_ring.center_of_geometry()
            for j, r in enumerate(TYR_RESIDS):
                tg = tyr_rings[r]
                if len(tg) == 0:
                    ring_dists[i, j] = np.nan
                else:
                    ring_dists[i, j] = float(np.linalg.norm(lig_ring_c - tg.center_of_geometry()))

        # D135 charge-charge: fasudil cationic N → D135 OD1/OD2 min distance
        if len(lig_cat_N) > 0 and len(d135_oxy) > 0:
            d_ch = distance_array(lig_cat_N.positions, d135_oxy.positions)
            d135_dist[i] = float(d_ch.min())
        else:
            d135_dist[i] = np.nan

    np.savetxt("pcoord.dat", pcoord, fmt="%.4f")
    np.savetxt("contact_map.dat", contact_map, fmt="%.3f")
    np.savetxt("ring_dists.dat", ring_dists, fmt="%.3f")
    np.savetxt("d135_charge_dist.dat", d135_dist, fmt="%.3f")
    print(f"wrote pcoord ({pcoord.shape}), contact_map ({contact_map.shape}), ring_dists ({ring_dists.shape}), d135 ({d135_dist.shape})")

if __name__ == "__main__":
    main()
