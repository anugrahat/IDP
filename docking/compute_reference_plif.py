#!/usr/bin/env python3
"""
Compute fasudil reference PLIF from the WE bound ensemble.
Loads trajectories directly with prmtop (gives ProLIF proper bond info).
"""
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import MDAnalysis as mda
import prolif as plf

WE_ROOT = Path("/home/anugraha/IDP/westpa")
TOPOLOGY = "/home/anugraha/IDP/build/complex.prmtop"
ENSEMBLE_DIR = Path("/home/anugraha/IDP/docking/receptor_ensemble")
OUT_DIR = Path("/home/anugraha/IDP/docking/plif")
OUT_DIR.mkdir(exist_ok=True, parents=True)

ALL_INTERACTIONS = ["Hydrophobic", "HBDonor", "HBAcceptor", "PiStacking",
                    "PiCation", "CationPi", "Anionic", "Cationic", "VdWContact"]

manifest = pd.read_csv(ENSEMBLE_DIR / "ensemble_manifest.csv")
print(f"[1/2] running ProLIF on {len(manifest)} bound frames (loaded from prmtop+nc)...")

# Aggregate interaction frequencies across all bound frames, weighted by cluster size
totals = {}     # key: f"{resname}{resid}|{itype}" → cumulative weight
n_frames_run = 0
for _, row in manifest.iterrows():
    traj = str(WE_ROOT / f"traj_segs/{int(row['iter']):06d}/{int(row['seg']):06d}/seg.nc")
    frame_idx = int(row["frame"])
    w = float(row["weight"])
    u = mda.Universe(TOPOLOGY, traj)
    if frame_idx >= len(u.trajectory):
        print(f"  iter={row['iter']} seg={row['seg']} f={frame_idx} out of range ({len(u.trajectory)} frames), skip")
        continue
    u.trajectory[frame_idx]
    # Select protein and fasudil — but we need to strip waters since they're irrelevant for PLIF here
    lig = u.select_atoms("resname FAS")
    prot = u.select_atoms("protein")
    if len(lig) == 0 or len(prot) == 0:
        print(f"  bound_{int(row.name)+1}: missing lig({len(lig)}) or prot({len(prot)})"); continue

    lig_mol = plf.Molecule.from_mda(lig)
    prot_mol = plf.Molecule.from_mda(prot)
    fp = plf.Fingerprint(ALL_INTERACTIONS)
    fp.run_from_iterable([lig_mol], prot_mol, progress=False)
    df = fp.to_dataframe()
    if df.empty or df.shape[0] == 0:
        print(f"  iter={row['iter']} seg={row['seg']}: no interactions")
        continue

    inter_this_frame = set()
    for col in df.columns:
        if df[col].iloc[0]:
            lig_name, prot_res, itype = col
            key = f"{prot_res}|{itype}"
            inter_this_frame.add(key)
    for k in inter_this_frame:
        totals[k] = totals.get(k, 0.0) + w
    n_frames_run += 1
    print(f"  frame {n_frames_run}/{len(manifest)}  iter={row['iter']} seg={row['seg']}  w={w:.3f}  {len(inter_this_frame)} interactions")

print(f"\n[2/2] aggregated {len(totals)} unique (residue,interaction) pairs")
sorted_inter = sorted(totals.items(), key=lambda x: -x[1])
print(f"\n  top-25 interactions (by weighted frequency):")
print(f"  {'residue|type':<35} {'freq':>9}")
for key, freq in sorted_inter[:25]:
    print(f"  {key:<35} {freq:>9.3f}")

with open(OUT_DIR / "reference_plif.json", "w") as f:
    json.dump(totals, f, indent=2)
print(f"\n  reference PLIF saved: {OUT_DIR}/reference_plif.json")
