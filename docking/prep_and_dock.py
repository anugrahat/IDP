#!/usr/bin/env python3
"""
1XQ8-vs-WE docking control experiment for α-syn + fasudil.

Three conditions:
  A_biased  receptor = 1XQ8 raw          box = C-term region (Y133/D135/Y136)
  A_blind   receptor = 1XQ8 raw          box = whole protein
  C_biased  receptor = WE bound pose     box = C-term region

Outputs:
  receptors/                  PDB + PDBQT for each receptor
  ligand/fasudil.pdbqt        Meeko-prepared ligand
  poses/{cond}.pdbqt          Vina top-20 poses
  poses/{cond}_top{1..5}.pdb  individual pose PDBs for visualization
  summary.txt                 numerical summary across conditions
"""
import os, sys, subprocess, shutil
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina

ROOT = Path("/home/anugraha/IDP/docking")
RECEPTORS = ROOT/"receptors"
POSES = ROOT/"poses"
LIGAND_DIR = ROOT/"ligand"
for d in [RECEPTORS, POSES, LIGAND_DIR]:
    d.mkdir(exist_ok=True, parents=True)

INPUT_1XQ8 = Path("/home/anugraha/IDP/inputs/protein/1XQ8.pdb")
INPUT_WE_BOUND = Path("/home/anugraha/IDP/bound_pose_protein_ligand.pdb")
INPUT_FAS_MOL2 = Path("/home/anugraha/IDP/build/fasudil_placed.mol2")

# ───────────────────────────── 1. RECEPTOR PREP ─────────────────────────────

def strip_pdb(in_path, out_path, keep_res=None, drop_res=("HOH","WAT","Na+","Cl-","K+","FAS","FAD")):
    """Strip waters/ions/optionally a ligand. Keep only ATOM records of standard protein."""
    with open(in_path) as f, open(out_path,"w") as g:
        for line in f:
            if line.startswith(("ATOM","HETATM")):
                resname = line[17:20].strip()
                if resname in drop_res:
                    continue
                if keep_res is not None and resname not in keep_res:
                    continue
                # Vina wants ATOM record for receptor
                if line.startswith("HETATM"):
                    line = "ATOM  " + line[6:]
                g.write(line)
            elif line.startswith(("TER","END")):
                g.write(line)
    return out_path

# A: 1XQ8 raw
strip_pdb(INPUT_1XQ8, RECEPTORS/"A_1xq8.pdb")
# C: WE bound pose, fasudil stripped
strip_pdb(INPUT_WE_BOUND, RECEPTORS/"C_we_bound.pdb")

# Convert receptors to PDBQT with mk_prepare_receptor.py (Meeko's CLI, more robust)
for name in ["A_1xq8", "C_we_bound"]:
    pdb = RECEPTORS/f"{name}.pdb"
    pdbqt = RECEPTORS/f"{name}.pdbqt"
    # Use openbabel as a robust fallback: add Hs at pH 7, output rigid PDBQT
    cmd = ["obabel", str(pdb), "-O", str(pdbqt), "-xr", "-p", "7.0"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not pdbqt.exists():
        print(f"[ERROR] obabel failed for {name}:\n{r.stderr}")
        sys.exit(1)
    print(f"[prep] receptor {name}.pdbqt ready")

# ───────────────────────────── 2. LIGAND PREP ─────────────────────────────
# Built separately by prep_fasudil.py (SMILES → RDKit 3D embed → Meeko)
fas_pdbqt = LIGAND_DIR/"fasudil.pdbqt"
if not fas_pdbqt.exists():
    print(f"[ERROR] ligand pdbqt not found at {fas_pdbqt}; run prep_fasudil.py first")
    sys.exit(1)
print(f"[prep] using ligand at {fas_pdbqt}")

# ───────────────────────────── 3. SEARCH BOXES ─────────────────────────────

def get_residue_centroid(pdb, resids):
    """Centroid of CA atoms of given residues."""
    pts = []
    with open(pdb) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resid = int(line[22:26])
                if resid in resids:
                    pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(pts).mean(axis=0)

def get_protein_extents(pdb):
    """Whole-protein bounding box."""
    pts = []
    with open(pdb) as f:
        for line in f:
            if line.startswith("ATOM"):
                pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    pts = np.array(pts)
    center = pts.mean(axis=0)
    size = pts.max(axis=0) - pts.min(axis=0) + 10  # +10 Å buffer
    return center, size

# C-term centroid in each receptor (Y133, D135, Y136)
center_A_cterm = get_residue_centroid(RECEPTORS/"A_1xq8.pdb", {133, 135, 136})
center_C_cterm = get_residue_centroid(RECEPTORS/"C_we_bound.pdb", {133, 135, 136})
BIAS_SIZE = np.array([22.0, 22.0, 22.0])   # 22 Å cube = enough to allow pose flexibility

# Whole-protein bbox for blind condition
center_A_blind, size_A_blind = get_protein_extents(RECEPTORS/"A_1xq8.pdb")
# Cap blind box at Vina's max efficient size (~50 Å)
size_A_blind = np.minimum(size_A_blind, 50.0)

print(f"\n[boxes]")
print(f"  A_biased  center={center_A_cterm.round(1)}  size={BIAS_SIZE.tolist()}")
print(f"  A_blind   center={center_A_blind.round(1)}  size={size_A_blind.round(1).tolist()}")
print(f"  C_biased  center={center_C_cterm.round(1)}  size={BIAS_SIZE.tolist()}")

# ───────────────────────────── 4. RUN VINA ─────────────────────────────

CONDITIONS = [
    ("A_biased", RECEPTORS/"A_1xq8.pdbqt",    center_A_cterm, BIAS_SIZE),
    ("A_blind",  RECEPTORS/"A_1xq8.pdbqt",    center_A_blind, size_A_blind),
    ("C_biased", RECEPTORS/"C_we_bound.pdbqt", center_C_cterm, BIAS_SIZE),
]

summary_rows = []

for name, rec_pdbqt, center, size in CONDITIONS:
    print(f"\n[dock] === {name} ===")
    v = Vina(sf_name="vina", cpu=8, verbosity=0)
    v.set_receptor(str(rec_pdbqt))
    v.set_ligand_from_file(str(fas_pdbqt))
    v.compute_vina_maps(center=center.tolist(), box_size=size.tolist())
    # exhaustiveness 16 (default 8 → bumped for confidence). n_poses=20.
    v.dock(exhaustiveness=16, n_poses=20)
    out_pdbqt = POSES/f"{name}.pdbqt"
    v.write_poses(str(out_pdbqt), n_poses=20, overwrite=True)
    print(f"  wrote {out_pdbqt}")

    # Parse Vina energies from the output PDBQT
    energies = []
    with open(out_pdbqt) as f:
        for line in f:
            if line.startswith("REMARK VINA RESULT"):
                e = float(line.split()[3])
                energies.append(e)
    print(f"  energies (top 20, kcal/mol): {[round(e,2) for e in energies[:20]]}")

    # Convert each pose PDBQT → PDB for visualization (top 5 only)
    # Splitting multi-model PDBQT with obabel
    split = subprocess.run(
        ["obabel", str(out_pdbqt), "-O", str(POSES/f"{name}_pose.pdb"), "-m"],
        capture_output=True, text=True
    )
    print(f"  split into individual pose PDBs ({split.returncode})")

    # Save record
    summary_rows.append((name, energies, str(out_pdbqt)))

# ───────────────────────────── 5. CONTACT ANALYSIS ─────────────────────────────

def receptor_residues(pdb):
    """Return dict: resid → list of heavy-atom xyz."""
    res = {}
    with open(pdb) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:14].strip()[0] != "H":
                resid = int(line[22:26])
                resname = line[17:20].strip()
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                res.setdefault(resid, {"name":resname, "xyz":[]})
                res[resid]["xyz"].append(xyz)
    return res

def pose_contacts(pose_pdb, receptor_res, dcut=5.0):
    """Returns set of resids within dcut Å of any pose heavy atom."""
    lig_xyz = []
    with open(pose_pdb) as f:
        for line in f:
            if line.startswith(("ATOM","HETATM")) and line[12:14].strip()[0] != "H":
                lig_xyz.append(np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]))
    if not lig_xyz:
        return set()
    lig_xyz = np.array(lig_xyz)
    contacts = set()
    for resid, info in receptor_res.items():
        for r_xyz in info["xyz"]:
            if np.min(np.linalg.norm(lig_xyz - r_xyz, axis=1)) < dcut:
                contacts.add((resid, info["name"]))
                break
    return contacts

# Reference contacts (the WE bound pose)
TARGET_SITE = {133, 135, 136}    # Y133, D135, Y136

with open(ROOT/"summary.txt","w") as f:
    f.write("="*78 + "\n")
    f.write("1XQ8-vs-WE Docking Control — Summary\n")
    f.write("="*78 + "\n\n")
    f.write(f"Ligand: fasudil (38 heavy atoms, +1 cationic diazepane N)\n")
    f.write(f"Target site (from WE + literature): Y133, D135, Y136\n")
    f.write(f"Docker: AutoDock Vina, exhaustiveness=16, n_poses=20\n\n")

    for cond_name, energies, _ in summary_rows:
        rec_pdb = RECEPTORS/("A_1xq8.pdb" if cond_name.startswith("A") else "C_we_bound.pdb")
        rec_res = receptor_residues(rec_pdb)
        f.write("-"*78 + "\n")
        f.write(f"Condition: {cond_name}\n")
        f.write(f"Top energies (kcal/mol): {[round(e,2) for e in energies[:10]]}\n\n")
        f.write(f"{'rank':>4} {'E':>7} {'site_hit':>9} {'#contacts':>10}  contacts_to_target\n")
        hits = 0
        for i in range(min(20, len(energies))):
            pose_pdb = POSES/f"{cond_name}_pose{i+1:02d}.pdb"
            if not pose_pdb.exists():
                # obabel naming
                pose_pdb = POSES/f"{cond_name}_pose{i+1}.pdb"
            if not pose_pdb.exists():
                continue
            contacts = pose_contacts(pose_pdb, rec_res)
            target_contacts = sorted([(rid, rn) for rid, rn in contacts if rid in TARGET_SITE])
            site_hit = "YES" if any(rid in TARGET_SITE for rid, _ in contacts) else "no"
            if site_hit == "YES":
                hits += 1
            tc_str = ", ".join(f"{rn}{rid}" for rid, rn in target_contacts) if target_contacts else "-"
            f.write(f"{i+1:>4} {energies[i]:>7.2f} {site_hit:>9} {len(contacts):>10}  {tc_str}\n")
        f.write(f"\n>>> {cond_name} hit rate to target site (Y133/D135/Y136): {hits}/20 = {hits*5}%\n\n")

print(f"\n[done] summary written to {ROOT/'summary.txt'}")
print(f"[done] top poses in {POSES}/")
