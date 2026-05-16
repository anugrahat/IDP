#!/usr/bin/env python3
"""
Scoring-chain ablation: do WE-derived terms change REINVENT rankings?

For each of 20 fasudil analogues (from REINVENT Mol2Mol):
  Dock into 1XQ8                       → score_1xq8        (SC1 raw)
  Dock into 10 WE receptors            → score_we_each, weighted mean (SC2 raw)
  Compute PLIF for best pose on each   → plif_tanimoto_1xq8, plif_tanimoto_we
  Combine to form 4 scoring chains:
      SC1 = -dock_1xq8
      SC2 = -weighted_mean(dock_we)
      SC3 = SC1 + α * plif_tanimoto_1xq8
      SC4 = SC2 + α * plif_tanimoto_we

Compute Spearman correlation + top-5 overlap between (SC1,SC2,SC3,SC4).
"""
import os, sys, subprocess, json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina
import MDAnalysis as mda
import prolif as plf
from scipy.stats import spearmanr

ROOT = Path("/home/anugraha/IDP/docking")
LIGAND_DIR = ROOT / "candidates"
LIGAND_DIR.mkdir(exist_ok=True)
POSES_DIR = ROOT / "ablation_poses"
POSES_DIR.mkdir(exist_ok=True)

CANDIDATES_CSV = Path("/home/anugraha/IDP/reinvent_work/mol2mol_fasudil.csv")
RECEPTORS_1XQ8 = ROOT / "receptors/A_1xq8.pdbqt"
RECEPTOR_1XQ8_PDB = ROOT / "receptors/A_1xq8.pdb"
ENSEMBLE_DIR = ROOT / "receptor_ensemble"
REF_PLIF = json.loads((ROOT / "plif/reference_plif.json").read_text())

ALL_INTERACTIONS = ["Hydrophobic","HBDonor","HBAcceptor","PiStacking",
                    "PiCation","CationPi","Anionic","Cationic","VdWContact"]
BIAS_SIZE = [22.0, 22.0, 22.0]

# ──────────────────────────────────────────────── 1) LIGAND PREP ─────────────────
def smiles_to_pdbqt(smi, out_path):
    """Build neutral SMILES → protonate basic amines → 3D → meeko PDBQT."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    # Protonate basic aliphatic amines (rough approximation)
    for atom in m.GetAtoms():
        if (atom.GetSymbol() == "N" and not atom.GetIsAromatic()
            and atom.GetTotalDegree() < 4
            and atom.GetTotalNumHs() >= 1
            and not any(n.GetSymbol() == "S" for n in atom.GetNeighbors())):
            atom.SetFormalCharge(+1)
            atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
            break
    m = Chem.AddHs(m)
    try:
        rc = AllChem.EmbedMolecule(m, randomSeed=42)
        if rc != 0: return False
        AllChem.MMFFOptimizeMolecule(m)
    except Exception:
        return False
    prep = MoleculePreparation()
    prep_mols = prep.prepare(m)
    if not prep_mols: return False
    pdbqt_str, ok, _ = PDBQTWriterLegacy.write_string(prep_mols[0])
    if not ok: return False
    out_path.write_text(pdbqt_str)
    return True

print("[1/5] preparing 20 candidate ligands as PDBQT...")
df_cand = pd.read_csv(CANDIDATES_CSV)
df_cand = df_cand.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
candidates = []
for i, row in df_cand.iterrows():
    smi = row["SMILES"]
    out = LIGAND_DIR / f"cand_{i:02d}.pdbqt"
    ok = smiles_to_pdbqt(smi, out)
    if ok:
        candidates.append({"id": i, "smiles": smi, "pdbqt": out, "tanimoto_fas": row.get("Tanimoto", np.nan)})
        print(f"  cand_{i:02d}  Tanimoto={row.get('Tanimoto',0):.3f}  {smi[:60]}")
    else:
        print(f"  cand_{i:02d} FAILED  {smi[:60]}")
print(f"  → {len(candidates)} candidates ready for docking")

# ─────────────────────────── 2) RECEPTOR PREP + BOX COORDS ───────────────────
def get_box_center(pdb, resids={133,135,136}):
    """Centroid of CA atoms in given residues."""
    pts = []
    for line in open(pdb):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            if int(line[22:26]) in resids:
                pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(pts).mean(axis=0)

# Make WE receptor PDBQTs
print("\n[2/5] preparing WE receptor PDBQTs + computing box centers...")
manifest = pd.read_csv(ENSEMBLE_DIR / "ensemble_manifest.csv")
recs = []
for _, row in manifest.iterrows():
    pdb = ENSEMBLE_DIR / row["receptor"]
    pdbqt = ENSEMBLE_DIR / pdb.name.replace(".pdb", ".pdbqt")
    if not pdbqt.exists():
        r = subprocess.run(["obabel", str(pdb), "-O", str(pdbqt), "-xr", "-p", "7.0"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  obabel failed for {pdb.name}: {r.stderr[-200:]}")
            continue
    center = get_box_center(pdb)
    recs.append({"name": pdb.stem, "pdbqt": pdbqt, "pdb": pdb,
                 "center": center, "weight": float(row["weight"])})
print(f"  WE receptors ready: {len(recs)}")
print(f"  weights sum: {sum(r['weight'] for r in recs):.3f}")

# 1XQ8 box
center_1xq8 = get_box_center(RECEPTOR_1XQ8_PDB)
print(f"  1XQ8 box center: {center_1xq8.round(1)}")

# ──────────────────────────────────────────── 3) DOCK ALL ──────────────────────
def dock_one(lig_pdbqt, rec_pdbqt, center, out_pdbqt, n_poses=5):
    v = Vina(sf_name="vina", cpu=4, verbosity=0)
    v.set_receptor(str(rec_pdbqt))
    v.set_ligand_from_file(str(lig_pdbqt))
    v.compute_vina_maps(center=list(center), box_size=BIAS_SIZE)
    v.dock(exhaustiveness=8, n_poses=n_poses)
    v.write_poses(str(out_pdbqt), n_poses=n_poses, overwrite=True)
    energies = []
    for line in open(out_pdbqt):
        if line.startswith("REMARK VINA RESULT"):
            energies.append(float(line.split()[3]))
    return energies[0] if energies else None

print("\n[3/5] docking 20 ligands × 11 receptors (1 × 1XQ8 + 10 × WE)...")
results = []
for c in candidates:
    cid = c["id"]
    row = {"id": cid, "smiles": c["smiles"], "tanimoto_fas": c["tanimoto_fas"]}
    # 1XQ8
    out = POSES_DIR / f"cand{cid:02d}_1xq8.pdbqt"
    e = dock_one(c["pdbqt"], RECEPTORS_1XQ8, center_1xq8, out)
    row["score_1xq8"] = e
    # 10 WE receptors
    for r in recs:
        out = POSES_DIR / f"cand{cid:02d}_{r['name']}.pdbqt"
        e = dock_one(c["pdbqt"], r["pdbqt"], r["center"], out)
        row[f"score_{r['name']}"] = e
    we_scores = np.array([row[f"score_{r['name']}"] for r in recs])
    weights = np.array([r["weight"] for r in recs])
    row["score_we_mean"] = float(np.average(we_scores, weights=weights))
    row["score_we_best"] = float(we_scores.min())
    results.append(row)
    print(f"  cand_{cid:02d}: 1XQ8={row['score_1xq8']:.2f}  WE_mean={row['score_we_mean']:.2f}  WE_best={row['score_we_best']:.2f}")

# ─────────────────────────────────── 4) PLIF PER DOCKED POSE ───────────────────
def compute_plif_for_pose(pose_pdbqt, receptor_pdb):
    """Convert pose PDBQT → PDB, combine with receptor, run ProLIF on rank-1 pose."""
    # Extract just the rank-1 pose as SDF (ProLIF prefers SDF/PDB with bond info)
    pose_sdf = pose_pdbqt.with_suffix(".pose1.sdf")
    r = subprocess.run(
        ["obabel", str(pose_pdbqt), "-O", str(pose_sdf), "-f", "1", "-l", "1"],
        capture_output=True, text=True
    )
    if r.returncode != 0 or not pose_sdf.exists():
        return None
    try:
        lig = plf.Molecule.from_rdkit(Chem.MolFromMolFile(str(pose_sdf), removeHs=False))
        # Receptor: load with MDAnalysis from PDB
        u_rec = mda.Universe(str(receptor_pdb))
        prot = u_rec.select_atoms("protein")
        prot_mol = plf.Molecule.from_mda(prot)
        fp = plf.Fingerprint(ALL_INTERACTIONS)
        fp.run_from_iterable([lig], prot_mol, progress=False)
        df = fp.to_dataframe()
        if df.empty: return set()
        ints = set()
        for col in df.columns:
            if df[col].iloc[0]:
                _, prot_res, itype = col
                ints.add(f"{prot_res}|{itype}")
        return ints
    except Exception as e:
        return None

def plif_tanimoto(pose_interactions):
    if not pose_interactions:
        return 0.0
    # Weighted Tanimoto: numerator = sum of ref_freq for shared, denominator = ref_freq for ref + 1 per unique-to-pose
    ref_keys = set(REF_PLIF.keys())
    shared = pose_interactions & ref_keys
    only_ref = ref_keys - pose_interactions
    only_pose = pose_interactions - ref_keys
    # Frequency-weighted Tanimoto
    num = sum(REF_PLIF[k] for k in shared)
    den = sum(REF_PLIF[k] for k in shared | only_ref) + len(only_pose)
    return num / den if den > 0 else 0.0

print("\n[4/5] computing PLIF Tanimoto for each docked pose vs fasudil reference...")
for row in results:
    cid = row["id"]
    # PLIF for 1XQ8 docked pose
    pose1 = POSES_DIR / f"cand{cid:02d}_1xq8.pdbqt"
    inter = compute_plif_for_pose(pose1, RECEPTOR_1XQ8_PDB)
    row["plif_tanimoto_1xq8"] = plif_tanimoto(inter) if inter else 0.0
    # PLIF averaged across WE receptors (weighted)
    plif_tans = []
    for r in recs:
        pose = POSES_DIR / f"cand{cid:02d}_{r['name']}.pdbqt"
        inter = compute_plif_for_pose(pose, r["pdb"])
        plif_tans.append(plif_tanimoto(inter) if inter else 0.0)
    weights = np.array([r["weight"] for r in recs])
    row["plif_tanimoto_we"] = float(np.average(plif_tans, weights=weights))
    print(f"  cand_{cid:02d}  PLIF_1XQ8={row['plif_tanimoto_1xq8']:.3f}  PLIF_WE={row['plif_tanimoto_we']:.3f}")

# ───────────────────────────────── 5) AGGREGATE INTO 4 SCORING CHAINS ──────────
df = pd.DataFrame(results)
ALPHA = 3.0   # weight for PLIF term (PLIF ∈ [0,1], dock ∈ ~[-5,0]) so 3*PLIF ≈ -dock magnitude
df["SC1"] = -df["score_1xq8"]
df["SC2"] = -df["score_we_mean"]
df["SC3"] = df["SC1"] + ALPHA * df["plif_tanimoto_1xq8"]
df["SC4"] = df["SC2"] + ALPHA * df["plif_tanimoto_we"]

print("\n[5/5] aggregated rankings:")
view_cols = ["id","tanimoto_fas","SC1","SC2","SC3","SC4","plif_tanimoto_we","smiles"]
print(df.sort_values("SC4", ascending=False)[view_cols].to_string(index=False))
df.to_csv(ROOT/"ablation_scores.csv", index=False)

# Correlation analysis
print("\n══════════ Spearman rank correlation between scoring chains ══════════")
for a in ["SC1","SC2","SC3","SC4"]:
    for b in ["SC1","SC2","SC3","SC4"]:
        if a >= b: continue
        rho, p = spearmanr(df[a], df[b])
        print(f"  {a} vs {b}:  ρ = {rho:+.3f}  (p = {p:.3f})")

print("\n══════════ Top-5 overlap between scoring chains ══════════")
for a in ["SC1","SC2","SC3","SC4"]:
    top_a = set(df.nlargest(5, a)["id"].tolist())
    for b in ["SC1","SC2","SC3","SC4"]:
        if a >= b: continue
        top_b = set(df.nlargest(5, b)["id"].tolist())
        overlap = len(top_a & top_b)
        print(f"  {a} ∩ {b} top-5:  {overlap}/5  (shared IDs: {sorted(top_a & top_b)})")

print(f"\n[done] full table written to {ROOT/'ablation_scores.csv'}")
