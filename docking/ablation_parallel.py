#!/usr/bin/env python3
"""
Parallel scoring-chain ablation.
Same logic as ablation_run.py but uses multiprocessing for the docking loop.
"""
import os, sys, subprocess, json, time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy
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

# Number of parallel docking workers
N_WORKERS = int(os.environ.get("ABLATION_WORKERS", "32"))

def smiles_to_pdbqt(args):
    """Protonate (proper RDKit pattern: SetFormalCharge + UpdatePropertyCache),
       embed 3D, write PDBQT.  Returns id on success, None on failure with
       reason printed to stderr (no silent fallback)."""
    i, smi, out_path = args
    m = Chem.MolFromSmiles(smi)
    if m is None:
        print(f"  [prep] cand {i} SMILES parse failed: {smi}", file=sys.stderr); return None
    # Protonate first basic aliphatic N (charge only, NO manual SetNumExplicitHs).
    for atom in m.GetAtoms():
        if (atom.GetSymbol() == "N" and not atom.GetIsAromatic()
            and atom.GetFormalCharge() == 0
            and atom.GetTotalDegree() < 4 and atom.GetTotalNumHs() >= 1
            and not any(n.GetSymbol() == "S" for n in atom.GetNeighbors())):
            atom.SetFormalCharge(+1)
            break
    m.UpdatePropertyCache(strict=False)
    m = Chem.AddHs(m)
    try:
        if AllChem.EmbedMolecule(m, randomSeed=42) != 0:
            print(f"  [prep] cand {i} embed failed: {smi}", file=sys.stderr); return None
        AllChem.MMFFOptimizeMolecule(m)
    except Exception as e:
        print(f"  [prep] cand {i} embed exception: {e}", file=sys.stderr); return None
    prep = MoleculePreparation()
    try:
        prep_mols = prep.prepare(m)
    except Exception as e:
        print(f"  [prep] cand {i} meeko exception: {e}", file=sys.stderr); return None
    if not prep_mols:
        print(f"  [prep] cand {i} meeko returned empty: {smi}", file=sys.stderr); return None
    pdbqt_str, ok, msg = PDBQTWriterLegacy.write_string(prep_mols[0])
    if not ok:
        print(f"  [prep] cand {i} pdbqt write failed: {msg}", file=sys.stderr); return None
    out_path.write_text(pdbqt_str)
    return i

def get_box_center(pdb, resids={133,135,136}):
    pts = []
    for line in open(pdb):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            if int(line[22:26]) in resids:
                pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(pts).mean(axis=0).tolist()

def dock_worker(args):
    """One docking job. Imports Vina inside (avoids fork issues)."""
    cid, rname, lig_pdbqt, rec_pdbqt, center, out_pdbqt = args
    from vina import Vina
    try:
        v = Vina(sf_name="vina", cpu=2, verbosity=0)
        v.set_receptor(str(rec_pdbqt))
        v.set_ligand_from_file(str(lig_pdbqt))
        v.compute_vina_maps(center=center, box_size=BIAS_SIZE)
        v.dock(exhaustiveness=8, n_poses=3)
        v.write_poses(str(out_pdbqt), n_poses=3, overwrite=True)
        e = None
        for line in open(out_pdbqt):
            if line.startswith("REMARK VINA RESULT"):
                e = float(line.split()[3]); break
        return (cid, rname, e)
    except Exception as exc:
        return (cid, rname, None, str(exc)[:200])

def compute_plif_for_pose(pose_pdbqt, receptor_pdb):
    pose_sdf = pose_pdbqt.with_suffix(".pose1.sdf")
    r = subprocess.run(
        ["obabel", str(pose_pdbqt), "-O", str(pose_sdf), "-f", "1", "-l", "1"],
        capture_output=True, text=True
    )
    if r.returncode != 0 or not pose_sdf.exists(): return None
    try:
        m = Chem.MolFromMolFile(str(pose_sdf), removeHs=False)
        if m is None: return None
        lig = plf.Molecule.from_rdkit(m)
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
                # Strip chain suffix (e.g., "TYR133.A" → "TYR133") so we
                # match the reference PLIF built from prmtop+nc (no chain ID).
                res_key = str(prot_res).split(".")[0]
                ints.add(f"{res_key}|{itype}")
        return ints
    except Exception:
        return None

def plif_tanimoto(pose_interactions):
    if not pose_interactions: return 0.0
    ref_keys = set(REF_PLIF.keys())
    shared = pose_interactions & ref_keys
    only_ref = ref_keys - pose_interactions
    only_pose = pose_interactions - ref_keys
    num = sum(REF_PLIF[k] for k in shared)
    den = sum(REF_PLIF[k] for k in shared | only_ref) + len(only_pose)
    return num / den if den > 0 else 0.0

# ─────────────────────────────────── MAIN ──────────────────────────────────────
def main():
    t0 = time.time()
    # 1) Ligand prep (parallel)
    print(f"[1/5] preparing candidate ligands ({N_WORKERS} workers)...")
    df_cand = pd.read_csv(CANDIDATES_CSV).drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    prep_args = [(i, row["SMILES"], LIGAND_DIR/f"cand_{i:02d}.pdbqt")
                 for i, row in df_cand.iterrows()]
    with Pool(min(N_WORKERS, len(prep_args))) as p:
        prepped_ids = [r for r in p.map(smiles_to_pdbqt, prep_args) if r is not None]
    candidates = []
    for i in prepped_ids:
        candidates.append({"id": i, "smiles": df_cand.iloc[i]["SMILES"],
                          "pdbqt": LIGAND_DIR/f"cand_{i:02d}.pdbqt",
                          "tanimoto_fas": df_cand.iloc[i].get("Tanimoto", np.nan)})
    print(f"  {len(candidates)} candidates prepared")

    # 2) WE receptor PDBQTs (sequential — fast)
    print(f"\n[2/5] preparing WE receptors...")
    manifest = pd.read_csv(ENSEMBLE_DIR / "ensemble_manifest.csv")
    recs = []
    for _, row in manifest.iterrows():
        pdb = ENSEMBLE_DIR / row["receptor"]
        pdbqt = ENSEMBLE_DIR / pdb.name.replace(".pdb", ".pdbqt")
        if not pdbqt.exists():
            subprocess.run(["obabel", str(pdb), "-O", str(pdbqt), "-xr", "-p", "7.0"],
                           capture_output=True)
        recs.append({"name": pdb.stem, "pdbqt": pdbqt, "pdb": pdb,
                     "center": get_box_center(pdb), "weight": float(row["weight"])})
    center_1xq8 = get_box_center(RECEPTOR_1XQ8_PDB)
    print(f"  {len(recs)} WE receptors + 1XQ8 (total {len(recs)+1})")

    # 3) DOCK ALL — parallel
    print(f"\n[3/5] docking {len(candidates)} × {len(recs)+1} = {len(candidates)*(len(recs)+1)} runs on {N_WORKERS} workers...")
    jobs = []
    for c in candidates:
        cid = c["id"]
        jobs.append((cid, "1xq8", c["pdbqt"], RECEPTORS_1XQ8, list(center_1xq8),
                     POSES_DIR/f"cand{cid:02d}_1xq8.pdbqt"))
        for r in recs:
            jobs.append((cid, r["name"], c["pdbqt"], r["pdbqt"], r["center"],
                         POSES_DIR/f"cand{cid:02d}_{r['name']}.pdbqt"))
    print(f"  total jobs: {len(jobs)}")
    t1 = time.time()
    with Pool(N_WORKERS) as p:
        dock_results = p.map(dock_worker, jobs)
    t2 = time.time()
    print(f"  docking done in {t2-t1:.1f}s ({len(jobs)/(t2-t1):.1f} jobs/s)")

    # Index by (cid, rname)
    score_lookup = {(cid, rname): e for cid, rname, e, *_ in dock_results}

    # 4) Compute PLIF per pose
    print(f"\n[4/5] computing PLIF per pose...")
    results = []
    for c in candidates:
        cid = c["id"]
        row = {"id": cid, "smiles": c["smiles"], "tanimoto_fas": c["tanimoto_fas"]}
        row["score_1xq8"] = score_lookup.get((cid, "1xq8"))
        for r in recs:
            row[f"score_{r['name']}"] = score_lookup.get((cid, r["name"]))
        we_scores = np.array([row[f"score_{r['name']}"] for r in recs], dtype=float)
        weights = np.array([r["weight"] for r in recs])
        valid = ~np.isnan(we_scores)
        row["score_we_mean"] = float(np.average(we_scores[valid], weights=weights[valid])) if valid.any() else None
        row["score_we_best"] = float(np.nanmin(we_scores))

        # PLIF
        p1 = POSES_DIR/f"cand{cid:02d}_1xq8.pdbqt"
        inter = compute_plif_for_pose(p1, RECEPTOR_1XQ8_PDB) if p1.exists() else None
        row["plif_tanimoto_1xq8"] = plif_tanimoto(inter) if inter else 0.0
        plif_we_vals = []
        for r in recs:
            pose = POSES_DIR/f"cand{cid:02d}_{r['name']}.pdbqt"
            if pose.exists():
                inter = compute_plif_for_pose(pose, r["pdb"])
                plif_we_vals.append(plif_tanimoto(inter) if inter else 0.0)
            else:
                plif_we_vals.append(0.0)
        row["plif_tanimoto_we"] = float(np.average(plif_we_vals, weights=weights))
        results.append(row)
    print(f"  PLIF done")

    # 5) Aggregate + analysis
    df = pd.DataFrame(results).dropna(subset=["score_1xq8","score_we_mean"])
    ALPHA = 3.0
    df["SC1"] = -df["score_1xq8"]
    df["SC2"] = -df["score_we_mean"]
    df["SC3"] = df["SC1"] + ALPHA * df["plif_tanimoto_1xq8"]
    df["SC4"] = df["SC2"] + ALPHA * df["plif_tanimoto_we"]

    print("\n[5/5] aggregated rankings (sorted by SC4):")
    cols = ["id","tanimoto_fas","score_1xq8","score_we_mean","plif_tanimoto_1xq8","plif_tanimoto_we","SC1","SC2","SC3","SC4"]
    print(df.sort_values("SC4", ascending=False)[cols].to_string(index=False, float_format=lambda x: f"{x:6.3f}"))
    df.to_csv(ROOT/"ablation_scores.csv", index=False)

    print("\n══════ Spearman ρ between scoring chains ══════")
    for a in ["SC1","SC2","SC3","SC4"]:
        for b in ["SC1","SC2","SC3","SC4"]:
            if a >= b: continue
            rho, p = spearmanr(df[a], df[b])
            print(f"  {a} vs {b}:  ρ = {rho:+.3f}  (p={p:.3f})")
    print("\n══════ Top-5 overlap ══════")
    for a in ["SC1","SC2","SC3","SC4"]:
        top_a = set(df.nlargest(5, a)["id"].tolist())
        for b in ["SC1","SC2","SC3","SC4"]:
            if a >= b: continue
            top_b = set(df.nlargest(5, b)["id"].tolist())
            shared = sorted(top_a & top_b)
            print(f"  {a} ∩ {b} top-5: {len(shared)}/5  shared IDs: {shared}")

    print(f"\n[done] total runtime {time.time()-t0:.1f}s; scores → {ROOT/'ablation_scores.csv'}")

if __name__ == "__main__":
    main()
