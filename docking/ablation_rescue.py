#!/usr/bin/env python3
"""
Rescue the ablation analysis using the 220 pose PDBQTs we have on disk.
Reuses existing pose1.sdf intermediates where present.
Adds a per-pose timeout so one bad ligand can't hang the whole pipeline.
"""
import os, json, signal, subprocess, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutTimeout
import numpy as np
import pandas as pd
from rdkit import Chem
import MDAnalysis as mda
import prolif as plf
from scipy.stats import spearmanr
from meeko import PDBQTMolecule, RDKitMolCreate    # ← proper PDBQT→RDKit

ROOT = Path("/home/anugraha/IDP/docking")
POSES = ROOT/"ablation_poses"
LIGANDS = ROOT/"candidates"
ENSEMBLE = ROOT/"receptor_ensemble"
CANDIDATES_CSV = Path("/home/anugraha/IDP/reinvent_work/mol2mol_fasudil.csv")
RECEPTOR_1XQ8 = ROOT/"receptors/A_1xq8.pdb"
REF_PLIF = json.loads((ROOT/"plif/reference_plif.json").read_text())

ALL_INTER = ["Hydrophobic","HBDonor","HBAcceptor","PiStacking",
             "PiCation","CationPi","Anionic","Cationic","VdWContact"]
ALPHA = 3.0

manifest = pd.read_csv(ENSEMBLE/"ensemble_manifest.csv")
RECEPTORS = [{"name": r.receptor[:-4], "pdb": ENSEMBLE/r.receptor, "weight": float(r.weight)}
             for r in manifest.itertuples()]
RECEPTORS_BY_NAME = {r["name"]: r for r in RECEPTORS}
RECEPTORS_BY_NAME["A_1xq8"] = {"name":"A_1xq8","pdb":RECEPTOR_1XQ8,"weight":1.0}

def compute_plif_one(args):
    """Compute PLIF using Meeko's PDBQT→RDKit (preserves the original
    molecule structure; bypasses obabel's broken bond perception)."""
    pose_pdbqt, receptor_pdb = args
    try:
        pmol = PDBQTMolecule.from_file(str(pose_pdbqt))
        mols = RDKitMolCreate.from_pdbqt_mol(pmol)
        if not mols: return None
        lig = plf.Molecule.from_rdkit(mols[0])    # rank-1 docked pose
        u = mda.Universe(str(receptor_pdb))
        prot = u.select_atoms("protein")
        prot_mol = plf.Molecule.from_mda(prot)
        fp = plf.Fingerprint(ALL_INTER)
        fp.run_from_iterable([lig], prot_mol, progress=False)
        df = fp.to_dataframe()
        if df.empty: return set()
        ints = set()
        for col in df.columns:
            if df[col].iloc[0]:
                _, prot_res, itype = col
                ints.add(f"{str(prot_res).split('.')[0]}|{itype}")
        return ints
    except Exception as e:
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

def parse_top_vina_energy(pdbqt_path):
    """Return rank-1 Vina energy (kcal/mol) from a multi-pose PDBQT."""
    for line in open(pdbqt_path):
        if line.startswith("REMARK VINA RESULT"):
            return float(line.split()[3])
    return None

def main():
    # Identify which candidates have all 11 poses on disk
    df_cand = pd.read_csv(CANDIDATES_CSV).drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    cands = []
    for i, row in df_cand.iterrows():
        poses_ok = all((POSES/f"cand{i:02d}_{r['name']}.pdbqt").exists()
                       for r in RECEPTORS + [{"name":"1xq8"}])
        if poses_ok:
            cands.append({"id": i, "smiles": row["SMILES"], "tanimoto_fas": row["Tanimoto"]})
    print(f"[1/3] {len(cands)} candidates with all 11 docking poses on disk")

    # Compute PLIF for each (cand, receptor) pair with timeout per call
    print(f"\n[2/3] computing PLIF per pose (with per-call timeout)...")
    plif_jobs = []
    for c in cands:
        plif_jobs.append((POSES/f"cand{c['id']:02d}_1xq8.pdbqt", RECEPTOR_1XQ8, c["id"], "1xq8"))
        for r in RECEPTORS:
            plif_jobs.append((POSES/f"cand{c['id']:02d}_{r['name']}.pdbqt", r["pdb"], c["id"], r["name"]))

    plif_results = {}
    t0 = time.time()
    n_done = n_timeout = n_fail = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(compute_plif_one, (j[0], j[1])): (j[2], j[3]) for j in plif_jobs}
        for fut, (cid, rname) in futures.items():
            try:
                result = fut.result(timeout=15)   # 15 s per pose max
                plif_results[(cid, rname)] = result if result is not None else set()
                if result is None: n_fail += 1
            except FutTimeout:
                plif_results[(cid, rname)] = set()
                n_timeout += 1
                fut.cancel()
            n_done += 1
            if n_done % 30 == 0:
                print(f"  {n_done}/{len(plif_jobs)}  (timeouts={n_timeout}, fails={n_fail}, elapsed={time.time()-t0:.0f}s)")
    print(f"  PLIF done: {len(plif_jobs)} pairs, {n_timeout} timed out, {n_fail} failed, {time.time()-t0:.0f}s total")

    # Aggregate
    print(f"\n[3/3] aggregating scoring chains...")
    rows = []
    for c in cands:
        cid = c["id"]
        score_1xq8 = parse_top_vina_energy(POSES/f"cand{cid:02d}_1xq8.pdbqt")
        we_scores = [parse_top_vina_energy(POSES/f"cand{cid:02d}_{r['name']}.pdbqt") for r in RECEPTORS]
        weights = np.array([r["weight"] for r in RECEPTORS])
        we_scores_a = np.array([s if s is not None else np.nan for s in we_scores])
        valid = ~np.isnan(we_scores_a)
        score_we_mean = float(np.average(we_scores_a[valid], weights=weights[valid])) if valid.any() else None

        plif_1xq8_t = plif_tanimoto(plif_results.get((cid, "1xq8"), set()))
        plif_we_t = []
        for r in RECEPTORS:
            plif_we_t.append(plif_tanimoto(plif_results.get((cid, r["name"]), set())))
        plif_we_t_w = float(np.average(plif_we_t, weights=weights))

        rows.append({
            "id": cid, "smiles": c["smiles"], "tanimoto_fas": c["tanimoto_fas"],
            "score_1xq8": score_1xq8, "score_we_mean": score_we_mean,
            "plif_tanimoto_1xq8": plif_1xq8_t, "plif_tanimoto_we": plif_we_t_w,
        })

    df = pd.DataFrame(rows).dropna(subset=["score_1xq8","score_we_mean"])
    df["SC1"] = -df["score_1xq8"]
    df["SC2"] = -df["score_we_mean"]
    df["SC3"] = df["SC1"] + ALPHA * df["plif_tanimoto_1xq8"]
    df["SC4"] = df["SC2"] + ALPHA * df["plif_tanimoto_we"]

    print("\nFull table (sorted by SC4):")
    print(df.sort_values("SC4", ascending=False).to_string(index=False, float_format=lambda x: f"{x:6.3f}"))

    df.to_csv(ROOT/"ablation_scores_rescued.csv", index=False)

    print("\n══════ Spearman ρ between scoring chains ══════")
    pairs = [("SC1","SC2"),("SC1","SC3"),("SC1","SC4"),
             ("SC2","SC3"),("SC2","SC4"),("SC3","SC4")]
    for a, b in pairs:
        rho, p = spearmanr(df[a], df[b])
        print(f"  {a} vs {b}:  ρ = {rho:+.3f}  (p={p:.3f})")

    print("\n══════ Top-5 overlap ══════")
    for a in ["SC1","SC2","SC3","SC4"]:
        top_a = set(df.nlargest(5, a)["id"].tolist())
        for b in ["SC1","SC2","SC3","SC4"]:
            if a >= b: continue
            top_b = set(df.nlargest(5, b)["id"].tolist())
            shared = sorted(top_a & top_b)
            print(f"  {a} ∩ {b} top-5:  {len(shared)}/5  shared IDs {shared}")

    print("\n══════ PLIF Tanimoto distributions ══════")
    print(df[["plif_tanimoto_1xq8","plif_tanimoto_we"]].describe())

    print(f"\n[done] results → {ROOT}/ablation_scores_rescued.csv")

if __name__ == "__main__":
    main()
