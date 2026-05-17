#!/usr/bin/env python3
"""
PLIF-only rescue for the Stage 5 v2 ablation.

Reuses the 3,960 docking poses already on disk from job 795588.
Identifies winning variant per (parent, receptor) and computes PLIF.

Key fix vs the timed-out original: each worker caches the 11 receptor
MDAnalysis Universes ONCE at startup, not per PLIF call. This eliminates
the ~1.5 sec/call file-reload overhead that caused the timeout.
"""
import os, json, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from rdkit import Chem
from meeko import PDBQTMolecule, RDKitMolCreate
import MDAnalysis as mda
import prolif as plf
from scipy.stats import spearmanr

ROOT = Path("/home/anugraha/IDP/docking")
POSES = ROOT / "stage5_v2_poses"
MANIFEST = json.loads((ROOT/"candidates_v4_full/manifest.json").read_text())
REF_PLIF = json.loads((ROOT/"plif/reference_plif.json").read_text())
RECEPTOR_1XQ8_PDB = ROOT/"receptors/A_1xq8.pdb"
ENSEMBLE = ROOT/"receptor_ensemble"
ENSEMBLE_MANIFEST = pd.read_csv(ENSEMBLE/"ensemble_manifest.csv")

ALL_INTER = ["Hydrophobic","HBDonor","HBAcceptor","PiStacking",
             "PiCation","CationPi","Anionic","Cationic","VdWContact"]
ALPHA = 3.0
N_WORKERS = int(os.environ.get("PLIF_WORKERS", "32"))


def build_receptor_cache():
    """Build (name → mda.Universe protein selection) cache."""
    cache = [{"name":"1xq8", "pdb":RECEPTOR_1XQ8_PDB, "weight":1.0}]
    for _, row in ENSEMBLE_MANIFEST.iterrows():
        cache.append({"name":Path(row.receptor).stem,
                      "pdb":ENSEMBLE/row.receptor, "weight":float(row.weight)})
    return cache


# ───────────────────── Verify input data ──────────────────────────────────────
n_poses = sum(1 for _ in POSES.glob("cand*_var*_*.pdbqt"))
print(f"[rescue] poses on disk: {n_poses}  (expected 3960)")
if n_poses < 3500:
    print("BAIL — too few poses, restart from docking"); sys.exit(1)

receptors_meta = build_receptor_cache()
print(f"[rescue] {len(receptors_meta)} receptors to cache per worker")


def vina_score(pose_pdbqt):
    for line in open(pose_pdbqt):
        if line.startswith("REMARK VINA RESULT"):
            return float(line.split()[3])
    return None


def plif_tanimoto(pose_keys):
    if not pose_keys: return 0.0
    ref_keys = set(REF_PLIF.keys())
    shared = pose_keys & ref_keys
    only_ref = ref_keys - pose_keys
    only_pose = pose_keys - ref_keys
    num = sum(REF_PLIF[k] for k in shared)
    den = sum(REF_PLIF[k] for k in shared | only_ref) + len(only_pose)
    return num/den if den > 0 else 0.0


def worker_plif_batch(args):
    """
    Worker function — receives a batch of (cand_id, receptor_name, pose_path)
    tuples and the receptor list. Caches all 11 receptor MDAnalysis Universes
    + ProLIF protein objects ONCE, then loops through the batch.

    Returns: list of (cand_id, receptor_name, energy, tanimoto, n_contacts).
    """
    batch, receptor_list = args
    # Build per-worker receptor cache
    receptor_cache = {}
    for r in receptor_list:
        u = mda.Universe(str(r["pdb"]))
        prot = u.select_atoms("protein")
        prot_mol = plf.Molecule.from_mda(prot)
        receptor_cache[r["name"]] = prot_mol

    results = []
    for cid, rname, pose_path in batch:
        try:
            pmol = PDBQTMolecule.from_file(pose_path)
            mols = RDKitMolCreate.from_pdbqt_mol(pmol)
            if not mols:
                results.append((cid, rname, None, 0.0, 0)); continue
            lig = plf.Molecule.from_rdkit(mols[0])
            prot_mol = receptor_cache[rname]
            fp = plf.Fingerprint(ALL_INTER)
            fp.run_from_iterable([lig], prot_mol, progress=False)
            df = fp.to_dataframe()
            ints = set()
            if not df.empty:
                for col in df.columns:
                    if df[col].iloc[0]:
                        _, prot_res, itype = col
                        ints.add(f"{str(prot_res).split('.')[0]}|{itype}")
            tani = plif_tanimoto(ints)
            energy = vina_score(pose_path)
            results.append((cid, rname, energy, tani, len(ints)))
        except Exception:
            results.append((cid, rname, None, 0.0, 0))
    return results


# ───────────────────── Identify winning poses ────────────────────────────────
print(f"\n[step 1] identifying winning variant per (parent, receptor)...")
winning = []   # list of (cid, rname, pose_path)
for entry in MANIFEST:
    pid = entry["cand_idx"]
    for r in receptors_meta:
        best_e = np.inf; best_pose = None
        for v in entry["variants"]:
            pose = POSES / f"cand{pid:03d}_var{v['var_idx']:02d}_{r['name']}.pdbqt"
            if not pose.exists(): continue
            e = vina_score(str(pose))
            if e is not None and e < best_e:
                best_e = e
                best_pose = str(pose)
        if best_pose is not None:
            winning.append((pid, r["name"], best_pose))
print(f"  {len(winning)} winning poses identified")


# ───────────────────── PLIF in batches (proper caching) ─────────────────────
print(f"\n[step 2] PLIF on winning poses, {N_WORKERS} workers, batched...")
batch_size = max(1, len(winning) // N_WORKERS + 1)
batches = [winning[i:i+batch_size] for i in range(0, len(winning), batch_size)]
print(f"  {len(batches)} batches of ~{batch_size} poses each")

t0 = time.time()
with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
    batch_results = list(ex.map(worker_plif_batch, [(b, receptors_meta) for b in batches]))
all_results = [r for batch in batch_results for r in batch]
print(f"  done in {time.time()-t0:.1f}s")


# ───────────────────── Aggregate scoring chains ──────────────────────────────
print(f"\n[step 3] aggregating scoring chains...")
# Index by (cid, rname) → (energy, tani, n)
lookup = {(c, r): (e, t, n) for c, r, e, t, n in all_results}

rows = []
for entry in MANIFEST:
    pid = entry["cand_idx"]
    s1xq8 = lookup.get((pid, "1xq8"))
    if s1xq8 is None or s1xq8[0] is None: continue
    row = {"id": pid, "smiles": entry["parent_smiles"],
           "score_1xq8": s1xq8[0], "plif_1xq8": s1xq8[1]}
    we_scores, we_plifs, weights = [], [], []
    for r in receptors_meta:
        if r["name"] == "1xq8": continue
        info = lookup.get((pid, r["name"]))
        if info is None or info[0] is None: continue
        we_scores.append(info[0]); we_plifs.append(info[1]); weights.append(r["weight"])
    if len(we_scores) == 0: continue
    weights = np.array(weights)
    row["score_we_mean"] = float(np.average(we_scores, weights=weights))
    row["plif_we_mean"]  = float(np.average(we_plifs, weights=weights))
    rows.append(row)

df = pd.DataFrame(rows).dropna(subset=["score_1xq8","score_we_mean"])
df["SC1"] = -df["score_1xq8"]
df["SC2"] = -df["score_we_mean"]
df["SC3"] = df["SC1"] + ALPHA * df["plif_1xq8"]
df["SC4"] = df["SC2"] + ALPHA * df["plif_we_mean"]
df.to_csv(ROOT/"stage5_v2_results.csv", index=False)
print(f"  {len(df)} candidates → stage5_v2_results.csv")

# Correlation matrix
print(f"\n══════ Spearman ρ between scoring chains ══════")
pairs = [("SC1","SC2"),("SC1","SC3"),("SC1","SC4"),("SC2","SC3"),("SC2","SC4"),("SC3","SC4")]
for a, b in pairs:
    rho, p = spearmanr(df[a], df[b])
    print(f"  {a} vs {b}:  ρ = {rho:+.3f}  (p={p:.3g})")

print(f"\n══════ Top-N overlap ══════")
for N in [5, 10, 20]:
    print(f"  Top-{N}:")
    for a, b in pairs:
        top_a = set(df.nlargest(N, a)["id"].tolist())
        top_b = set(df.nlargest(N, b)["id"].tolist())
        print(f"    {a} ∩ {b}: {len(top_a & top_b)}/{N}")

print(f"\n══════ PLIF Tanimoto stats ══════")
print(df[["plif_1xq8","plif_we_mean"]].describe().to_string(float_format=lambda x: f"{x:.3f}"))

print(f"\n[done] total {time.time()-t0:.0f}s")
