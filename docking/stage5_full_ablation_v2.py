#!/usr/bin/env python3
"""
Stage 5 v2 — full ablation on the 172-parent expanded set.

Differences from stage5_full_ablation.py:
  - reads candidates_v4_full/manifest.json (172 parents, 360 variants)
  - parallelizes the PLIF computation (was sequential, ~30 min on 220 calls;
    here it's 1892 PLIF calls so parallelism is required)
  - same pre-flight geometry checks
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
POSES = ROOT / "stage5_v2_poses"; POSES.mkdir(exist_ok=True)
MANIFEST = json.loads((ROOT/"candidates_v4_full/manifest.json").read_text())
REF_PLIF = json.loads((ROOT/"plif/reference_plif.json").read_text())
RECEPTOR_1XQ8_PDB = ROOT/"receptors/A_1xq8.pdb"
RECEPTOR_1XQ8_PDBQT = ROOT/"receptors/A_1xq8.pdbqt"
ENSEMBLE = ROOT/"receptor_ensemble"
ENSEMBLE_MANIFEST = pd.read_csv(ENSEMBLE/"ensemble_manifest.csv")

ALL_INTER = ["Hydrophobic","HBDonor","HBAcceptor","PiStacking",
             "PiCation","CationPi","Anionic","Cationic","VdWContact"]
ALPHA = 3.0
BIAS_SIZE = [22.0, 22.0, 22.0]
N_WORKERS = int(os.environ.get("STAGE5_WORKERS", "32"))


def get_box_center(pdb, resids={133,135,136}):
    pts = []
    for line in open(pdb):
        if line.startswith("ATOM") and line[12:16].strip()=="CA" and int(line[22:26]) in resids:
            pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(pts).mean(axis=0).tolist()

def y133_ca(pdb):
    for line in open(pdb):
        if line.startswith("ATOM") and line[12:16].strip()=="CA" and int(line[22:26])==133:
            return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return None


# ───────────────────── Pre-flight: receptors + variant round-trip ─────────────
print(f"[pre-flight] verifying box geometry for all receptors...")
RECEPTORS = [{"name":"1xq8", "pdb":RECEPTOR_1XQ8_PDB, "pdbqt":RECEPTOR_1XQ8_PDBQT,
              "center":get_box_center(RECEPTOR_1XQ8_PDB), "weight":1.0}]
half = np.array(BIAS_SIZE)/2
for _, row in ENSEMBLE_MANIFEST.iterrows():
    pdb = ENSEMBLE/row.receptor
    pdbqt = ENSEMBLE/row.receptor.replace(".pdb",".pdbqt")
    center = get_box_center(pdb)
    RECEPTORS.append({"name":pdb.stem, "pdb":pdb, "pdbqt":pdbqt, "center":center,
                      "weight":float(row.weight)})
for r in RECEPTORS:
    y133 = y133_ca(r["pdb"])
    inside = all(abs(y133[k] - r["center"][k]) <= half[k] for k in range(3))
    if not inside:
        print(f"  BAIL — {r['name']}: Y133 {y133.round(1)} not in box"); sys.exit(1)
print(f"  ✓ all {len(RECEPTORS)} receptors have Y133 inside box")

# Sanity-check one random variant PDBQT
tv = MANIFEST[0]["variants"][0]
pmol = PDBQTMolecule.from_file(tv["pdbqt"])
mols = RDKitMolCreate.from_pdbqt_mol(pmol)
rt = Chem.MolToSmiles(Chem.RemoveHs(mols[0]))
exp = Chem.MolToSmiles(Chem.MolFromSmiles(tv["smiles"]))
assert rt == exp, f"BAIL — variant round-trip failed: {rt} vs {exp}"
print(f"  ✓ variant round-trip sanity check passed")


# ───────────────────── Build job lists ────────────────────────────────────────
dock_jobs = []
for entry in MANIFEST:
    pid = entry["cand_idx"]
    for v in entry["variants"]:
        for r in RECEPTORS:
            out = POSES / f"cand{pid:03d}_var{v['var_idx']:02d}_{r['name']}.pdbqt"
            dock_jobs.append((pid, v["var_idx"], v["pdbqt"], r["name"], str(r["pdbqt"]), r["center"], str(out)))
n_dock = len(dock_jobs)
print(f"\n[stage 5 v2] {len(MANIFEST)} parents × ~2 variants × {len(RECEPTORS)} receptors = {n_dock} dockings\n")


def dock_one(args):
    pid, vid, lig_pdbqt, rname, rec_pdbqt, center, out_path = args
    from vina import Vina
    try:
        vv = Vina(sf_name="vina", cpu=2, verbosity=0, seed=42)
        vv.set_receptor(rec_pdbqt)
        vv.set_ligand_from_file(lig_pdbqt)
        vv.compute_vina_maps(center=list(center), box_size=BIAS_SIZE)
        vv.dock(exhaustiveness=8, n_poses=3)
        vv.write_poses(out_path, n_poses=3, overwrite=True)
        e = None
        for line in open(out_path):
            if line.startswith("REMARK VINA RESULT"):
                e = float(line.split()[3]); break
        return (pid, vid, rname, e, None)
    except Exception as exc:
        return (pid, vid, rname, None, str(exc)[:200])


# ───────────────────── DOCK (parallel) ────────────────────────────────────────
t0 = time.time()
print(f"[dock] launching {N_WORKERS} parallel workers on {n_dock} jobs...")
with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
    dock_results = list(ex.map(dock_one, dock_jobs))
print(f"  done in {time.time()-t0:.1f}s ({n_dock/(time.time()-t0):.1f} dockings/s)")
e_lookup = {(pid,vid,rname): e for pid,vid,rname,e,err in dock_results}
errors = [(pid,vid,rname,err) for pid,vid,rname,e,err in dock_results if err]
print(f"  errors: {len(errors)} / {n_dock}")
if errors[:3]:
    for r in errors[:3]: print(f"    {r}")


# ───────────────────── Identify winning variant per (parent, receptor) ────────
winning = []  # list of (pid, rname, var_idx, score, pose_path, receptor_pdb)
for entry in MANIFEST:
    pid = entry["cand_idx"]
    for r in RECEPTORS:
        best_v = None; best_e = np.inf
        for v in entry["variants"]:
            e = e_lookup.get((pid, v["var_idx"], r["name"]))
            if e is not None and e < best_e:
                best_e = e; best_v = v
        if best_v is None: continue
        pose_path = POSES / f"cand{pid:03d}_var{best_v['var_idx']:02d}_{r['name']}.pdbqt"
        winning.append((pid, r["name"], best_v["var_idx"], best_e,
                        str(pose_path), str(r["pdb"])))
print(f"\n[winning poses] {len(winning)} parent×receptor pairs")


def compute_plif_one(args):
    pid, rname, vid, energy, pose_path, receptor_pdb = args
    try:
        pmol = PDBQTMolecule.from_file(pose_path)
        mols = RDKitMolCreate.from_pdbqt_mol(pmol)
        if not mols: return (pid, rname, vid, energy, 0.0, 0)
        lig = plf.Molecule.from_rdkit(mols[0])
        u = mda.Universe(receptor_pdb)
        prot = u.select_atoms("protein")
        prot_mol = plf.Molecule.from_mda(prot)
        fp = plf.Fingerprint(ALL_INTER)
        fp.run_from_iterable([lig], prot_mol, progress=False)
        df = fp.to_dataframe()
        ints = set()
        if not df.empty:
            for col in df.columns:
                if df[col].iloc[0]:
                    _, prot_res, itype = col
                    ints.add(f"{str(prot_res).split('.')[0]}|{itype}")
        # Compute Tanimoto
        ref_keys = set(REF_PLIF.keys())
        shared = ints & ref_keys
        only_ref = ref_keys - ints
        only_pose = ints - ref_keys
        num = sum(REF_PLIF[k] for k in shared)
        den = sum(REF_PLIF[k] for k in shared | only_ref) + len(only_pose)
        tani = num/den if den > 0 else 0.0
        return (pid, rname, vid, energy, tani, len(ints))
    except Exception:
        return (pid, rname, vid, energy, 0.0, 0)


# ───────────────────── PLIF (parallel) ────────────────────────────────────────
t1 = time.time()
print(f"\n[plif] computing PLIF on {len(winning)} winning poses (parallel, {N_WORKERS} workers)...")
with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
    plif_results = list(ex.map(compute_plif_one, winning))
print(f"  done in {time.time()-t1:.1f}s")

# Index → (pid, rname) → (var_idx, score, tani, n_int)
plif_lookup = {(pid, rname): (vid, e, tani, n) for pid, rname, vid, e, tani, n in plif_results}


# ───────────────────── Aggregate scoring chains ───────────────────────────────
rows = []
for entry in MANIFEST:
    pid = entry["cand_idx"]
    row = {"id": pid, "smiles": entry["parent_smiles"]}
    s1xq8 = plif_lookup.get((pid, "1xq8"))
    if s1xq8 is None: continue
    row["score_1xq8"] = s1xq8[1]; row["plif_1xq8"] = s1xq8[2]
    we_scores = []; we_plifs = []; weights = []
    for r in RECEPTORS:
        if r["name"] == "1xq8": continue
        info = plif_lookup.get((pid, r["name"]))
        if info is None: continue
        we_scores.append(info[1]); we_plifs.append(info[2]); weights.append(r["weight"])
    weights = np.array(weights)
    if len(we_scores) == 0: continue
    row["score_we_mean"] = float(np.average(we_scores, weights=weights))
    row["plif_we_mean"]  = float(np.average(we_plifs, weights=weights))
    rows.append(row)

df = pd.DataFrame(rows).dropna(subset=["score_1xq8","score_we_mean"])
df["SC1"] = -df["score_1xq8"]
df["SC2"] = -df["score_we_mean"]
df["SC3"] = df["SC1"] + ALPHA * df["plif_1xq8"]
df["SC4"] = df["SC2"] + ALPHA * df["plif_we_mean"]
df.to_csv(ROOT/"stage5_v2_results.csv", index=False)
print(f"\n[results] {len(df)} candidates → stage5_v2_results.csv")

# ───────────────────── Correlations + top-N overlap ───────────────────────────
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
        shared = sorted(top_a & top_b)
        print(f"    {a} ∩ {b}: {len(shared)}/{N}")

print(f"\n══════ PLIF Tanimoto stats ══════")
print(df[["plif_1xq8","plif_we_mean"]].describe().to_string(float_format=lambda x: f"{x:.3f}"))

print(f"\n[done] total elapsed {time.time()-t0:.0f}s")
