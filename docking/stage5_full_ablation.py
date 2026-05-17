#!/usr/bin/env python3
"""
Stage 5 — Full Level-2 ablation using v4 chemistry-filtered variants.

Per (parent_molecule, receptor):
  1. Dock every variant of the parent into the receptor.
  2. Pick the variant with the best Vina score = the "winning" pose for this
     (parent, receptor) pair.
  3. PLIF on the winning pose, Tanimoto vs fasudil reference.
  4. Aggregate into 4 scoring chains:
       SC1 = -dock(1XQ8, best variant)
       SC2 = -weighted_mean(dock(WE_r, best variant per r))
       SC3 = SC1 + α · PLIF_T(1XQ8 winning pose)
       SC4 = SC2 + α · weighted_mean(PLIF_T(WE_r winning pose))
  5. Spearman ρ and top-5 overlap between (SC1, SC2, SC3, SC4).

Pre-flight (per the "verify before chaining" rule):
  - For every receptor: verify Y133 CA is inside the docking box.
  - For every variant PDBQT: confirm Meeko round-trip preserves SMILES
    (already verified at prep time, double-check 1 random variant here).
"""
import os, json, signal, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutTimeout
import numpy as np
import pandas as pd
from rdkit import Chem
from meeko import PDBQTMolecule, RDKitMolCreate
import MDAnalysis as mda
import prolif as plf
from scipy.stats import spearmanr

ROOT = Path("/home/anugraha/IDP/docking")
POSES = ROOT / "stage5_poses"; POSES.mkdir(exist_ok=True)
MANIFEST = json.loads((ROOT/"candidates_v4/manifest.json").read_text())
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
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            if int(line[22:26]) in resids:
                pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(pts).mean(axis=0).tolist()


def y133_ca(pdb):
    for line in open(pdb):
        if line.startswith("ATOM") and line[12:16].strip()=="CA" and int(line[22:26])==133:
            return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return None


# ───────────────────── Pre-flight: verify all receptor boxes ─────────────────
print(f"[pre-flight] verifying box geometry for all receptors...")
RECEPTORS = []
center_1xq8 = get_box_center(RECEPTOR_1XQ8_PDB)
y133 = y133_ca(RECEPTOR_1XQ8_PDB)
half = np.array(BIAS_SIZE)/2
inside = all(abs(y133[k] - center_1xq8[k]) <= half[k] for k in range(3))
print(f"  1xq8: Y133={y133.round(1)} center={[round(x,1) for x in center_1xq8]} inside={inside}")
assert inside, "BAIL — Y133 not in 1XQ8 box"
RECEPTORS.append({"name":"1xq8", "pdb":RECEPTOR_1XQ8_PDB, "pdbqt":RECEPTOR_1XQ8_PDBQT,
                  "center":center_1xq8, "weight":1.0})

for _, row in ENSEMBLE_MANIFEST.iterrows():
    pdb = ENSEMBLE/row.receptor
    pdbqt = ENSEMBLE/row.receptor.replace(".pdb",".pdbqt")
    center = get_box_center(pdb)
    y133 = y133_ca(pdb)
    inside = all(abs(y133[k] - center[k]) <= half[k] for k in range(3))
    print(f"  {pdb.stem}: Y133={y133.round(1)} center={[round(x,1) for x in center]} inside={inside}")
    assert inside, f"BAIL — Y133 not in {pdb.stem} box"
    RECEPTORS.append({"name":pdb.stem, "pdb":pdb, "pdbqt":pdbqt, "center":center,
                      "weight":float(row.weight)})
print(f"  ✓ all {len(RECEPTORS)} receptors have Y133 inside box\n")


# ───────────────────── Pre-flight: sanity-check one variant PDBQT ──────────────
print(f"[pre-flight] sanity-checking one random v4 variant for Meeko round-trip...")
test_var = MANIFEST[0]["variants"][0]
pmol = PDBQTMolecule.from_file(test_var["pdbqt"])
mols = RDKitMolCreate.from_pdbqt_mol(pmol)
rt = Chem.MolToSmiles(Chem.RemoveHs(mols[0]))
exp = Chem.MolToSmiles(Chem.MolFromSmiles(test_var["smiles"]))
print(f"  cand00_var00: got={rt}, want={exp}, match={rt==exp}")
assert rt == exp, "BAIL — variant PDBQT does not round-trip cleanly"
print(f"  ✓ sanity check passed\n")


# ───────────────────── Build job list ──────────────────────────────────────────
jobs = []
for entry in MANIFEST:
    pid = entry["cand_idx"]
    for v in entry["variants"]:
        for r in RECEPTORS:
            out = POSES / f"cand{pid:02d}_var{v['var_idx']:02d}_{r['name']}.pdbqt"
            jobs.append((pid, v["var_idx"], v["pdbqt"], r["name"], str(r["pdbqt"]), r["center"], str(out)))
print(f"[stage 5] {len(MANIFEST)} parents × ~2 variants × {len(RECEPTORS)} receptors = {len(jobs)} dockings\n")


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


# ───────────────────── DOCK ────────────────────────────────────────────────────
t0 = time.time()
print(f"[dock] launching {N_WORKERS} parallel workers...")
with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
    dock_results = list(ex.map(dock_one, jobs))
print(f"  done in {time.time()-t0:.1f}s ({len(jobs)/(time.time()-t0):.1f} dockings/s)")

# Index by (pid, vid, rname) → energy
e_lookup = {(pid,vid,rname): e for pid,vid,rname,e,err in dock_results}
errors = [(pid,vid,rname,err) for pid,vid,rname,e,err in dock_results if err]
print(f"  errors: {len(errors)} / {len(jobs)}")
if errors[:3]:
    for r in errors[:3]: print(f"    {r}")


# ───────────────────── PLIF on winning pose per (parent, receptor) ─────────────
def compute_plif(pose_pdbqt, receptor_pdb):
    try:
        pmol = PDBQTMolecule.from_file(pose_pdbqt)
        mols = RDKitMolCreate.from_pdbqt_mol(pmol)
        if not mols: return None
        lig = plf.Molecule.from_rdkit(mols[0])
        u = mda.Universe(str(receptor_pdb))
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
        return ints
    except Exception:
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

print(f"\n[plif] computing PLIF on winning pose per (parent × receptor)...")
results = []
for entry in MANIFEST:
    pid = entry["cand_idx"]
    row = {"id": pid, "smiles": entry["parent_smiles"]}
    for r in RECEPTORS:
        # Find best Vina score among this parent's variants for this receptor
        best_v = None; best_e = np.inf
        for v in entry["variants"]:
            e = e_lookup.get((pid, v["var_idx"], r["name"]))
            if e is not None and e < best_e:
                best_e = e; best_v = v
        if best_v is None:
            row[f"score_{r['name']}"] = None
            row[f"plif_{r['name']}"] = 0.0
            continue
        row[f"score_{r['name']}"] = best_e
        row[f"winner_var_{r['name']}"] = best_v["var_idx"]
        # PLIF on winning pose
        pose = POSES / f"cand{pid:02d}_var{best_v['var_idx']:02d}_{r['name']}.pdbqt"
        ints = compute_plif(str(pose), str(r["pdb"]))
        row[f"plif_{r['name']}"] = plif_tanimoto(ints) if ints else 0.0
    # Aggregate WE: weighted mean of score, weighted mean of plif
    we_scores = []; we_plifs = []; weights = []
    for r in RECEPTORS:
        if r["name"] == "1xq8": continue
        s = row.get(f"score_{r['name']}"); p = row.get(f"plif_{r['name']}", 0.0)
        if s is not None:
            we_scores.append(s); we_plifs.append(p); weights.append(r["weight"])
    weights = np.array(weights)
    row["score_we_mean"] = float(np.average(we_scores, weights=weights)) if we_scores else None
    row["plif_we_mean"]  = float(np.average(we_plifs,  weights=weights)) if we_plifs else 0.0
    results.append(row)

df = pd.DataFrame(results).dropna(subset=["score_1xq8","score_we_mean"])
df["SC1"] = -df["score_1xq8"]
df["SC2"] = -df["score_we_mean"]
df["SC3"] = df["SC1"] + ALPHA * df["plif_1xq8"]
df["SC4"] = df["SC2"] + ALPHA * df["plif_we_mean"]

print(f"\n[results] table (sorted by SC4):")
view = ["id","score_1xq8","score_we_mean","plif_1xq8","plif_we_mean","SC1","SC2","SC3","SC4"]
print(df.sort_values("SC4", ascending=False)[view].to_string(index=False, float_format=lambda x: f"{x:6.3f}"))
df.to_csv(ROOT/"stage5_results.csv", index=False)

print(f"\n══════ Spearman ρ between scoring chains ══════")
pairs = [("SC1","SC2"),("SC1","SC3"),("SC1","SC4"),("SC2","SC3"),("SC2","SC4"),("SC3","SC4")]
for a, b in pairs:
    rho, p = spearmanr(df[a], df[b])
    print(f"  {a} vs {b}:  ρ = {rho:+.3f}  (p={p:.3f})")

print(f"\n══════ Top-5 overlap ══════")
for a in ["SC1","SC2","SC3","SC4"]:
    top_a = set(df.nlargest(5, a)["id"].tolist())
    for b in ["SC1","SC2","SC3","SC4"]:
        if a >= b: continue
        top_b = set(df.nlargest(5, b)["id"].tolist())
        shared = sorted(top_a & top_b)
        print(f"  {a} ∩ {b} top-5:  {len(shared)}/5  shared IDs {shared}")

print(f"\n══════ PLIF Tanimoto distributions ══════")
print(df[["plif_1xq8","plif_we_mean"]].describe().to_string(float_format=lambda x: f"{x:.3f}"))

print(f"\n[done] full results: stage5_results.csv  (total elapsed {time.time()-t0:.0f}s)")
