#!/usr/bin/env python3
"""
PLIF-only rescue v2 — adds per-PLIF SIGALRM timeout so one bad pose
can't hang the whole batch (which is what caused 795670 to deadlock).

Each PLIF call gets 10 seconds max. If it hangs, raise TimeoutError, log
the pose ID, record (energy from disk, plif=0, n_int=0), continue.

Also: writes intermediate partial results every batch so progress survives
any future crash.
"""
import os, json, sys, time, signal, traceback
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
PLIF_TIMEOUT_SEC = 10        # per-pose hard limit
PARTIAL_CSV = ROOT/"stage5_v2_plif_partial.csv"


def build_receptor_meta():
    out = [{"name":"1xq8", "pdb":RECEPTOR_1XQ8_PDB, "weight":1.0}]
    for _, row in ENSEMBLE_MANIFEST.iterrows():
        out.append({"name":Path(row.receptor).stem,
                    "pdb":ENSEMBLE/row.receptor, "weight":float(row.weight)})
    return out


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


def _alarm_handler(signum, frame):
    raise TimeoutError("PLIF call timed out")


def worker_plif_batch(args):
    """Per-worker: cache 11 receptors once, then loop with per-pose SIGALRM timeout."""
    batch, receptor_meta = args
    receptor_cache = {}
    for r in receptor_meta:
        try:
            u = mda.Universe(str(r["pdb"]))
            prot = u.select_atoms("protein")
            prot_mol = plf.Molecule.from_mda(prot)
            receptor_cache[r["name"]] = prot_mol
        except Exception as e:
            print(f"  worker: failed to cache {r['name']}: {e}", file=sys.stderr)

    signal.signal(signal.SIGALRM, _alarm_handler)
    results = []
    for cid, rname, pose_path in batch:
        energy = vina_score(pose_path)
        if rname not in receptor_cache:
            results.append((cid, rname, energy, 0.0, 0, "no_receptor")); continue
        signal.alarm(PLIF_TIMEOUT_SEC)
        try:
            pmol = PDBQTMolecule.from_file(pose_path)
            mols = RDKitMolCreate.from_pdbqt_mol(pmol)
            if not mols:
                results.append((cid, rname, energy, 0.0, 0, "no_mol")); continue
            lig = plf.Molecule.from_rdkit(mols[0])
            fp = plf.Fingerprint(ALL_INTER)
            fp.run_from_iterable([lig], receptor_cache[rname], progress=False)
            df = fp.to_dataframe()
            ints = set()
            if not df.empty:
                for col in df.columns:
                    if df[col].iloc[0]:
                        _, prot_res, itype = col
                        ints.add(f"{str(prot_res).split('.')[0]}|{itype}")
            tani = plif_tanimoto(ints)
            results.append((cid, rname, energy, tani, len(ints), "ok"))
        except TimeoutError:
            results.append((cid, rname, energy, 0.0, 0, "timeout"))
        except Exception as e:
            results.append((cid, rname, energy, 0.0, 0, f"err:{type(e).__name__}"))
        finally:
            signal.alarm(0)
    return results


# ───────────────────────── pre-flight ────────────────────────────────────────
n_poses = sum(1 for _ in POSES.glob("cand*_var*_*.pdbqt"))
print(f"[rescue v2] poses on disk: {n_poses}  (need 3960)")
if n_poses < 3500:
    print("BAIL — too few poses"); sys.exit(1)

receptors_meta = build_receptor_meta()
print(f"[rescue v2] {len(receptors_meta)} receptors per worker  |  timeout {PLIF_TIMEOUT_SEC}s per pose")

# Identify winning poses
print("\n[step 1] winning variants...")
winning = []
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
print(f"  {len(winning)} winning poses")


# ───────────────────────── PLIF with timeout + incremental save ───────────────
print(f"\n[step 2] PLIF with {N_WORKERS} workers, batches of 20, partial save every batch...")
BATCH_SIZE = 20   # small batches → granular failure isolation + incremental save
batches = [winning[i:i+BATCH_SIZE] for i in range(0, len(winning), BATCH_SIZE)]
print(f"  {len(batches)} batches of ≤{BATCH_SIZE} poses each")

t0 = time.time()
all_results = []
status_counts = {"ok":0, "timeout":0, "no_mol":0, "no_receptor":0}
# Write header to partial CSV
with open(PARTIAL_CSV, "w") as f:
    f.write("cand_id,receptor,energy,tanimoto,n_contacts,status\n")

with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
    # Submit in chunks so we can write progress incrementally
    chunks_done = 0
    chunk_size_workers = N_WORKERS    # do N_WORKERS batches at a time
    for chunk_start in range(0, len(batches), chunk_size_workers):
        chunk = batches[chunk_start:chunk_start+chunk_size_workers]
        chunk_args = [(b, receptors_meta) for b in chunk]
        chunk_results = list(ex.map(worker_plif_batch, chunk_args))
        for batch_results in chunk_results:
            all_results.extend(batch_results)
            with open(PARTIAL_CSV, "a") as f:
                for r in batch_results:
                    status_counts[r[5] if r[5] in status_counts else "ok"] = status_counts.get(r[5] if r[5] in status_counts else "ok", 0) + 1
                    f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}\n")
        chunks_done += len(chunk)
        elapsed = time.time() - t0
        print(f"  batches {chunks_done}/{len(batches)} done  ({len(all_results)}/{len(winning)} poses)  elapsed {elapsed:.0f}s  status: {dict(status_counts)}")

print(f"\n  TOTAL PLIF: {len(all_results)} poses in {time.time()-t0:.0f}s")
print(f"  status breakdown: {dict(status_counts)}")


# ───────────────────────── aggregate scoring chains ───────────────────────────
print(f"\n[step 3] aggregating scoring chains...")
lookup = {(c, r): (e, t, n) for c, r, e, t, n, _ in all_results}

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
    if not we_scores: continue
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

print(f"\n══════ Spearman ρ ══════")
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

print(f"\n══════ PLIF stats ══════")
print(df[["plif_1xq8","plif_we_mean"]].describe().to_string(float_format=lambda x: f"{x:.3f}"))

print(f"\n[done] total {time.time()-t0:.0f}s")
