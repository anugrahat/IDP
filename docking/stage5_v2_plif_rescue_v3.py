#!/usr/bin/env python3
"""
PLIF rescue v3 — uses MULTIPROCESSING SPAWN (not fork) to avoid the
fork-after-import deadlock that hung v1 and v2.

With spawn, each worker process is a fresh Python interpreter that
re-imports rdkit / MDAnalysis / ProLIF from scratch. No locks are
inherited from the parent. SIGALRM timeouts (10 sec per pose) work
because the interpreter state is clean.

Architecture:
  1. Parent loads NOTHING heavy at top level — only stdlib + numpy/pandas
  2. Workers receive (batch, receptor_pdb_paths) and re-import + cache fresh
  3. Each PLIF call wrapped in SIGALRM 10-sec timeout
  4. Per-batch partial CSV save → crash-resistant
"""
import os, sys, json, time, multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# Force spawn — this MUST happen before any worker is created
mp.set_start_method("spawn", force=True)

ROOT = Path("/home/anugraha/IDP/docking")
POSES = ROOT / "stage5_v2_poses"
RECEPTOR_1XQ8_PDB = ROOT/"receptors/A_1xq8.pdb"
ENSEMBLE = ROOT/"receptor_ensemble"

ALL_INTER = ["Hydrophobic","HBDonor","HBAcceptor","PiStacking",
             "PiCation","CationPi","Anionic","Cationic","VdWContact"]
ALPHA = 3.0
N_WORKERS = int(os.environ.get("PLIF_WORKERS", "32"))
PLIF_TIMEOUT_SEC = 10
PARTIAL_CSV = ROOT/"stage5_v2_plif_partial.csv"


def vina_score(pose_pdbqt):
    for line in open(pose_pdbqt):
        if line.startswith("REMARK VINA RESULT"):
            return float(line.split()[3])
    return None


def worker_init_and_run(args):
    """
    Worker entry point. With spawn, this runs in a fresh Python interpreter.
    All heavy imports + receptor caching happen INSIDE here, so the parent's
    library state never gets inherited.
    """
    import signal
    from rdkit import Chem
    from meeko import PDBQTMolecule, RDKitMolCreate
    import MDAnalysis as mda
    import prolif as plf

    batch, receptor_paths, ref_plif_json = args
    # ref_plif_json = path to reference_plif.json (don't pass big dicts to spawn)
    ref_plif = json.loads(Path(ref_plif_json).read_text())

    def _alarm(signum, frame):
        raise TimeoutError("PLIF timed out")
    signal.signal(signal.SIGALRM, _alarm)

    # Build receptor cache fresh in this worker
    cache = {}
    for name, pdb_path in receptor_paths:
        try:
            u = mda.Universe(str(pdb_path))
            prot = u.select_atoms("protein")
            cache[name] = plf.Molecule.from_mda(prot)
        except Exception as e:
            print(f"  cache fail {name}: {e}", file=sys.stderr)

    results = []
    for cid, rname, pose_path in batch:
        energy = vina_score(pose_path)
        if rname not in cache:
            results.append((cid, rname, energy, 0.0, 0, "no_receptor"))
            continue
        signal.alarm(PLIF_TIMEOUT_SEC)
        try:
            pmol = PDBQTMolecule.from_file(pose_path)
            mols = RDKitMolCreate.from_pdbqt_mol(pmol)
            if not mols:
                results.append((cid, rname, energy, 0.0, 0, "no_mol")); continue
            lig = plf.Molecule.from_rdkit(mols[0])
            fp = plf.Fingerprint(ALL_INTER)
            fp.run_from_iterable([lig], cache[rname], progress=False)
            df = fp.to_dataframe()
            ints = set()
            if not df.empty:
                for col in df.columns:
                    if df[col].iloc[0]:
                        _, prot_res, itype = col
                        ints.add(f"{str(prot_res).split('.')[0]}|{itype}")
            ref_keys = set(ref_plif.keys())
            shared = ints & ref_keys
            only_ref = ref_keys - ints
            only_pose = ints - ref_keys
            num = sum(ref_plif[k] for k in shared)
            den = sum(ref_plif[k] for k in shared | only_ref) + len(only_pose)
            tani = num/den if den > 0 else 0.0
            results.append((cid, rname, energy, tani, len(ints), "ok"))
        except TimeoutError:
            results.append((cid, rname, energy, 0.0, 0, "timeout"))
        except Exception as e:
            results.append((cid, rname, energy, 0.0, 0, f"err:{type(e).__name__}"))
        finally:
            signal.alarm(0)
    return results


def main():
    MANIFEST = json.loads((ROOT/"candidates_v4_full/manifest.json").read_text())
    ENSEMBLE_MANIFEST = pd.read_csv(ENSEMBLE/"ensemble_manifest.csv")

    receptors_meta = [{"name":"1xq8", "pdb":str(RECEPTOR_1XQ8_PDB), "weight":1.0}]
    for _, row in ENSEMBLE_MANIFEST.iterrows():
        receptors_meta.append({"name":Path(row.receptor).stem,
                               "pdb":str(ENSEMBLE/row.receptor),
                               "weight":float(row.weight)})

    n_poses = sum(1 for _ in POSES.glob("cand*_var*_*.pdbqt"))
    print(f"[v3 spawn] poses on disk: {n_poses}")
    print(f"[v3 spawn] {len(receptors_meta)} receptors, {N_WORKERS} workers, {PLIF_TIMEOUT_SEC}s per-pose timeout")

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

    # Build batches
    BATCH_SIZE = 20
    batches = [winning[i:i+BATCH_SIZE] for i in range(0, len(winning), BATCH_SIZE)]
    print(f"\n[step 2] PLIF with spawn-method workers, {len(batches)} batches of ≤{BATCH_SIZE}...")

    receptor_paths = [(r["name"], r["pdb"]) for r in receptors_meta]
    ref_plif_json = str(ROOT/"plif/reference_plif.json")

    job_args = [(b, receptor_paths, ref_plif_json) for b in batches]

    # Initialize partial CSV
    with open(PARTIAL_CSV, "w") as f:
        f.write("cand_id,receptor,energy,tanimoto,n_contacts,status\n")

    t0 = time.time()
    all_results = []
    status_counts = {}
    # Use spawn context
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as ex:
        done = 0
        for batch_result in ex.map(worker_init_and_run, job_args):
            done += 1
            all_results.extend(batch_result)
            with open(PARTIAL_CSV, "a") as f:
                for r in batch_result:
                    status_counts[r[5]] = status_counts.get(r[5], 0) + 1
                    f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}\n")
            print(f"  batch {done}/{len(batches)} done ({len(all_results)}/{len(winning)} poses, {time.time()-t0:.0f}s)  status: {status_counts}")

    print(f"\n[plif] total {time.time()-t0:.0f}s")
    print(f"  status: {status_counts}")

    # Aggregate
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

    from scipy.stats import spearmanr
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


if __name__ == "__main__":
    main()
