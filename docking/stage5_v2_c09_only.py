#!/usr/bin/env python3
"""
Ablation variant: use ONLY the PCA cluster_09 medoid as the WE-aware
receptor, instead of the 10 contact-map medoids. Compare ρ(SC1, SC2)
to see if a single-conformation WE receptor sharpens the signal.

Reuses 360 v4 ligand variants. Docks each into cluster_09 once, then
computes PLIF + scoring chains.
"""
import os, json, sys, time, multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutTimeout

mp.set_start_method("spawn", force=True)

ROOT = Path("/home/anugraha/IDP/docking")
POSES_DIR = ROOT / "c09_poses"
POSES_DIR.mkdir(exist_ok=True)
RECEPTOR_PDB = ROOT/"c09_receptor.pdb"
RECEPTOR_PDBQT = ROOT/"c09_receptor.pdbqt"
MANIFEST = json.loads((ROOT/"candidates_v4_full/manifest.json").read_text())
REF_PLIF = json.loads((ROOT/"plif/reference_plif.json").read_text())
STAGE5_RESULTS = ROOT/"stage5_v2_results.csv"   # has SC1 from 1XQ8

ALL_INTER = ["Hydrophobic","HBDonor","HBAcceptor","PiStacking",
             "PiCation","CationPi","Anionic","Cationic","VdWContact"]
ALPHA = 3.0
N_WORKERS = int(os.environ.get("PLIF_WORKERS", "32"))
BOX_SIZE = [22.0, 22.0, 22.0]


def get_box_center(pdb, resids={133,135,136}):
    pts = []
    for line in open(pdb):
        if line.startswith("ATOM") and line[12:16].strip()=="CA" and int(line[22:26]) in resids:
            pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(pts).mean(axis=0).tolist()


def vina_score(pose_pdbqt):
    for line in open(pose_pdbqt):
        if line.startswith("REMARK VINA RESULT"):
            return float(line.split()[3])
    return None


def dock_one(args):
    """Dock a single variant into cluster_09."""
    for k in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
        os.environ[k] = "1"
    cid, vid, lig_pdbqt, center, out_path = args
    from vina import Vina
    try:
        v = Vina(sf_name="vina", cpu=2, verbosity=0, seed=42)
        v.set_receptor(str(RECEPTOR_PDBQT))
        v.set_ligand_from_file(lig_pdbqt)
        v.compute_vina_maps(center=center, box_size=BOX_SIZE)
        v.dock(exhaustiveness=8, n_poses=3)
        v.write_poses(out_path, n_poses=3, overwrite=True)
        e = vina_score(out_path)
        return (cid, vid, e, None)
    except Exception as exc:
        return (cid, vid, None, str(exc)[:200])


def plif_batch(args):
    for k in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
        os.environ[k] = "1"
    batch, receptor_pdb, ref_plif_json = args
    import signal
    from rdkit import Chem
    from meeko import PDBQTMolecule, RDKitMolCreate
    import MDAnalysis as mda
    import prolif as plf
    ref_plif = json.loads(Path(ref_plif_json).read_text())
    def _alarm(s,f): raise TimeoutError()
    signal.signal(signal.SIGALRM, _alarm)
    u = mda.Universe(str(receptor_pdb))
    prot = u.select_atoms("protein")
    prot_mol = plf.Molecule.from_mda(prot)
    results = []
    for cid, pose_path in batch:
        e = vina_score(pose_path)
        signal.alarm(15)
        try:
            pmol = PDBQTMolecule.from_file(pose_path)
            mols = RDKitMolCreate.from_pdbqt_mol(pmol)
            if not mols: results.append((cid, e, 0.0, 0, "no_mol")); continue
            lig = plf.Molecule.from_rdkit(mols[0])
            fp = plf.Fingerprint(ALL_INTER)
            try:
                fp.run_from_iterable([lig], prot_mol, n_jobs=1, progress=False)
            except TypeError:
                fp.run_from_iterable([lig], prot_mol, progress=False)
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
            den = sum(ref_plif[k] for k in shared|only_ref) + len(only_pose)
            tani = num/den if den>0 else 0.0
            results.append((cid, e, tani, len(ints), "ok"))
        except TimeoutError:
            results.append((cid, e, 0.0, 0, "timeout"))
        except Exception as ex:
            results.append((cid, e, 0.0, 0, f"err:{type(ex).__name__}"))
        finally:
            signal.alarm(0)
    return results


def main():
    # Pre-flight: box geometry
    center = get_box_center(RECEPTOR_PDB)
    half = np.array(BOX_SIZE)/2
    y133 = None
    for line in open(RECEPTOR_PDB):
        if line.startswith("ATOM") and line[12:16].strip()=="CA" and int(line[22:26])==133:
            y133 = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            break
    inside = all(abs(y133[i] - center[i]) <= half[i] for i in range(3))
    print(f"[c09] center={[round(x,1) for x in center]}  Y133={y133.round(1)}  inside={inside}")
    assert inside, "BAIL — Y133 outside box"

    # Build dock jobs (one per variant)
    dock_jobs = []
    for entry in MANIFEST:
        pid = entry["cand_idx"]
        for v in entry["variants"]:
            out = POSES_DIR / f"cand{pid:03d}_var{v['var_idx']:02d}.pdbqt"
            dock_jobs.append((pid, v["var_idx"], v["pdbqt"], center, str(out)))
    print(f"[c09] {len(dock_jobs)} variants to dock into cluster_09")

    # DOCK
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as ex:
        results = list(ex.map(dock_one, dock_jobs))
    print(f"[dock] done in {time.time()-t0:.0f}s, errors: {sum(1 for r in results if r[3])}")

    e_lookup = {(c,v): e for c,v,e,err in results}

    # Identify winning variant per parent
    winning = []
    for entry in MANIFEST:
        pid = entry["cand_idx"]
        best_e = np.inf; best_v = None
        for v in entry["variants"]:
            e = e_lookup.get((pid, v["var_idx"]))
            if e is not None and e < best_e:
                best_e = e; best_v = v["var_idx"]
        if best_v is not None:
            pose = str(POSES_DIR / f"cand{pid:03d}_var{best_v:02d}.pdbqt")
            winning.append((pid, pose))
    print(f"[winning] {len(winning)} per-parent winners")

    # PLIF
    BATCH_SIZE = 20
    batches = [winning[i:i+BATCH_SIZE] for i in range(0, len(winning), BATCH_SIZE)]
    ref_plif_json = str(ROOT/"plif/reference_plif.json")
    t1 = time.time()
    print(f"[plif] {len(batches)} batches, spawn workers, n_jobs=1")
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as ex:
        plif_chunks = list(ex.map(plif_batch, [(b, str(RECEPTOR_PDB), ref_plif_json) for b in batches]))
    plif_results = [r for chunk in plif_chunks for r in chunk]
    print(f"[plif] done in {time.time()-t1:.0f}s")
    plif_lookup = {cid: (e, t, n) for cid, e, t, n, _ in plif_results}

    # Build c09 dataframe + merge with existing SC1
    rows = []
    for pid, info in plif_lookup.items():
        rows.append({"id": pid, "score_c09": info[0], "plif_c09": info[1]})
    df_c09 = pd.DataFrame(rows)
    df_old = pd.read_csv(STAGE5_RESULTS)
    df = df_old.merge(df_c09, on="id", how="inner")
    df["SC2_c09"] = -df["score_c09"]
    df["SC4_c09"] = df["SC2_c09"] + ALPHA * df["plif_c09"]
    df.to_csv(ROOT/"stage5_v2_c09_results.csv", index=False)
    print(f"\n[results] {len(df)} candidates → stage5_v2_c09_results.csv")

    # Compare
    from scipy.stats import spearmanr
    print(f"\n══════ Spearman ρ (CONTACT-MAP 10 receptors vs PCA C09 single) ══════")
    print(f"  SC1 (1xq8)      vs SC2 (10 WE)   : ρ = {spearmanr(df.SC1, df.SC2)[0]:+.3f}")
    print(f"  SC1 (1xq8)      vs SC2_c09 (PCA): ρ = {spearmanr(df.SC1, df.SC2_c09)[0]:+.3f}")
    print(f"  SC1 (1xq8)      vs SC4 (10 WE+PLIF): ρ = {spearmanr(df.SC1, df.SC4)[0]:+.3f}")
    print(f"  SC1 (1xq8)      vs SC4_c09 (PCA+PLIF): ρ = {spearmanr(df.SC1, df.SC4_c09)[0]:+.3f}")
    print(f"  SC2 (10 WE)    vs SC2_c09 (PCA): ρ = {spearmanr(df.SC2, df.SC2_c09)[0]:+.3f}")
    print(f"  SC4 (10 WE+PLIF) vs SC4_c09 (PCA+PLIF): ρ = {spearmanr(df.SC4, df.SC4_c09)[0]:+.3f}")

    print(f"\n══════ Top-5 overlap ══════")
    chains = ["SC1","SC2","SC2_c09","SC4","SC4_c09"]
    for a in chains:
        for b in chains:
            if a >= b: continue
            ta = set(df.nlargest(5, a)["id"].tolist())
            tb = set(df.nlargest(5, b)["id"].tolist())
            print(f"  {a:<10} ∩ {b:<10} top-5: {len(ta & tb)}/5")

    print(f"\n══════ PLIF stats: 10-rec vs c09 ══════")
    print(df[["plif_we_mean","plif_c09"]].describe().to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
