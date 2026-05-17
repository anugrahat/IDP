#!/usr/bin/env python3
"""
100-seed replication of the 3-condition docking control (Task #26).

For each of the 3 conditions × 100 independent Vina seeds × 20 poses each,
compute hit rate to the Y133/D135/Y136 binding triad and pose distance to
the binding site centroid.

Output: control_100seed_results.csv + control_100seed_summary.txt
Runtime: ~5-10 min on 64 CPUs (32 parallel workers).
"""
import os, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd

ROOT = Path("/home/anugraha/IDP/docking")
N_SEEDS = 100
N_POSES_PER_SEED = 20
N_WORKERS = int(os.environ.get("CONTROL_WORKERS", "32"))

LIGAND = ROOT / "ligand/fasudil.pdbqt"
RECEPTOR_1XQ8 = ROOT / "receptors/A_1xq8.pdbqt"
RECEPTOR_1XQ8_PDB = ROOT / "receptors/A_1xq8.pdb"
ENSEMBLE_DIR = ROOT / "receptor_ensemble"
TARGET_RESIDS = {133, 135, 136}
HIT_DIST = 5.0

def get_residues_xyz(pdb, target_resids):
    """Return dict resid → ndarray (n, 3) of heavy-atom coords."""
    pts = {r: [] for r in target_resids}
    for line in open(pdb):
        if not line.startswith("ATOM"): continue
        atom = line[12:16].strip()
        if atom.startswith("H"): continue
        r = int(line[22:26])
        if r in target_resids:
            pts[r].append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return {r: np.array(v) for r, v in pts.items() if v}

def get_centroid(pdb, resids):
    pts = []
    for line in open(pdb):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            if int(line[22:26]) in resids: pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(pts).mean(axis=0).tolist()

def get_extent(pdb, cap=None):
    """Return (center, box_size) covering the protein with +10 Å buffer.
    For A_blind on elongated 1XQ8 we MUST NOT cap (would clip the C-terminal
    binding site, which is ~80 Å from COM along the long axis)."""
    pts = []
    for line in open(pdb):
        if line.startswith("ATOM"):
            pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    pts = np.array(pts)
    extent = pts.max(axis=0) - pts.min(axis=0) + 10
    if cap is not None:
        extent = np.minimum(extent, cap)
    return pts.mean(axis=0).tolist(), extent.tolist()

def parse_pose_atoms_from_pdbqt(text):
    """Return list of poses, each a (N,3) ndarray of heavy-atom coords."""
    poses = []
    curr = []
    for line in text.splitlines():
        if line.startswith("MODEL"):
            curr = []
        elif line.startswith("ENDMDL"):
            if curr: poses.append(np.array(curr))
            curr = []
        elif line.startswith(("ATOM","HETATM")):
            elem = line[76:78].strip()
            if elem and elem[0] != "H":
                curr.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if curr: poses.append(np.array(curr))
    return poses

def dock_one_seed(args):
    cond_name, receptor_pdbqt, receptor_pdb, center, box_size, seed, out_dir = args
    from vina import Vina
    out_pdbqt = out_dir / f"{cond_name}_seed{seed:03d}.pdbqt"
    try:
        v = Vina(sf_name="vina", cpu=2, verbosity=0, seed=seed)
        v.set_receptor(str(receptor_pdbqt))
        v.set_ligand_from_file(str(LIGAND))
        v.compute_vina_maps(center=center, box_size=box_size)
        v.dock(exhaustiveness=8, n_poses=N_POSES_PER_SEED)
        v.write_poses(str(out_pdbqt), n_poses=N_POSES_PER_SEED, overwrite=True)
    except Exception as e:
        return (cond_name, seed, [], [], str(e))

    text = out_pdbqt.read_text()
    energies = [float(line.split()[3]) for line in text.splitlines()
                if line.startswith("REMARK VINA RESULT")]
    poses = parse_pose_atoms_from_pdbqt(text)
    # For each pose: hit?, distance to target centroid
    target_xyz = get_residues_xyz(receptor_pdb, TARGET_RESIDS)
    target_all = np.concatenate(list(target_xyz.values()), axis=0) if target_xyz else None
    centroid = np.array(get_centroid(receptor_pdb, TARGET_RESIDS))
    hits, dists = [], []
    for p in poses:
        if target_all is not None:
            mind = np.linalg.norm(p[:,None,:] - target_all[None,:,:], axis=2).min()
            hits.append(int(mind < HIT_DIST))
        else:
            hits.append(0)
        # COM-to-centroid distance
        com = p.mean(axis=0)
        dists.append(float(np.linalg.norm(com - centroid)))
    return (cond_name, seed, energies, hits, dists)

def main():
    out_dir = ROOT / "control_100seed_poses"
    out_dir.mkdir(exist_ok=True)

    # Define 3 conditions
    center_1xq8_c = get_centroid(RECEPTOR_1XQ8_PDB, TARGET_RESIDS)
    # NO CAP — must cover the full elongated 1XQ8 (~155 Å on y-axis)
    bbox_1xq8 = get_extent(RECEPTOR_1XQ8_PDB, cap=None)
    # Pick a single "canonical" WE receptor (highest-weight cluster) for the C_biased condition
    manifest = pd.read_csv(ENSEMBLE_DIR / "ensemble_manifest.csv").sort_values("weight", ascending=False)
    top_we = manifest.iloc[0]
    we_rec_pdb = ENSEMBLE_DIR / top_we["receptor"]
    we_rec_pdbqt = we_rec_pdb.with_suffix(".pdbqt")
    we_center = get_centroid(we_rec_pdb, TARGET_RESIDS)

    print(f"A_blind  receptor=1XQ8, box=whole protein center={bbox_1xq8[0]}")
    print(f"A_biased receptor=1XQ8, box=C-term centroid {center_1xq8_c}")
    print(f"C_biased receptor={top_we['receptor']} (weight {top_we['weight']:.3f}), box=C-term centroid {we_center}")

    BIAS_BOX = [22.0, 22.0, 22.0]
    conditions = [
        ("A_blind",  RECEPTOR_1XQ8, RECEPTOR_1XQ8_PDB, bbox_1xq8[0],   bbox_1xq8[1]),
        ("A_biased", RECEPTOR_1XQ8, RECEPTOR_1XQ8_PDB, center_1xq8_c,  BIAS_BOX),
        ("C_biased", we_rec_pdbqt,  we_rec_pdb,         we_center,      BIAS_BOX),
    ]

    jobs = []
    for cond_name, rec_pdbqt, rec_pdb, center, box in conditions:
        for seed in range(1, N_SEEDS+1):
            jobs.append((cond_name, rec_pdbqt, rec_pdb, center, box, seed, out_dir))
    print(f"\nTotal: {len(jobs)} dockings (3 conds × {N_SEEDS} seeds), {N_WORKERS} workers")

    t0 = time.time()
    with Pool(N_WORKERS) as p:
        results = p.map(dock_one_seed, jobs)
    print(f"Docking done in {time.time()-t0:.1f}s")

    # Aggregate
    rows = []
    for cond_name, seed, energies, hits, dists in results:
        for i, (e, h, d) in enumerate(zip(energies, hits, dists)):
            rows.append({"condition": cond_name, "seed": seed, "pose_rank": i+1,
                         "energy": e, "hit": h, "com_dist_to_site": d})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT/"control_100seed_results.csv", index=False)

    print("\n══════ HIT RATE per condition (95% Clopper-Pearson CI) ══════")
    from scipy.stats import beta
    for cond in ["A_blind","A_biased","C_biased"]:
        sub = df[df["condition"]==cond]
        n = len(sub); hits = int(sub["hit"].sum())
        rate = hits/n if n else 0
        # 95% CI
        lo = beta.ppf(0.025, hits, n-hits+1) if hits>0 else 0
        hi = beta.ppf(0.975, hits+1, n-hits) if hits<n else 1
        print(f"  {cond:<10}  n={n:5d} hits={hits:5d}  rate={rate*100:6.2f}%  CI=[{lo*100:5.2f}, {hi*100:5.2f}]%")

    print("\n══════ COM distance to Y133/D135/Y136 centroid (Å) ══════")
    print(df.groupby("condition")["com_dist_to_site"].describe()[["mean","std","min","50%","max"]])
    print("\n══════ Vina score (kcal/mol) ══════")
    print(df.groupby("condition")["energy"].describe()[["mean","std","min","50%","max"]])

    summary_path = ROOT/"control_100seed_summary.txt"
    summary_path.write_text(
        df.groupby("condition").agg(
            n=("hit","size"), hits=("hit","sum"),
            hit_rate=("hit","mean"),
            mean_dist=("com_dist_to_site","mean"),
            mean_E=("energy","mean")
        ).to_string()
    )
    print(f"\n[done] full results: control_100seed_results.csv  summary: {summary_path}")

if __name__ == "__main__":
    main()
