#!/usr/bin/env python3
"""
WE-weighted conformational landscape — PARALLEL version.

Speed fix over pca_analysis.py: featurization is parallelized across CPUs.
Each worker loads its own MDAnalysis topology ONCE, then processes its
share of segments. Master thread waits for all workers, then runs PCA +
clustering + plots + PDB extraction (single-threaded since these are fast).

Run via run_pca.sbatch (64 CPUs).
"""
import os, sys, json, time
from pathlib import Path
from glob import glob
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

ROOT = Path("/home/anugraha/IDP/conformational_ensemble")
WE_ROOT = Path("/home/anugraha/IDP/westpa")
TOPOLOGY = "/home/anugraha/IDP/build/complex.prmtop"
SNAPSHOT = ROOT / "data/west_snapshot.h5"
DATA = ROOT / "data"; FIG = ROOT / "figures"; REPS = ROOT / "representatives"
for d in [DATA, FIG, REPS]: d.mkdir(exist_ok=True, parents=True)

STRIDE = 10
N_CLUSTERS = 10
CTERM_RES = list(range(96, 141))
N_WORKERS = int(os.environ.get("PCA_WORKERS", "32"))


def featurize_iteration(args):
    """One worker — process all segments of ONE iteration.
    Loads topology only once per worker thanks to MDAnalysis caching."""
    iter_dir, seg_weights_subset = args
    import MDAnalysis as mda
    u_top = mda.Universe(TOPOLOGY)
    cterm_ca_idx = [u_top.select_atoms(f"protein and resid {r} and name CA")[0].index for r in CTERM_RES]
    if len(cterm_ca_idx) != 45:
        return []

    it = int(Path(iter_dir).name)
    rows = []
    for seg_dir in sorted(glob(f"{iter_dir}/*")):
        sg = int(Path(seg_dir).name)
        seg_nc = Path(seg_dir) / "seg.nc"
        if not seg_nc.exists(): continue
        try:
            pc = np.loadtxt(f"{seg_dir}/pcoord.dat")
            rd = np.loadtxt(f"{seg_dir}/ring_dists.dat")
            ch = np.atleast_1d(np.loadtxt(f"{seg_dir}/d135_charge_dist.dat"))
            if pc.ndim == 1: pc = pc[None,:]
            if rd.ndim == 1: rd = rd[None,:]
        except Exception: continue
        try:
            u = mda.Universe(TOPOLOGY, str(seg_nc))
        except Exception: continue
        weight = seg_weights_subset.get((it, sg))
        if weight is None: continue
        weight_per_frame = weight / max(1, len(u.trajectory))
        for f_idx in range(0, len(u.trajectory), STRIDE):
            u.trajectory[f_idx]
            coords = u.atoms[cterm_ca_idx].positions
            centered = (coords - coords.mean(axis=0)).flatten()
            d_min = float(pc[f_idx, 0]) if f_idx < len(pc) else np.nan
            mr = float(rd[f_idx].min()) if (f_idx < len(rd) and rd.ndim==2) else np.nan
            dch = float(ch[f_idx]) if f_idx < len(ch) else np.nan
            bound = int(d_min < 4.5 and (mr < 6.0 or dch < 5.0)) if not np.isnan(d_min) else 0
            rows.append((it, sg, f_idx, weight_per_frame, bound, d_min, centered.astype(np.float32)))
    return rows


def main():
    # Step 1: weights
    print("[step 1] reading per-segment weights from snapshot...")
    seg_weights = {}
    with h5py.File(SNAPSHOT, "r") as f:
        for it_name in sorted(f["iterations"].keys()):
            it = int(it_name.split("_")[-1])
            for sg, w in enumerate(f["iterations"][it_name]["seg_index"]["weight"][:]):
                seg_weights[(it, sg)] = float(w)
    print(f"  {len(seg_weights)} segments")

    # Step 2: parallel featurization
    print(f"\n[step 2] parallel featurization across {N_WORKERS} workers...")
    traj_dirs = sorted(glob(str(WE_ROOT/"traj_segs/0*")))
    traj_dirs = [d for d in traj_dirs if int(Path(d).name) > 0]
    print(f"  iterations to process: {len(traj_dirs)}")

    # Per-worker subset of weights to keep payload small
    jobs = [(d, {k:v for k,v in seg_weights.items() if k[0] == int(Path(d).name)}) for d in traj_dirs]

    t0 = time.time()
    all_rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(featurize_iteration, j): j[0] for j in jobs}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                rows = fut.result()
                all_rows.extend(rows)
                if done % 10 == 0:
                    print(f"  {done}/{len(jobs)} iters processed, {len(all_rows)} frames so far ({time.time()-t0:.0f}s elapsed)")
            except Exception as e:
                print(f"  iter {futures[fut]} FAILED: {e}")
    print(f"  total: {len(all_rows)} frames in {time.time()-t0:.0f}s")

    if len(all_rows) == 0:
        print("BAIL — no frames featurized"); sys.exit(1)

    # Assemble
    meta = pd.DataFrame([(r[0],r[1],r[2],r[3],r[4],r[5]) for r in all_rows],
                        columns=["iter","seg","frame","weight","bound","d_min"])
    X = np.stack([r[6] for r in all_rows])
    print(f"  feature matrix: {X.shape}")
    print(f"  weighted sum: {meta['weight'].sum():.2f}")
    print(f"  bound fraction (weighted): {(meta['bound']*meta['weight']).sum()/meta['weight'].sum():.3f}")
    np.save(DATA/"features.npy", X)
    meta.to_csv(DATA/"frame_meta.csv", index=False)
    print(f"  ✓ saved")

    # Step 3: PCA
    print(f"\n[step 3] PCA...")
    pca = PCA(n_components=10)
    Z = pca.fit_transform(X)
    print(f"  variance ratio (first 5): {[f'{v:.3f}' for v in pca.explained_variance_ratio_[:5]]}")
    print(f"  cumulative through PC5: {pca.explained_variance_ratio_[:5].sum():.3f}")
    np.savez(DATA/"pca.npz", components=pca.components_, mean=pca.mean_,
             explained_variance_ratio=pca.explained_variance_ratio_, Z=Z)
    for k in range(3):
        meta[f"PC{k+1}"] = Z[:,k]
    meta.to_csv(DATA/"frame_meta.csv", index=False)

    # Scree
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(range(1,11), pca.explained_variance_ratio_*100)
    ax.set_xlabel("PC"); ax.set_ylabel("% variance"); ax.set_title("Scree — α-syn C-term (96-140) Cα PCA")
    fig.tight_layout(); fig.savefig(FIG/"pca_variance.png", dpi=140); plt.close(fig)

    # Step 4: weighted KDE
    print(f"\n[step 4] weighted 2D landscape...")
    w = meta["weight"].values
    w_norm = w / w.sum()
    kde = gaussian_kde(np.vstack([Z[:,0], Z[:,1]]), weights=w_norm, bw_method=0.3)
    xi = np.linspace(Z[:,0].min(), Z[:,0].max(), 100)
    yi = np.linspace(Z[:,1].min(), Z[:,1].max(), 100)
    XX, YY = np.meshgrid(xi, yi)
    ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    F = -np.log(ZZ + 1e-30); F -= F.min()

    fig, ax = plt.subplots(figsize=(7,6))
    c = ax.contourf(XX, YY, F, levels=20, cmap="viridis_r")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("α-syn C-term conformational free-energy landscape (WE-weighted)")
    plt.colorbar(c, label="ΔG (kT)")
    fig.tight_layout(); fig.savefig(FIG/"landscape_2d.png", dpi=140); plt.close(fig)

    # Bound overlay
    fig, ax = plt.subplots(figsize=(7,6))
    ax.contourf(XX, YY, F, levels=20, cmap="Greys")
    bm = meta["bound"] == 1
    ax.scatter(meta.loc[~bm,"PC1"], meta.loc[~bm,"PC2"], c="lightblue", s=2, alpha=0.3, label=f"unbound (n={(~bm).sum()})")
    ax.scatter(meta.loc[bm,"PC1"], meta.loc[bm,"PC2"], c="red", s=4, alpha=0.6, label=f"bound (n={bm.sum()})")
    ax.legend()
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Landscape with bound (red) vs unbound (light blue)")
    fig.tight_layout(); fig.savefig(FIG/"landscape_bound_vs_unbound.png", dpi=140); plt.close(fig)
    print(f"  ✓ landscape figures saved")

    # Step 5: clustering
    print(f"\n[step 5] weighted k-means clustering (k={N_CLUSTERS})...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(Z[:, :4], sample_weight=w_norm)
    meta["cluster"] = labels

    cluster_stats = []
    print(f"\n  {'cluster':>7}  {'n_frames':>9}  {'sum_weight':>11}  {'pop%':>5}  {'bound%':>7}")
    for k in range(N_CLUSTERS):
        mk = labels == k
        wt = float(w[mk].sum())
        b = float((meta.loc[mk,"bound"] * w[mk]).sum() / wt) if wt > 0 else 0
        cluster_stats.append({"cluster":k, "n_frames":int(mk.sum()), "sum_weight":wt,
                              "pop_fraction":wt/w.sum(), "bound_fraction":b})
        print(f"  {k:>7}  {int(mk.sum()):>9}  {wt:>11.3e}  {wt/w.sum()*100:>5.1f}  {b*100:>6.2f}")
    pd.DataFrame(cluster_stats).to_csv(DATA/"cluster_stats.csv", index=False)
    meta.to_csv(DATA/"frame_meta.csv", index=False)

    fig, ax = plt.subplots(figsize=(7,6))
    for k in range(N_CLUSTERS):
        mk = labels == k
        ax.scatter(Z[mk,0], Z[mk,1], s=4, alpha=0.5, label=f"C{k} ({cluster_stats[k]['pop_fraction']*100:.1f}%)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"Weighted k-means (k={N_CLUSTERS})")
    fig.tight_layout(); fig.savefig(FIG/"clusters_in_PC.png", dpi=140); plt.close(fig)

    # Step 6: medoid PDBs
    print(f"\n[step 6] extracting medoid PDBs...")
    import subprocess
    CPPTRAJ = "/software/amber/24/ucdhpc-20.04/ambertools25/bin/cpptraj"
    extract_script = REPS/"extract.cpptraj"
    manifest = []
    with open(extract_script,"w") as cf:
        cf.write(f"parm {TOPOLOGY}\n")
        for k in range(N_CLUSTERS):
            mk = labels == k
            if not mk.any(): continue
            idx_in = np.where(mk)[0]
            centroid = Z[mk,:4].mean(axis=0)
            med = idx_in[np.argmin(np.linalg.norm(Z[idx_in,:4] - centroid, axis=1))]
            r = meta.iloc[med]
            traj = WE_ROOT / f"traj_segs/{int(r['iter']):06d}/{int(r['seg']):06d}/seg.nc"
            cf.write(f"trajin {traj} {int(r['frame'])+1} {int(r['frame'])+1} 1\n")
            manifest.append({"cluster":k, "iter":int(r['iter']), "seg":int(r['seg']),
                             "frame":int(r['frame']),
                             "PC1":float(r['PC1']), "PC2":float(r['PC2']),
                             "weight":float(r['weight']), "bound":int(r['bound']),
                             "cluster_pop_pct":float(cluster_stats[k]['pop_fraction'])*100,
                             "cluster_bound_pct":float(cluster_stats[k]['bound_fraction'])*100})
        cf.write(f"autoimage :1-140,FAS firstatom\nstrip :WAT,Na+,Cl-,K+\n")
        cf.write(f"trajout {REPS}/cluster.pdb pdb multi\nrun\nquit\n")

    res = subprocess.run([CPPTRAJ, "-i", str(extract_script)], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"cpptraj FAILED: {res.stderr[-500:]}"); sys.exit(1)
    pdbs = sorted(REPS.glob("cluster.pdb.*"), key=lambda p: int(p.suffix[1:]))
    for i, p in enumerate(pdbs):
        new = REPS / f"cluster_{manifest[i]['cluster']:02d}.pdb"
        p.rename(new); manifest[i]["pdb"] = new.name
    pd.DataFrame(manifest).to_csv(REPS/"manifest.csv", index=False)
    print(f"  ✓ extracted {len(manifest)} cluster medoids")

    # Step 7: per-cluster contacts
    print(f"\n[step 7] per-cluster contact frequency...")
    cm_per_c = defaultdict(lambda: np.zeros(140))
    w_per_c = defaultdict(float)
    for row in meta.itertuples():
        cm_path = WE_ROOT / f"traj_segs/{row.iter:06d}/{row.seg:06d}/contact_map.dat"
        if not cm_path.exists(): continue
        try:
            cm = np.loadtxt(cm_path)
            if cm.ndim == 1: cm = cm[None,:]
        except Exception: continue
        if row.frame >= len(cm): continue
        contact = (cm[row.frame] < 5.0).astype(float)
        cm_per_c[row.cluster] += contact * row.weight
        w_per_c[row.cluster] += row.weight
    for k in cm_per_c:
        if w_per_c[k] > 0: cm_per_c[k] /= w_per_c[k]

    fig, axes = plt.subplots(N_CLUSTERS, 1, figsize=(11, N_CLUSTERS*0.6), sharex=True)
    for k in range(N_CLUSTERS):
        ax = axes[k]
        freq = cm_per_c.get(k, np.zeros(140))
        ax.bar(range(1,141), freq, color="C0" if cluster_stats[k]['bound_fraction']>0.5 else "lightgray")
        for r in [125, 133, 135, 136, 130, 137]:
            if freq[r-1] > 0.1:
                ax.axvline(r, color="red", alpha=0.3, linewidth=0.5)
        ax.set_ylim(0, max(0.2, freq.max()*1.1))
        ax.set_ylabel(f"C{k}\np{cluster_stats[k]['pop_fraction']*100:.0f}", rotation=0, fontsize=7, labelpad=20)
        ax.set_yticks([])
    axes[-1].set_xlabel("α-syn residue number")
    axes[0].set_title("Per-cluster fasudil-contact frequency (weighted; red lines: binding-site residues)")
    fig.tight_layout(); fig.savefig(FIG/"per_cluster_contacts.png", dpi=140); plt.close(fig)
    print(f"  ✓ saved figures/per_cluster_contacts.png")

    print(f"\n[done] all outputs in {ROOT}")


if __name__ == "__main__":
    main()
