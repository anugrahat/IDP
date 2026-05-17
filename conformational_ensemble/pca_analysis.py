#!/usr/bin/env python3
"""
Full WE-weighted conformational landscape of α-syn from the WE-MD run.

Pipeline:
  1. Snapshot west.h5 → read per-segment weights (already done)
  2. Stratified sample of frames from all segments
  3. Compute features: C-term (96-140) Cα coordinates → 135-D vectors
  4. Standardize, PCA → project all frames onto first 4 PCs
  5. Weighted 2D KDE on (PC1, PC2) → free-energy landscape
  6. Weighted k-medoids cluster (k=10) → representative conformations
  7. For each cluster: extract medoid PDB, compute per-residue fasudil-contact frequency
  8. Save: features, PCA loadings, cluster manifest, plots, representative PDBs

Verify each output before chaining (per memory rule).
"""
import os, sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
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

STRIDE = 10                 # 1 frame per N frames from each segment
N_CLUSTERS = 10
CTERM_RES = list(range(96, 141))   # residues 96-140 (C-term region where fasudil binds)
RNG = np.random.default_rng(42)


# ─────────────────────── Step 1: read WE weights from snapshot ────────────────
print("[step 1] reading per-segment weights from snapshot...")
seg_weights = {}     # (iter, seg) → weight
with h5py.File(SNAPSHOT, "r") as f:
    for it_name in sorted(f["iterations"].keys()):
        it = int(it_name.split("_")[-1])
        weights = f["iterations"][it_name]["seg_index"]["weight"][:]
        for sg, w in enumerate(weights):
            seg_weights[(it, sg)] = float(w)
print(f"  {len(seg_weights)} segments × weight values loaded")
print(f"  weight range: [{min(seg_weights.values()):.2e}, {max(seg_weights.values()):.2e}]")


# ────────────── Step 2: stratified sample of frames + compute features ──────────
print(f"\n[step 2] sampling frames (stride {STRIDE}) and featurizing...")

from glob import glob
traj_dirs = sorted(glob(str(WE_ROOT/"traj_segs/0*")))
print(f"  found {len(traj_dirs)} iteration dirs")

# We need a topology Universe to know atom selection
u_top = mda.Universe(TOPOLOGY)
cterm_ca_idx = []
for resid in CTERM_RES:
    sel = u_top.select_atoms(f"protein and resid {resid} and name CA")
    if len(sel) == 1:
        cterm_ca_idx.append(sel[0].index)
print(f"  C-term Cα atoms: {len(cterm_ca_idx)} (residues {CTERM_RES[0]}-{CTERM_RES[-1]})")
assert len(cterm_ca_idx) == 45, f"BAIL — expected 45 C-term CAs, got {len(cterm_ca_idx)}"

# Stride frames + compute features
all_features = []      # each row = 135-D Cα coordinates
all_meta = []          # (iter, seg, frame, weight, bound)
n_skipped = 0
for it_dir in traj_dirs:
    it = int(Path(it_dir).name)
    if it == 0: continue
    for seg_dir in sorted(glob(f"{it_dir}/*")):
        sg = int(Path(seg_dir).name)
        seg_nc = Path(seg_dir) / "seg.nc"
        if not seg_nc.exists():
            n_skipped += 1; continue
        # Load auxiliary data to get bound flag per frame
        try:
            pc = np.loadtxt(f"{seg_dir}/pcoord.dat")
            rd = np.loadtxt(f"{seg_dir}/ring_dists.dat")
            ch = np.atleast_1d(np.loadtxt(f"{seg_dir}/d135_charge_dist.dat"))
            if pc.ndim == 1: pc = pc[None, :]
            if rd.ndim == 1: rd = rd[None, :]
        except Exception:
            n_skipped += 1; continue

        # Load traj
        try:
            u = mda.Universe(TOPOLOGY, str(seg_nc))
        except Exception:
            n_skipped += 1; continue

        weight = seg_weights.get((it, sg), None)
        if weight is None:
            n_skipped += 1; continue
        weight_per_frame = weight / max(1, len(u.trajectory))

        # Stride through frames
        for f_idx in range(0, len(u.trajectory), STRIDE):
            u.trajectory[f_idx]
            coords = u.atoms[cterm_ca_idx].positions.flatten()
            # Center on Cα COM (translation-invariant)
            c_mean = coords.reshape(-1,3).mean(axis=0)
            centered = (coords.reshape(-1,3) - c_mean).flatten()
            all_features.append(centered)
            # Bound flag
            if f_idx < len(pc):
                d_min = float(pc[f_idx, 0])
                mr = float(rd[f_idx].min()) if rd.ndim==2 else float(rd[f_idx])
                dch = float(ch[f_idx]) if f_idx < len(ch) else np.nan
                bound = int(d_min < 4.5 and (mr < 6.0 or dch < 5.0))
            else:
                bound = 0
                d_min = np.nan
            all_meta.append((it, sg, f_idx, weight_per_frame, bound, d_min))

print(f"  skipped {n_skipped} bad segments")
print(f"  features extracted from {len(all_features)} frames")
assert len(all_features) > 100, "BAIL — too few frames"

X = np.array(all_features, dtype=np.float32)
meta = pd.DataFrame(all_meta, columns=["iter","seg","frame","weight","bound","d_min"])
print(f"  feature matrix shape: {X.shape}  (frames × 135 coords)")
print(f"  any NaN in X? {np.isnan(X).any()}")
print(f"  weighted sum (should ≈ N_iter): {meta['weight'].sum():.2f}")
print(f"  bound fraction (weighted): {(meta['bound'] * meta['weight']).sum() / meta['weight'].sum():.3f}")
np.save(DATA/"features.npy", X)
meta.to_csv(DATA/"frame_meta.csv", index=False)
print(f"  ✓ saved features.npy, frame_meta.csv")


# ─────────────────────── Step 3: PCA + projection ──────────────────────────────
print(f"\n[step 3] PCA on Cα coords (no weights — PCA is sample covariance)...")
pca = PCA(n_components=10)
Z = pca.fit_transform(X)
print(f"  explained variance (first 5 PCs): {[f'{v:.3f}' for v in pca.explained_variance_ratio_[:5]]}")
print(f"  cumulative variance through PC5: {pca.explained_variance_ratio_[:5].sum():.3f}")
np.savez(DATA/"pca.npz", components=pca.components_, mean=pca.mean_,
         explained_variance_ratio=pca.explained_variance_ratio_, Z=Z)
meta["PC1"] = Z[:,0]; meta["PC2"] = Z[:,1]; meta["PC3"] = Z[:,2]
meta.to_csv(DATA/"frame_meta.csv", index=False)
print(f"  ✓ saved pca.npz, updated frame_meta.csv")

# scree plot
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(range(1,11), pca.explained_variance_ratio_*100)
ax.set_xlabel("PC")
ax.set_ylabel("% variance explained")
ax.set_title("Scree plot — α-syn C-term (96-140) Cα PCA")
fig.tight_layout(); fig.savefig(FIG/"pca_variance.png", dpi=140); plt.close(fig)
print(f"  ✓ saved figures/pca_variance.png")


# ─────────────────────── Step 4: weighted KDE landscape ──────────────────────
print(f"\n[step 4] weighted 2D KDE on (PC1, PC2)...")
w = meta["weight"].values
w_norm = w / w.sum()
kde = gaussian_kde(np.vstack([Z[:,0], Z[:,1]]), weights=w_norm, bw_method=0.3)
xi = np.linspace(Z[:,0].min(), Z[:,0].max(), 80)
yi = np.linspace(Z[:,1].min(), Z[:,1].max(), 80)
XX, YY = np.meshgrid(xi, yi)
ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
# Free energy: -kT ln(P)
F = -np.log(ZZ + 1e-30); F = F - F.min()

fig, ax = plt.subplots(figsize=(7, 6))
c = ax.contourf(XX, YY, F, levels=20, cmap="viridis_r")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_title("α-syn C-term conformational free-energy landscape (WE-weighted)")
cbar = plt.colorbar(c); cbar.set_label("ΔG (kT)")
fig.tight_layout(); fig.savefig(FIG/"landscape_2d.png", dpi=140); plt.close(fig)
print(f"  ✓ saved figures/landscape_2d.png")

# Overlay bound frames
fig, ax = plt.subplots(figsize=(7, 6))
ax.contourf(XX, YY, F, levels=20, cmap="Greys")
bm = meta["bound"] == 1
ax.scatter(meta.loc[~bm, "PC1"], meta.loc[~bm, "PC2"], c="lightblue", s=2, alpha=0.3, label=f"unbound (n={(~bm).sum()})")
ax.scatter(meta.loc[bm, "PC1"], meta.loc[bm, "PC2"], c="red", s=4, alpha=0.6, label=f"bound (n={bm.sum()})")
ax.legend()
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_title("α-syn C-term landscape — bound (red) vs unbound (light blue)")
fig.tight_layout(); fig.savefig(FIG/"landscape_bound_vs_unbound.png", dpi=140); plt.close(fig)
print(f"  ✓ saved figures/landscape_bound_vs_unbound.png")


# ─────────────────── Step 5: weighted k-means clustering ───────────────────
print(f"\n[step 5] weighted k-means clustering (k={N_CLUSTERS}) in PC1-PC4 space...")
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
labels = km.fit_predict(Z[:, :4], sample_weight=w_norm)
meta["cluster"] = labels
# Cluster summary
print(f"\n  {'cluster':>7}  {'n_frames':>9}  {'sum_weight':>11}  {'%pop':>5}  {'bound%':>7}")
cluster_stats = []
for k in range(N_CLUSTERS):
    mk = labels == k
    n_frm = int(mk.sum())
    wt = float(w[mk].sum())
    bound_frac = float((meta.loc[mk, "bound"] * w[mk]).sum() / wt) if wt > 0 else 0
    cluster_stats.append({"cluster": k, "n_frames": n_frm, "sum_weight": wt,
                          "pop_fraction": wt/w.sum(), "bound_fraction": bound_frac})
    print(f"  {k:>7}  {n_frm:>9}  {wt:>11.3e}  {wt/w.sum()*100:>5.1f}  {bound_frac*100:>6.2f}")
pd.DataFrame(cluster_stats).to_csv(DATA/"cluster_stats.csv", index=False)
meta.to_csv(DATA/"frame_meta.csv", index=False)
print(f"  ✓ saved cluster_stats.csv, frame_meta.csv")

# Plot clusters in PC space
fig, ax = plt.subplots(figsize=(7, 6))
for k in range(N_CLUSTERS):
    mk = labels == k
    ax.scatter(Z[mk, 0], Z[mk, 1], s=4, alpha=0.5, label=f"C{k} (pop {w[mk].sum()/w.sum()*100:.1f}%)")
ax.legend(fontsize=8, ncol=2, loc='best')
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title(f"Weighted k-means clusters (k={N_CLUSTERS})")
fig.tight_layout(); fig.savefig(FIG/"clusters_in_PC.png", dpi=140); plt.close(fig)
print(f"  ✓ saved figures/clusters_in_PC.png")


# ─────────────────── Step 6: pick representative frame per cluster ──────────
print(f"\n[step 6] extracting medoid frames per cluster as PDB...")
# Medoid = frame closest to cluster centroid in PC1-PC4
import subprocess
CPPTRAJ = "/software/amber/24/ucdhpc-20.04/ambertools25/bin/cpptraj"
extract_script = REPS/"extract.cpptraj"
manifest = []
with open(extract_script, "w") as cf:
    cf.write(f"parm {TOPOLOGY}\n")
    for k in range(N_CLUSTERS):
        mk = labels == k
        if not mk.any(): continue
        idx_in_cluster = np.where(mk)[0]
        centroid = Z[mk, :4].mean(axis=0)
        # Closest frame to centroid
        dists = np.linalg.norm(Z[idx_in_cluster, :4] - centroid, axis=1)
        med_global_idx = idx_in_cluster[np.argmin(dists)]
        row = meta.iloc[med_global_idx]
        traj = WE_ROOT / f"traj_segs/{int(row['iter']):06d}/{int(row['seg']):06d}/seg.nc"
        # cpptraj frame is 1-indexed
        cf.write(f"trajin {traj} {int(row['frame'])+1} {int(row['frame'])+1} 1\n")
        manifest.append({"cluster": k, "iter": int(row['iter']), "seg": int(row['seg']),
                         "frame": int(row['frame']),
                         "PC1": float(row['PC1']), "PC2": float(row['PC2']),
                         "weight": float(row['weight']), "bound": int(row['bound']),
                         "cluster_pop_pct": float(cluster_stats[k]['pop_fraction'])*100,
                         "cluster_bound_pct": float(cluster_stats[k]['bound_fraction'])*100})
    cf.write(f"autoimage :1-140,FAS firstatom\n")
    cf.write(f"strip :WAT,Na+,Cl-,K+\n")
    cf.write(f"trajout {REPS}/cluster.pdb pdb multi\n")
    cf.write(f"run\nquit\n")

r = subprocess.run([CPPTRAJ, "-i", str(extract_script)], capture_output=True, text=True)
if r.returncode != 0:
    print(f"  cpptraj FAILED:\n{r.stderr[-500:]}"); sys.exit(1)

# Rename cluster.pdb.N → cluster_NN.pdb and verify
pdbs = sorted(REPS.glob("cluster.pdb.*"), key=lambda p: int(p.suffix[1:]))
for i, p in enumerate(pdbs):
    new = REPS / f"cluster_{manifest[i]['cluster']:02d}.pdb"
    p.rename(new)
    manifest[i]["pdb"] = str(new.name)
pd.DataFrame(manifest).to_csv(REPS/"manifest.csv", index=False)
print(f"  ✓ extracted {len(manifest)} cluster medoid PDBs")
for m in manifest:
    print(f"    cluster_{m['cluster']:02d}.pdb  pop={m['cluster_pop_pct']:5.1f}%  bound={m['cluster_bound_pct']:5.1f}%  iter={m['iter']} seg={m['seg']} f={m['frame']}")


# ─────────────────── Step 7: per-cluster fasudil contact summary ───────────────
print(f"\n[step 7] per-cluster fasudil-contact frequency (weighted)...")
# Load contact_map data we already have on disk
from glob import glob
contact_map_per_cluster = defaultdict(lambda: np.zeros(140))
weight_per_cluster = defaultdict(float)
for row in meta.itertuples():
    cm_path = WE_ROOT / f"traj_segs/{row.iter:06d}/{row.seg:06d}/contact_map.dat"
    if not cm_path.exists(): continue
    try:
        cm = np.loadtxt(cm_path)
        if cm.ndim == 1: cm = cm[None, :]
    except Exception: continue
    if row.frame >= len(cm): continue
    contact_freq = (cm[row.frame] < 5.0).astype(float)   # 1 if within 5 Å
    contact_map_per_cluster[row.cluster] += contact_freq * row.weight
    weight_per_cluster[row.cluster] += row.weight

# Normalize → fraction of cluster mass that has each residue in contact
for k in contact_map_per_cluster:
    if weight_per_cluster[k] > 0:
        contact_map_per_cluster[k] /= weight_per_cluster[k]

fig, axes = plt.subplots(N_CLUSTERS, 1, figsize=(11, N_CLUSTERS*0.6), sharex=True)
for k in range(N_CLUSTERS):
    ax = axes[k]
    freq = contact_map_per_cluster.get(k, np.zeros(140))
    ax.bar(range(1, 141), freq, color="C0" if cluster_stats[k]['bound_fraction']>0.5 else "lightgray")
    # Mark key residues
    for r in [125, 133, 135, 136, 130, 137]:
        if freq[r-1] > 0.1:
            ax.axvline(r, color="red", alpha=0.3, linewidth=0.5)
    ax.set_ylim(0, max(0.2, freq.max()*1.1))
    ax.set_ylabel(f"C{k}\np{cluster_stats[k]['pop_fraction']*100:.0f}", rotation=0, fontsize=7, labelpad=20)
    ax.set_yticks([])
axes[-1].set_xlabel("α-syn residue number")
axes[0].set_title("Per-cluster fasudil-contact frequency (weighted; red lines: known binding-site residues)")
fig.tight_layout(); fig.savefig(FIG/"per_cluster_contacts.png", dpi=140); plt.close(fig)
print(f"  ✓ saved figures/per_cluster_contacts.png")

print(f"\n[done] all outputs in {ROOT}")
print(f"  data/         per-frame meta + features + PCA")
print(f"  figures/      4 figures (scree, landscape, bound overlay, clusters, contacts)")
print(f"  representatives/  {len(manifest)} medoid PDBs + manifest.csv")
