#!/usr/bin/env python3
"""
Extract a WE-derived receptor ensemble for ensemble docking.

Strategy:
1. Identify all 'bound' frames from WE traj_segs (using pcoord/aux .dat files
   we already have — west.h5 is locked by the running master).
2. Build per-frame contact_map vectors (140-D, ligand-residue min distances).
3. K-medoids cluster into N_CLUSTERS.
4. For each medoid frame: extract that frame from seg.nc with cpptraj,
   strip fasudil + waters, save as receptor PDB.
"""
import glob, os, sys, subprocess
import numpy as np
from pathlib import Path

N_CLUSTERS = 10
WE_ROOT = Path("/home/anugraha/IDP/westpa")
TOPOLOGY = Path("/home/anugraha/IDP/build/complex.prmtop")
OUT_DIR = Path("/home/anugraha/IDP/docking/receptor_ensemble")
OUT_DIR.mkdir(exist_ok=True, parents=True)
CPPTRAJ = "/software/amber/24/ucdhpc-20.04/ambertools25/bin/cpptraj"

os.chdir(WE_ROOT)

# 1) Scan for bound frames + load contact maps for each
print("[1/4] scanning WE segments for bound frames + loading contact maps...")
records = []           # (iter, seg, frame, contact_map_vector)
n_seg_processed = 0
n_bound = 0
for it_dir in sorted(glob.glob('traj_segs/0*')):
    it = int(os.path.basename(it_dir))
    if it > 73 or it == 0:
        continue
    for seg_dir in sorted(glob.glob(f'{it_dir}/*')):
        sg = int(os.path.basename(seg_dir))
        try:
            pc = np.loadtxt(f'{seg_dir}/pcoord.dat')
            rd = np.loadtxt(f'{seg_dir}/ring_dists.dat')
            ch = np.loadtxt(f'{seg_dir}/d135_charge_dist.dat')
            cm = np.loadtxt(f'{seg_dir}/contact_map.dat')
            if pc.ndim == 1: pc = pc[None, :]
            if rd.ndim == 1: rd = rd[None, :]
            ch = np.atleast_1d(ch)
            if cm.ndim == 1: cm = cm[None, :]
        except Exception:
            continue
        n_seg_processed += 1
        n_f = min(len(pc), len(rd), len(ch), len(cm))
        for f in range(n_f):
            d_min = pc[f, 0]
            mr = rd[f].min()
            dch = ch[f]
            if d_min < 4.5 and (mr < 6.0 or dch < 5.0):
                records.append((it, sg, f, cm[f]))
                n_bound += 1

print(f"  scanned {n_seg_processed} segments")
print(f"  bound frames: {n_bound}")

if len(records) < N_CLUSTERS:
    print(f"[ERROR] only {len(records)} bound frames, need ≥{N_CLUSTERS}")
    sys.exit(1)

# 2) Cluster using k-medoids on contact_map vectors
print(f"\n[2/4] k-medoids clustering into {N_CLUSTERS} representatives...")
contact_maps = np.stack([r[3] for r in records])
# Replace NaN with a large value (no contact)
contact_maps = np.nan_to_num(contact_maps, nan=20.0)

# Simple k-medoids (PAM-style) using sklearn-extra if available, else a quick
# Lloyd-style on cluster medoids
try:
    from sklearn_extra.cluster import KMedoids
    km = KMedoids(n_clusters=N_CLUSTERS, random_state=42, method='alternate', max_iter=200)
    labels = km.fit_predict(contact_maps)
    medoid_indices = km.medoid_indices_
except ImportError:
    # Fallback: k-means then pick the nearest real frame to each centroid
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(contact_maps)
    medoid_indices = []
    for c in range(N_CLUSTERS):
        in_c = np.where(labels == c)[0]
        if len(in_c) == 0:
            continue
        dists = np.linalg.norm(contact_maps[in_c] - km.cluster_centers_[c], axis=1)
        medoid_indices.append(in_c[np.argmin(dists)])

print(f"  cluster sizes: {sorted([(c, int((labels==c).sum())) for c in range(N_CLUSTERS)], key=lambda x:-x[1])}")
print(f"  medoid frames:")
for i, mi in enumerate(medoid_indices):
    it, sg, f, _ = records[mi]
    n_in = int((labels == i).sum())
    print(f"    cluster {i}  (n={n_in:4d})  iter={it} seg={sg} frame={f}")

# 3) Extract each medoid frame with cpptraj
print(f"\n[3/4] extracting medoid frames as receptor PDBs (fasudil stripped)...")
cpp_script = OUT_DIR / "extract.cpptraj"
with open(cpp_script, "w") as cf:
    cf.write(f"parm {TOPOLOGY}\n")
    for i, mi in enumerate(medoid_indices):
        it, sg, f, _ = records[mi]
        traj = WE_ROOT / f"traj_segs/{it:06d}/{sg:06d}/seg.nc"
        # cpptraj frame is 1-indexed, our f is 0-indexed
        cf.write(f"trajin {traj} {f+1} {f+1} 1\n")
    cf.write(f"autoimage :1-140,FAS firstatom\n")
    # Strip fasudil + waters/ions, keep only α-syn for docking receptor
    cf.write(f"strip :WAT,Na+,Cl-,K+,FAS\n")
    cf.write(f"trajout {OUT_DIR}/receptor_ensemble.pdb pdb multi\n")
    cf.write(f"run\nquit\n")

r = subprocess.run([CPPTRAJ, "-i", str(cpp_script)], capture_output=True, text=True)
if r.returncode != 0:
    print(f"[ERROR] cpptraj failed:\n{r.stderr[-500:]}")
    sys.exit(1)

# cpptraj with "multi" produces files receptor_ensemble.pdb.1, .2, ... — rename to receptor_NN.pdb
pdb_files = sorted(OUT_DIR.glob("receptor_ensemble.pdb.*"))
print(f"  cpptraj produced {len(pdb_files)} PDB files")
for i, p in enumerate(pdb_files):
    new_path = OUT_DIR / f"receptor_{i+1:02d}.pdb"
    p.rename(new_path)
    n_atoms = sum(1 for line in open(new_path) if line.startswith("ATOM"))
    it, sg, f, _ = records[medoid_indices[i]]
    print(f"  receptor_{i+1:02d}.pdb  ({n_atoms} atoms)   from iter={it} seg={sg} frame={f}")

# 4) Save a manifest with weights = cluster sizes (proxy for WE-derived weight)
print(f"\n[4/4] writing manifest with cluster weights...")
with open(OUT_DIR / "ensemble_manifest.csv", "w") as f:
    f.write("receptor,iter,seg,frame,cluster_size,weight\n")
    total = sum((labels == c).sum() for c in range(N_CLUSTERS))
    for i, mi in enumerate(medoid_indices):
        it, sg, fr, _ = records[mi]
        size = int((labels == i).sum())
        w = size / total
        f.write(f"receptor_{i+1:02d}.pdb,{it},{sg},{fr},{size},{w:.4f}\n")
print(f"  manifest: {OUT_DIR}/ensemble_manifest.csv")
print(f"\n[done] receptor ensemble in {OUT_DIR}/")
