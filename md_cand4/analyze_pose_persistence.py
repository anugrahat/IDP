#!/usr/bin/env python3
"""
Post-MD analysis of cand 4 pose persistence on c09 α-syn.

Reports:
  1. Ligand RMSD vs starting pose over time
  2. Ligand center-of-mass drift from Y133 / E130 / D135
  3. Fraction of frames making each key contact (Y133, E130, D135)
  4. Verdict: "pose holds" / "drifts" / "lost"
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSD
from MDAnalysis.analysis.distances import distance_array

ROOT = Path("/home/anugraha/IDP/md_cand4")
PRMTOP = ROOT / "complex_cand4.prmtop"
TRAJ = ROOT / "prod.nc"
REF_RST = ROOT / "equil.rst7"   # the starting structure for production

if not TRAJ.exists():
    print(f"[!] {TRAJ} not yet present — MD still running")
    sys.exit(1)

u = mda.Universe(str(PRMTOP), str(TRAJ))
# Use first frame of production as the reference (starting pose after equilibration)
ref = mda.Universe(str(PRMTOP), str(TRAJ))
ref.trajectory[0]

# Ligand = UNL
lig_sel = "resname UNL"
prot_sel = "protein"
y133 = "resid 133 and not name H*"
e130 = "resid 130 and not name H*"
d135 = "resid 135 and not name H*"

n_frames = len(u.trajectory)
dt_ns = 0.01   # ntwx=5000 × dt=0.002 ps = 10 ps per frame
print(f"loaded {n_frames} frames ({n_frames*dt_ns:.1f} ns total)")

# 1. Ligand RMSD vs starting pose (align on protein backbone)
rmsd_analysis = RMSD(u, ref, select="protein and backbone", groupselections=[lig_sel])
rmsd_analysis.run()
times_ns = rmsd_analysis.results.rmsd[:, 1] / 1000.0  # ps → ns
backbone_rmsd = rmsd_analysis.results.rmsd[:, 2]
ligand_rmsd = rmsd_analysis.results.rmsd[:, 3]
print(f"\nbackbone RMSD: mean {backbone_rmsd.mean():.2f} Å, max {backbone_rmsd.max():.2f}")
print(f"ligand RMSD:   mean {ligand_rmsd.mean():.2f} Å, max {ligand_rmsd.max():.2f}")

# 2. Min distance from ligand to key residues over time
y133_atoms_ref = ref.select_atoms(y133)
e130_atoms_ref = ref.select_atoms(e130)
d135_atoms_ref = ref.select_atoms(d135)
lig_atoms = u.select_atoms(lig_sel)
y133_atoms = u.select_atoms(y133)
e130_atoms = u.select_atoms(e130)
d135_atoms = u.select_atoms(d135)

print("computing per-frame min distances...")
d_y133, d_e130, d_d135 = [], [], []
for ts in u.trajectory:
    d_y133.append(distance_array(lig_atoms.positions, y133_atoms.positions).min())
    d_e130.append(distance_array(lig_atoms.positions, e130_atoms.positions).min())
    d_d135.append(distance_array(lig_atoms.positions, d135_atoms.positions).min())
d_y133 = np.array(d_y133)
d_e130 = np.array(d_e130)
d_d135 = np.array(d_d135)

CONTACT_CUT = 5.0
print(f"\nfraction of frames within {CONTACT_CUT} Å of:")
print(f"  TYR133: {(d_y133 < CONTACT_CUT).mean()*100:.1f}%")
print(f"  GLU130: {(d_e130 < CONTACT_CUT).mean()*100:.1f}%")
print(f"  ASP135: {(d_d135 < CONTACT_CUT).mean()*100:.1f}%")

# Verdict
final_rmsd = ligand_rmsd[-min(10, len(ligand_rmsd)):].mean()   # last 100 ps avg
if final_rmsd < 3.0 and (d_y133 < CONTACT_CUT).mean() > 0.5:
    verdict = "POSE HOLDS — RMSD < 3 Å, TYR133 contact > 50%"
elif final_rmsd < 6.0:
    verdict = "PARTIAL — RMSD < 6 Å but contacts may have rearranged"
else:
    verdict = "POSE LOST — RMSD > 6 Å, fasudil-mode binding NOT reproduced"
print(f"\n══════ VERDICT: {verdict} ══════")

# 3. Plot
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].plot(times_ns, ligand_rmsd, label="ligand RMSD", color="C0")
axes[0].plot(times_ns, backbone_rmsd, label="protein backbone RMSD", color="gray", alpha=0.5)
axes[0].axhline(3.0, color="green", linestyle="--", alpha=0.4, label="3 Å threshold")
axes[0].set_ylabel("RMSD (Å)")
axes[0].legend()
axes[0].set_title(f"Cand 4 pose persistence on c09 α-syn — VERDICT: {verdict}")

axes[1].plot(times_ns, d_y133, label="cand 4 → TYR133", color="red")
axes[1].plot(times_ns, d_e130, label="cand 4 → GLU130", color="blue")
axes[1].plot(times_ns, d_d135, label="cand 4 → ASP135", color="green")
axes[1].axhline(CONTACT_CUT, color="gray", linestyle="--", alpha=0.5, label=f"{CONTACT_CUT} Å (contact cutoff)")
axes[1].set_xlabel("Time (ns)")
axes[1].set_ylabel("Min heavy-atom distance (Å)")
axes[1].legend()

fig.tight_layout()
fig.savefig(ROOT/"cand4_pose_persistence.png", dpi=140)
plt.close(fig)
print(f"\nfigure: {ROOT/'cand4_pose_persistence.png'}")

# Save numerical data
import pandas as pd
df = pd.DataFrame({
    "time_ns": times_ns, "ligand_rmsd": ligand_rmsd, "backbone_rmsd": backbone_rmsd,
    "d_y133": d_y133, "d_e130": d_e130, "d_d135": d_d135
})
df.to_csv(ROOT/"cand4_pose_persistence.csv", index=False)
print(f"data: {ROOT/'cand4_pose_persistence.csv'}")
