"""
Render the solvated α-syn + fasudil + waters + ions system to a PNG.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import MDAnalysis as mda

PRMTOP = "/home/anugraha/IDP/build/complex.prmtop"
COORDS = "/home/anugraha/IDP/build/complex.rst7"
OUT_PNG = "/home/anugraha/IDP/build/complex_view.png"

u = mda.Universe(PRMTOP, COORDS, format="INPCRD")
print(f"loaded {u.atoms.n_atoms:,} atoms")

protein  = u.select_atoms("protein and name CA")
fas      = u.select_atoms("resname FAS")
water_O  = u.select_atoms("resname WAT and name O*")
ions_Na  = u.select_atoms("resname Na+")
ions_Cl  = u.select_atoms("resname Cl-")

print(f"  protein CA: {len(protein):,}")
print(f"  fasudil:    {len(fas):,}")
print(f"  waters O:   {len(water_O):,}")
print(f"  Na+:        {len(ions_Na):,}")
print(f"  Cl-:        {len(ions_Cl):,}")

# Render
fig = plt.figure(figsize=(14, 7))

# --- Left: 3D scatter with waters (sparse) ---
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
# sparse waters (every 30th to keep the plot legible)
w = water_O.positions[::30]
ax1.scatter(w[:,0], w[:,1], w[:,2], c="lightblue", s=2, alpha=0.18, label=f"water O (every 30, total {len(water_O):,})")
# ions
if len(ions_Na):
    ax1.scatter(ions_Na.positions[:,0], ions_Na.positions[:,1], ions_Na.positions[:,2], c="purple",  s=18, marker="o", label=f"Na+ ({len(ions_Na)})")
if len(ions_Cl):
    ax1.scatter(ions_Cl.positions[:,0], ions_Cl.positions[:,1], ions_Cl.positions[:,2], c="green",   s=18, marker="o", label=f"Cl- ({len(ions_Cl)})")
# α-syn CA as a line + colored gradient by residue index
ca = protein.positions
resids = protein.resids
sc = ax1.scatter(ca[:,0], ca[:,1], ca[:,2], c=resids, cmap="coolwarm", s=14, label=f"α-syn CA (140)")
ax1.plot(ca[:,0], ca[:,1], ca[:,2], "k-", lw=0.6, alpha=0.5)
# fasudil heavy atoms
f = fas.select_atoms("not name H*").positions
ax1.scatter(f[:,0], f[:,1], f[:,2], c="orange", s=60, marker="D", edgecolors="black", linewidths=0.8, label=f"fasudil ({len(f)} heavy)")
ax1.set_xlabel("x (Å)"); ax1.set_ylabel("y (Å)"); ax1.set_zlabel("z (Å)")
ax1.set_title("Solvated complex: α-syn (1XQ8) + fasudil + TIP4P-D + ions")
ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)
plt.colorbar(sc, ax=ax1, label="α-syn residue index", shrink=0.5, pad=0.06)

# --- Right: zoomed protein + ligand only (no water) ---
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
sc2 = ax2.scatter(ca[:,0], ca[:,1], ca[:,2], c=resids, cmap="coolwarm", s=24)
ax2.plot(ca[:,0], ca[:,1], ca[:,2], "k-", lw=0.8, alpha=0.6)
# C-terminal binding region (residues 125-140) emphasised
cterm = u.select_atoms("protein and name CA and resid 125-140")
ax2.scatter(cterm.positions[:,0], cterm.positions[:,1], cterm.positions[:,2],
            facecolors="none", edgecolors="red", s=120, linewidths=1.6, label="C-term binding region (125-140)")
# Y133, D135, Y136 highlighted
hilights = u.select_atoms("protein and name CA and resid 133 135 136")
for atom in hilights:
    ax2.text(atom.position[0], atom.position[1], atom.position[2], f"  {atom.resname}{atom.resid}",
             fontsize=8, color="darkred", weight="bold")
# fasudil
ax2.scatter(f[:,0], f[:,1], f[:,2], c="orange", s=80, marker="D", edgecolors="black", linewidths=0.8, label="fasudil (placed)")
ax2.set_xlabel("x (Å)"); ax2.set_ylabel("y (Å)"); ax2.set_zlabel("z (Å)")
ax2.set_title("Protein + ligand (no solvent)")
ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)
plt.colorbar(sc2, ax=ax2, label="α-syn residue index", shrink=0.5, pad=0.06)

# Compute distance + system stats for caption
asyn_heavy = u.select_atoms("protein and not name H*").positions
fas_heavy  = fas.select_atoms("not name H*").positions
from scipy.spatial.distance import cdist
d_min = cdist(asyn_heavy, fas_heavy).min()
all_xyz = u.atoms.positions
extent = all_xyz.max(axis=0) - all_xyz.min(axis=0)

fig.suptitle(
    f"System: {u.atoms.n_atoms:,} atoms  ({len(water_O):,} waters, {len(ions_Na)} Na+, {len(ions_Cl)} Cl−)  "
    f"|  occupied extent: {extent[0]:.1f} × {extent[1]:.1f} × {extent[2]:.1f} Å  |  "
    f"min α-syn–fasudil heavy-atom distance: {d_min:.2f} Å",
    fontsize=10,
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
