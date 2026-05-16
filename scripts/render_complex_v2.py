"""
Cleaner solvated-box visualization:
- truncated octahedron box drawn as a wireframe
- 2D projections (xy, xz) so the position of fasudil relative to the water envelope is unambiguous
- denser water sampling
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import MDAnalysis as mda

PRMTOP = "/home/anugraha/IDP/build/complex.prmtop"
COORDS = "/home/anugraha/IDP/build/complex.rst7"
OUT_PNG = "/home/anugraha/IDP/build/complex_view2.png"

u = mda.Universe(PRMTOP, COORDS, format="INPCRD")
prot   = u.select_atoms("protein and name CA").positions
cterm  = u.select_atoms("protein and name CA and resid 125-140").positions
fas    = u.select_atoms("resname FAS and not name H*").positions
fas_com= fas.mean(axis=0)
wat_O  = u.select_atoms("resname WAT and name O*").positions
naP    = u.select_atoms("resname Na+").positions
clM    = u.select_atoms("resname Cl-").positions

# Truncated octahedron box: edge length, angles
with open(COORDS) as f:
    box_line = f.readlines()[-1].split()
edge = float(box_line[0])
print(f"box edge: {edge:.2f} Å (truncated octahedron, angles 109.47°)")

# Truncated octahedron geometry — for Amber's convention, the box spans -edge/2 to +edge/2
# in a wrapped sense, but the *unwrapped* coordinates from rst7 are typically all positive.
# For drawing, we'll show:
#   1) the bounding cube edge = the same `edge` (worst-case enclosing cube of trunc-oct)
#   2) a circle/ellipse representing the trunc-oct inscribed sphere (radius = edge*sqrt(3)/4)
# centered at the mean of all coords.
all_xyz = u.atoms.positions
center = all_xyz.mean(axis=0)
R_inscribed = edge * np.sqrt(3) / 4   # standard formula for trunc-oct inscribed sphere
print(f"inscribed-sphere radius: {R_inscribed:.2f} Å, center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

def draw_panel(ax, i, j, names):
    # waters as background (denser — every 5th)
    w = wat_O[::5]
    ax.scatter(w[:,i], w[:,j], c="lightblue", s=1.5, alpha=0.15, label=f"water O (~{len(w):,} of {len(wat_O):,})")
    # ions
    if len(naP): ax.scatter(naP[:,i], naP[:,j], c="purple", s=20, label=f"Na+ ({len(naP)})")
    if len(clM): ax.scatter(clM[:,i], clM[:,j], c="green",  s=20, label=f"Cl- ({len(clM)})")
    # protein CA gradient
    resids = u.select_atoms("protein and name CA").resids
    sc = ax.scatter(prot[:,i], prot[:,j], c=resids, cmap="coolwarm", s=18, zorder=3)
    ax.plot(prot[:,i], prot[:,j], "k-", lw=0.7, alpha=0.5, zorder=2)
    # C-term region emphasised
    ax.scatter(cterm[:,i], cterm[:,j], facecolors="none", edgecolors="red", s=110, linewidths=1.5, zorder=4, label="C-term 125-140")
    # fasudil
    ax.scatter(fas[:,i], fas[:,j], c="orange", s=70, marker="D", edgecolors="black", linewidths=0.9, zorder=5, label="fasudil")
    ax.scatter([fas_com[i]], [fas_com[j]], c="red", s=20, marker="x", zorder=6, label="fasudil COM")
    # inscribed sphere boundary (approximation of trunc-oct outer shell in this projection)
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(center[i] + R_inscribed*np.cos(theta), center[j] + R_inscribed*np.sin(theta), "k--", lw=1.2, alpha=0.7, label="inscribed sphere")
    # enclosing cube edges (truncOct lives inside this cube)
    half = edge / 2
    ax.add_patch(plt.Rectangle((center[i]-half, center[j]-half), edge, edge,
                               fill=False, ec="gray", ls=":", lw=1.0, label="enclosing cube"))
    ax.set_xlabel(f"{names[0]} (Å)"); ax.set_ylabel(f"{names[1]} (Å)")
    ax.set_aspect("equal")
    return sc

sc = draw_panel(axes[0], 0, 1, ["x", "y"])
axes[0].set_title("xy projection")
draw_panel(axes[1], 0, 2, ["x", "z"])
axes[1].set_title("xz projection")
draw_panel(axes[2], 1, 2, ["y", "z"])
axes[2].set_title("yz projection")
axes[0].legend(loc="upper left", fontsize=7, framealpha=0.9)
plt.colorbar(sc, ax=axes, label="α-syn residue index", shrink=0.6, location="right")

fig.suptitle(
    f"Solvated system: 242,902 atoms — truncated octahedron edge {edge:.1f} Å (inscribed-sphere radius {R_inscribed:.1f} Å)\n"
    f"Dashed circle ≈ trunc-oct inscribed sphere;  dotted square = enclosing cube;  fasudil COM {tuple(np.round(fas_com,1))}",
    fontsize=11,
)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
