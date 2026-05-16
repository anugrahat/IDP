"""
Place protonated fasudil so its min heavy-atom distance to α-syn is ~12 Å,
along the line from α-syn COM through the C-terminal binding region (residues 125-140).

This biases the starting state toward the biologically-relevant binding face
of α-syn (Y133/D135/Y136 region per Robustelli 2022), keeps fasudil well
inside the truncated-octahedron inscribed sphere, and makes WE binding events
kinetically accessible (vs. starting 80 Å away in solvent).

Reads:
    /home/anugraha/IDP/inputs/protein/1XQ8_amber.pdb
    /home/anugraha/IDP/inputs/ligand/ac_work/fasudil_gaff2.mol2
Writes:
    /home/anugraha/IDP/build/fasudil_placed.mol2
    /home/anugraha/IDP/build/asyn_fasudil_placed.pdb  (visualization only)
"""
import numpy as np
import MDAnalysis as mda
from scipy.spatial.distance import cdist

ASYN_PDB     = "/home/anugraha/IDP/inputs/protein/1XQ8_amber.pdb"
FAS_MOL2_IN  = "/home/anugraha/IDP/inputs/ligand/ac_work/fasudil_gaff2.mol2"
FAS_MOL2_OUT = "/home/anugraha/IDP/build/fasudil_placed.mol2"
ASYN_FAS_PDB = "/home/anugraha/IDP/build/asyn_fasudil_placed.pdb"
TARGET_MIN_DIST = 12.0    # min heavy-atom α-syn ↔ fasudil distance after placement (Å)
# Binding triad per Robustelli 2022: Y133 + Y136 (π-stacking) + D135 (salt bridge)
TRIAD_RESIDS = [133, 135, 136]

# --- α-syn + C-terminal centroid ---
asyn = mda.Universe(ASYN_PDB)
asyn_xyz = asyn.atoms.positions
asyn_com = asyn_xyz.mean(axis=0)

triad_sel = "protein and not name H* and (" + " or ".join(f"resid {r}" for r in TRIAD_RESIDS) + ")"
triad_atoms = asyn.select_atoms(triad_sel)
if len(triad_atoms) == 0:
    raise SystemExit("no triad atoms selected — check resid range")
triad_xyz = triad_atoms.positions
triad_com = triad_xyz.mean(axis=0)
print(f"α-syn COM: ({asyn_com[0]:.2f}, {asyn_com[1]:.2f}, {asyn_com[2]:.2f}) Å")
print(f"Y133/D135/Y136 triad centroid (heavy atoms): ({triad_com[0]:.2f}, {triad_com[1]:.2f}, {triad_com[2]:.2f}) Å")

# --- Direction: α-syn COM → triad centroid (outward through binding face) ---
direction = triad_com - asyn_com
direction /= np.linalg.norm(direction)
print(f"placement direction (α-syn COM → Y133/D135/Y136): {direction}")

# --- Parse fasudil mol2, get current COM ---
with open(FAS_MOL2_IN) as f:
    lines = f.readlines()
atom_start = atom_end = None
for i, ln in enumerate(lines):
    if ln.startswith("@<TRIPOS>ATOM"):
        atom_start = i + 1
    elif ln.startswith("@<TRIPOS>BOND") and atom_start is not None:
        atom_end = i
        break
atom_lines = lines[atom_start:atom_end]
fas_xyz = np.array([[float(p) for p in ln.split()[2:5]] for ln in atom_lines])
fas_com = fas_xyz.mean(axis=0)

# --- Search over directions + offsets to satisfy BOTH constraints ---
# Constraint 1: min(triad, fasudil) ≈ TARGET_MIN_DIST  (well-positioned near binding face)
# Constraint 2: min(any α-syn, fasudil) ≥ MIN_ALL      (no contacts with disordered tail residues)
# Sample directions on a unit sphere; for each direction find d that satisfies (1), then check (2).
MIN_ALL_DIST = 10.0
MAX_TRIAD_DIST = 18.0     # allow triad distance up to this if needed to clear tail residues
N_DIRS = 800

rng = np.random.default_rng(seed=20260515)
# anchor direction = α-syn COM → triad centroid (binding face normal)
anchor = direction.copy()

candidates = []
# include the anchor direction + a Fibonacci sphere of perturbations
phi = (1 + 5 ** 0.5) / 2
for k in range(N_DIRS):
    z = 1 - 2 * (k + 0.5) / N_DIRS
    r = np.sqrt(max(0.0, 1 - z * z))
    theta = 2 * np.pi * k / phi
    sample = np.array([r * np.cos(theta), r * np.sin(theta), z])
    # bias toward the anchor by mixing 60/40
    d_try = 0.7 * anchor + 0.3 * sample
    d_try /= np.linalg.norm(d_try)
    candidates.append(d_try)
candidates.insert(0, anchor)   # try anchor first

best = None
for cand_dir in candidates:
    # search offset along this direction; accept any (d, m_tri, m_all) where
    # m_tri ∈ [TARGET, MAX_TRIAD] AND m_all >= MIN_ALL.  Prefer smaller m_tri.
    for d in np.linspace(0.0, 60.0, 601):
        com_try = triad_com + cand_dir * d
        fas_t = fas_xyz - fas_com + com_try
        m_tri = float(cdist(triad_xyz, fas_t).min())
        m_all = float(cdist(asyn_xyz,  fas_t).min())
        if (m_tri >= TARGET_MIN_DIST - 0.5) and (m_tri <= MAX_TRIAD_DIST) and (m_all >= MIN_ALL_DIST):
            # Prefer the smallest (closest to triad) feasible m_tri across all directions
            score = -m_tri + 0.1 * (m_all - MIN_ALL_DIST)
            if best is None or score > best["score"]:
                best = dict(dir=cand_dir.copy(), d=d, m_tri=m_tri, m_all=m_all, score=score, com=com_try.copy())
            break

if best is None:
    raise SystemExit("could not find a placement satisfying triad ≈ 12 AND all ≥ 10 — relax tolerances or pick a different scheme")

print(f"chosen direction: ({best['dir'][0]:.3f}, {best['dir'][1]:.3f}, {best['dir'][2]:.3f})")
print(f"  offset along direction: {best['d']:.2f} Å")
print(f"  min distance to TRIAD (Y133/D135/Y136): {best['m_tri']:.2f} Å  ← target {TARGET_MIN_DIST}")
print(f"  min distance to ANY α-syn residue:       {best['m_all']:.2f} Å  ← must be ≥ {MIN_ALL_DIST}")
target_fas_com = best['com']
best_d = best['d']
best_min_triad = best['m_tri']
best_min_all   = best['m_all']
print(f"fasudil COM after placement: ({target_fas_com[0]:.2f}, {target_fas_com[1]:.2f}, {target_fas_com[2]:.2f}) Å")
translation = target_fas_com - fas_com
xyz_new = fas_xyz + translation

# --- Write new mol2 with updated coords (preserve atom names, types, charges) ---
new_lines = list(lines)
for j, ln in enumerate(atom_lines):
    fields = ln.split()
    new_lines[atom_start + j] = (
        f"  {int(fields[0]):>4d} {fields[1]:<8s}"
        f"{xyz_new[j,0]:>10.4f}{xyz_new[j,1]:>10.4f}{xyz_new[j,2]:>10.4f} "
        f"{fields[5]:<5s}    {fields[6]} {fields[7]:<7s}  {float(fields[8]):>9.6f}\n"
    )
with open(FAS_MOL2_OUT, "w") as f:
    f.writelines(new_lines)
print(f"wrote {FAS_MOL2_OUT}")

# --- Combined PDB for visualization only ---
fas_u = mda.Universe(FAS_MOL2_IN)
fas_u.atoms.positions = xyz_new
merged = mda.Merge(asyn.atoms, fas_u.atoms)
merged.atoms.write(ASYN_FAS_PDB)
print(f"wrote {ASYN_FAS_PDB}")
