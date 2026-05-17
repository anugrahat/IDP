#!/usr/bin/env python3
"""
Stage 2 sanity: dock cand_0's 7 variants into receptor_03, verify each
docking is geometrically + chemically reasonable BEFORE scaling.

Pass criteria per variant:
  ✓ Vina score in [-8, -1] kcal/mol  (sensible range for IDP target)
  ✓ Meeko PDBQT→RDKit on docked pose returns the same molecule (no fragmentation)
  ✓ PLIF on pose has ≥1 interaction (not silently empty)
  ✓ Tanimoto to reference is a real number in [0, 1]

If all 7 variants pass, the pipeline is safe to scale to all 20 × 11 receptors.
"""
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
import MDAnalysis as mda
import prolif as plf
from meeko import PDBQTMolecule, RDKitMolCreate
from vina import Vina

ROOT = Path("/home/anugraha/IDP/docking")
MANIFEST = json.loads((ROOT/"candidates_v4/manifest.json").read_text())
REF_PLIF = json.loads((ROOT/"plif/reference_plif.json").read_text())

RECEPTOR_PDB = ROOT/"receptor_ensemble/receptor_03.pdb"
RECEPTOR_PDBQT = ROOT/"receptor_ensemble/receptor_03.pdbqt"

# Box: C-term centroid, 22 Å cube
def get_box_center(pdb, resids={133,135,136}):
    pts = []
    for line in open(pdb):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            if int(line[22:26]) in resids:
                pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(pts).mean(axis=0).tolist()

BOX_CENTER = get_box_center(RECEPTOR_PDB)
BOX_SIZE = [22.0, 22.0, 22.0]
print(f"receptor: receptor_03.pdb  (iter 28 / seg 39 family, cluster size 545)")
print(f"box center: {[round(x,1) for x in BOX_CENTER]}")
print(f"box size:   {BOX_SIZE}")

# Verify box geometry covers the binding site (per the new memory rule)
import numpy as np
center = np.array(BOX_CENTER)
half = np.array(BOX_SIZE) / 2
y133_ca = None
for line in open(RECEPTOR_PDB):
    if line.startswith("ATOM") and line[12:16].strip() == "CA" and int(line[22:26]) == 133:
        y133_ca = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        break
y133_in_box = all(abs(y133_ca[k] - center[k]) <= half[k] for k in range(3))
print(f"\nGEOMETRY CHECK: Y133 CA at {y133_ca.round(1)}, box center at {center.round(1)}, half={half}")
print(f"  Y133 inside box? {y133_in_box}  (must be True)")
assert y133_in_box, "BAIL OUT — Y133 not inside docking box"

# Locate cand_0's variants
cand_0 = next(c for c in MANIFEST if c["cand_idx"] == 0)
print(f"\ncand_0 parent: {cand_0['parent_smiles']}")
print(f"  {len(cand_0['variants'])} variants to dock\n")

ALL_INTER = ["Hydrophobic","HBDonor","HBAcceptor","PiStacking",
             "PiCation","CationPi","Anionic","Cationic","VdWContact"]

def compute_plif(pose_pdbqt, receptor_pdb):
    """Use Meeko PDBQT→RDKit (the correct path) then ProLIF."""
    pmol = PDBQTMolecule.from_file(str(pose_pdbqt))
    mols = RDKitMolCreate.from_pdbqt_mol(pmol)
    if not mols: return None, None
    m = mols[0]
    # Verify the molecule is intact (no fragmentation)
    roundtrip_smi = Chem.MolToSmiles(Chem.RemoveHs(m))
    lig = plf.Molecule.from_rdkit(m)
    u = mda.Universe(str(receptor_pdb))
    prot = u.select_atoms("protein")
    prot_mol = plf.Molecule.from_mda(prot)
    fp = plf.Fingerprint(ALL_INTER)
    fp.run_from_iterable([lig], prot_mol, progress=False)
    df = fp.to_dataframe()
    ints = set()
    if not df.empty:
        for col in df.columns:
            if df[col].iloc[0]:
                _, prot_res, itype = col
                ints.add(f"{str(prot_res).split('.')[0]}|{itype}")
    return roundtrip_smi, ints

def plif_tanimoto(pose_keys):
    if not pose_keys: return 0.0
    ref_keys = set(REF_PLIF.keys())
    shared = pose_keys & ref_keys
    only_ref = ref_keys - pose_keys
    only_pose = pose_keys - ref_keys
    num = sum(REF_PLIF[k] for k in shared)
    den = sum(REF_PLIF[k] for k in shared | only_ref) + len(only_pose)
    return num / den if den > 0 else 0.0

# Dock each variant
results = []
for v in cand_0["variants"]:
    var_pdbqt = Path(v["pdbqt"])
    out_pose = ROOT/f"stage2_poses/cand0_var{v['var_idx']:02d}.pdbqt"
    out_pose.parent.mkdir(exist_ok=True)

    try:
        vina = Vina(sf_name="vina", cpu=4, verbosity=0, seed=42)
        vina.set_receptor(str(RECEPTOR_PDBQT))
        vina.set_ligand_from_file(str(var_pdbqt))
        vina.compute_vina_maps(center=BOX_CENTER, box_size=BOX_SIZE)
        vina.dock(exhaustiveness=16, n_poses=5)
        vina.write_poses(str(out_pose), n_poses=5, overwrite=True)
    except Exception as e:
        results.append({"var": v, "error": f"VINA: {e}"})
        continue

    # Parse top score
    energies = []
    for line in open(out_pose):
        if line.startswith("REMARK VINA RESULT"):
            energies.append(float(line.split()[3]))
    top_score = energies[0] if energies else None

    # PLIF on top pose with Meeko round-trip check
    roundtrip_smi, ints = compute_plif(out_pose, RECEPTOR_PDB)
    expected_smi = Chem.MolToSmiles(Chem.MolFromSmiles(v["smiles"]))
    intact = (roundtrip_smi == expected_smi)
    tani = plif_tanimoto(ints) if ints else 0.0

    # Per-variant pass/fail
    score_ok = top_score is not None and -8 <= top_score <= -1
    intact_ok = intact
    plif_ok = ints is not None and len(ints) >= 1
    all_ok = score_ok and intact_ok and plif_ok

    results.append({
        "var": v, "top_score": top_score, "intact": intact,
        "n_contacts": len(ints) if ints else 0, "tanimoto": tani,
        "ints": ints, "all_ok": all_ok,
    })

# Print results
print(f"{'var':>4} {'chg':>4} {'site':<35} {'E_top':>7} {'intact':>6} {'#int':>5} {'PLIF_T':>7} {'contacts (target res)':<40} pass")
print("-"*150)
TARGET = {"TYR133", "GLU130", "GLU137", "ASP135"}
for r in results:
    if "error" in r:
        print(f"  {r['var']['var_idx']:>2}  ERROR: {r['error']}"); continue
    v = r["var"]
    target_contacts = sorted(c for c in (r["ints"] or set()) if any(c.startswith(t) for t in TARGET))
    tcs = ",".join(target_contacts) if target_contacts else "-"
    flag = "✓" if r["all_ok"] else "✗"
    site_or_reason = v.get('site') or v.get('reason', '?')
    print(f"  {v['var_idx']:>2} {v['charge']:>+3} {site_or_reason:<35} {r['top_score']:>7.2f}   {str(r['intact']):>5}   {r['n_contacts']:>3}   {r['tanimoto']:>6.3f}   {tcs:<40} {flag}")

# Summary
n_pass = sum(1 for r in results if r.get("all_ok"))
print(f"\n{'='*70}")
print(f"PASS: {n_pass}/{len(results)} variants meet all criteria")
print(f"  best Vina score: {min(r['top_score'] for r in results if r.get('top_score') is not None):.2f} kcal/mol")
best = max(results, key=lambda r: r.get("tanimoto", 0))
print(f"  best PLIF Tanimoto: {best['tanimoto']:.3f}  (var{best['var']['var_idx']}, charge {best['var']['charge']:+d})")
