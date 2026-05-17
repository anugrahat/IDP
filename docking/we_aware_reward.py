#!/usr/bin/env python3
"""
WE-aware reward function for REINVENT 4 staged learning (Task #24).

Components (each in [0,1], weighted sum → final reward):
  w_qed              = 0.10   QED drug-likeness
  w_ecfp_window      = 0.10   ECFP4 Tanimoto-to-fasudil, windowed (peak at sim≈0.5)
  w_pharm            = 0.10   pharmacophore match (aromatic ring + basic amine)
  w_dock_c09         = 0.20   Vina dock into PCA cluster_09 receptor, normalized
  w_plif             = 0.30   PLIF Tanimoto to fasudil reference on c09 pose
  w_consistency      = 0.10   TYR133 contact present (binary 0/1)
  w_richness         = 0.10   # of fasudil-reference contacts reproduced (cap at 5)

Usage:
    python3 we_aware_reward.py --smiles "O=S(=O)(N1CCNCCC1)c1ccnc2ccccc12"
    python3 we_aware_reward.py --input smiles.txt --output rewards.csv

As library:
    from we_aware_reward import score_smiles
    reward = score_smiles("...")
"""
import os, sys, json, argparse, tempfile, signal
from pathlib import Path
import numpy as np

# Thread caps before heavy imports
for k in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(k, "1")

from rdkit import Chem
from rdkit.Chem import AllChem, QED, DataStructs
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

ROOT = Path("/home/anugraha/IDP/docking")
REF_PLIF_PATH = ROOT / "plif/reference_plif.json"
C09_RECEPTOR_PDB = ROOT / "c09_receptor.pdb"
C09_RECEPTOR_PDBQT = ROOT / "c09_receptor.pdbqt"
FASUDIL_SMILES = "O=S(=O)(N1CCNCCC1)c1ccnc2ccccc12"
BOX_SIZE = [22.0, 22.0, 22.0]

WEIGHTS = {
    "qed":         0.10,
    "ecfp_window": 0.10,
    "pharm":       0.10,
    "dock_c09":    0.20,
    "plif":        0.30,
    "consistency": 0.10,
    "richness":    0.10,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6


def _load_refs():
    if hasattr(_load_refs, "_cache"):
        return _load_refs._cache
    ref_plif = json.loads(REF_PLIF_PATH.read_text())
    fas_fp = AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(FASUDIL_SMILES), 2, 2048)
    pts = []
    for line in open(C09_RECEPTOR_PDB):
        if line.startswith("ATOM") and line[12:16].strip()=="CA" and int(line[22:26]) in {133,135,136}:
            pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    center = np.array(pts).mean(axis=0).tolist()
    _load_refs._cache = {"ref_plif": ref_plif, "fas_fp": fas_fp, "center": center}
    return _load_refs._cache


def score_qed(mol):
    try: return float(QED.qed(mol))
    except Exception: return 0.0


def score_ecfp_window(mol, center=0.5, width=0.3):
    refs = _load_refs()
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    t = DataStructs.TanimotoSimilarity(fp, refs["fas_fp"])
    return float(max(0.0, 1.0 - abs(t - center) / width))


def score_pharmacophore(mol):
    aromatic = any(a.GetIsAromatic() and a.GetSymbol() in ("C","N") for a in mol.GetAtoms())
    basic_n = any(
        a.GetSymbol()=="N" and not a.GetIsAromatic()
        and a.GetTotalDegree() < 4
        and not any(n.GetSymbol()=="S" for n in a.GetNeighbors())
        for a in mol.GetAtoms()
    )
    return float(0.5 * int(aromatic) + 0.5 * int(basic_n))


def score_dock_c09(mol):
    """Dock cand into c09. Returns (reward, pose_path) — pose used downstream for PLIF."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from vina import Vina
    m = Chem.Mol(mol)
    for atom in m.GetAtoms():
        if (atom.GetSymbol()=="N" and not atom.GetIsAromatic()
            and atom.GetFormalCharge()==0 and atom.GetTotalDegree()<4
            and atom.GetTotalNumHs()>=1
            and not any(n.GetSymbol()=="S" for n in atom.GetNeighbors())):
            atom.SetFormalCharge(+1); break
    m.UpdatePropertyCache(strict=False)
    m = Chem.AddHs(m)
    try:
        if AllChem.EmbedMolecule(m, randomSeed=42) != 0: return 0.0, None
        AllChem.MMFFOptimizeMolecule(m)
    except Exception:
        return 0.0, None
    prep = MoleculePreparation()
    pm = prep.prepare(m)
    if not pm: return 0.0, None
    pdbqt_str, ok, _ = PDBQTWriterLegacy.write_string(pm[0])
    if not ok: return 0.0, None
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as tf:
        tf.write(pdbqt_str.encode()); lig_pdbqt = tf.name
    out_pose = lig_pdbqt.replace(".pdbqt", "_pose.pdbqt")
    try:
        v = Vina(sf_name="vina", cpu=2, verbosity=0, seed=42)
        v.set_receptor(str(C09_RECEPTOR_PDBQT))
        v.set_ligand_from_file(lig_pdbqt)
        v.compute_vina_maps(center=_load_refs()["center"], box_size=BOX_SIZE)
        v.dock(exhaustiveness=8, n_poses=3)
        v.write_poses(out_pose, n_poses=3, overwrite=True)
        e = None
        for line in open(out_pose):
            if line.startswith("REMARK VINA RESULT"):
                e = float(line.split()[3]); break
        if e is None: return 0.0, None
        # Map: -8 → 1.0, -2 → 0.0 (linear)
        return max(0.0, min(1.0, (-e - 2.0) / 6.0)), out_pose
    finally:
        if os.path.exists(lig_pdbqt): os.unlink(lig_pdbqt)


def _plif_contacts(pose_pdbqt, receptor_pdb):
    from meeko import PDBQTMolecule, RDKitMolCreate
    import MDAnalysis as mda
    import prolif as plf
    try:
        pmol = PDBQTMolecule.from_file(pose_pdbqt)
        mols = RDKitMolCreate.from_pdbqt_mol(pmol)
        if not mols: return set()
        lig = plf.Molecule.from_rdkit(mols[0])
        u = mda.Universe(str(receptor_pdb))
        prot_mol = plf.Molecule.from_mda(u.select_atoms("protein"))
        fp = plf.Fingerprint(["Hydrophobic","HBDonor","HBAcceptor","PiStacking",
                              "PiCation","CationPi","Anionic","Cationic","VdWContact"])
        try: fp.run_from_iterable([lig], prot_mol, n_jobs=1, progress=False)
        except TypeError: fp.run_from_iterable([lig], prot_mol, progress=False)
        df = fp.to_dataframe()
        ints = set()
        if not df.empty:
            for col in df.columns:
                if df[col].iloc[0]:
                    _, prot_res, itype = col
                    ints.add(f"{str(prot_res).split('.')[0]}|{itype}")
        return ints
    except Exception:
        return set()


def score_plif(pose_path):
    if pose_path is None: return 0.0, set()
    ints = _plif_contacts(pose_path, C09_RECEPTOR_PDB)
    ref = _load_refs()["ref_plif"]
    if not ints: return 0.0, set()
    ref_keys = set(ref.keys())
    shared = ints & ref_keys
    only_ref = ref_keys - ints
    only_pose = ints - ref_keys
    num = sum(ref[k] for k in shared)
    den = sum(ref[k] for k in shared|only_ref) + len(only_pose)
    return (num/den if den>0 else 0.0), ints


def score_consistency(pose_contacts):
    return 1.0 if any(c.startswith("TYR133") for c in pose_contacts) else 0.0


def score_richness(pose_contacts, cap=5):
    ref = _load_refs()["ref_plif"]
    return min(len(pose_contacts & set(ref.keys())) / cap, 1.0)


def score_smiles(smiles: str, return_breakdown=False, timeout=60):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ({"reward":0.0, "error":"parse_fail", "smiles":smiles}
                if return_breakdown else 0.0)
    def _alarm(s,f): raise TimeoutError()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        s_qed = score_qed(mol)
        s_ecfp = score_ecfp_window(mol)
        s_pharm = score_pharmacophore(mol)
        s_dock, pose = score_dock_c09(mol)
        s_plif, ints = score_plif(pose) if pose else (0.0, set())
        s_cons = score_consistency(ints)
        s_rich = score_richness(ints)
        if pose and os.path.exists(pose): os.unlink(pose)
    except TimeoutError:
        return ({"reward":0.0, "error":"timeout", "smiles":smiles}
                if return_breakdown else 0.0)
    finally:
        signal.alarm(0)
    comps = {"qed":s_qed, "ecfp_window":s_ecfp, "pharm":s_pharm,
             "dock_c09":s_dock, "plif":s_plif,
             "consistency":s_cons, "richness":s_rich}
    reward = sum(WEIGHTS[k]*comps[k] for k in WEIGHTS)
    if return_breakdown:
        return {"reward":reward, **comps, "smiles":smiles}
    return reward


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smiles")
    g.add_argument("--input")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.smiles:
        print(json.dumps(score_smiles(args.smiles, return_breakdown=True), indent=2, default=str))
    else:
        import pandas as pd
        smis = [l.strip() for l in open(args.input) if l.strip() and not l.startswith("#")]
        print(f"scoring {len(smis)} molecules...")
        rows = []
        for i, smi in enumerate(smis):
            r = score_smiles(smi, return_breakdown=True)
            r["index"] = i
            rows.append(r)
            print(f"  {i+1}/{len(smis)}  reward={r['reward']:.3f}  {smi[:60]}")
        out = args.output or args.input + ".rewards.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"saved → {out}")


if __name__ == "__main__":
    main()
