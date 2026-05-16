#!/usr/bin/env python3
"""
v2 ligand prep using Dimorphite-DL for pKa-aware protonation.

Run in the `reinvent4` conda env (Python 3.11):
    conda activate reinvent4
    python3 ligand_prep_v2.py --input fasudil.smi --output ligands/

For each input SMILES:
  1. Enumerate protonation states at pH 6.4–8.4 via Dimorphite-DL (Durrant 2019)
  2. Pick the +1 variant if it exists (matches our WE-derived reference state);
     otherwise pick the neutral one; warn if neither.
  3. Embed 3D with RDKit + MMFF94 optimize
  4. Meeko → PDBQT for Vina

This replaces the SetFormalCharge-on-first-basic-N heuristic in
ablation_parallel.py for the full Task #24 campaign.

Reference: Ropp et al, J. Cheminform. 11:14 (2019)
https://link.springer.com/article/10.1186/s13321-019-0336-9
"""
import argparse, sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from dimorphite_dl import protonate_smiles
from meeko import MoleculePreparation, PDBQTWriterLegacy


def pick_protonated(smi: str, target_charge: int = 1) -> Chem.Mol | None:
    """Return RDKit mol at preferred protonation state, or None if prep fails."""
    variants = protonate_smiles(smi, ph_min=6.4, ph_max=8.4, max_variants=20)
    if not variants:
        return None
    candidates = []
    for v in variants:
        m = Chem.MolFromSmiles(v)
        if m is None:
            continue
        candidates.append((Chem.GetFormalCharge(m), m, v))
    if not candidates:
        return None
    # Preference: target_charge > 0 > others (closest to target_charge)
    candidates.sort(key=lambda x: (abs(x[0] - target_charge), abs(x[0])))
    chosen_charge, mol, smi_proto = candidates[0]
    return mol, smi_proto, chosen_charge


def smiles_to_pdbqt(smi: str, out_path: Path, target_charge: int = 1) -> tuple[bool, str]:
    """Build PDBQT for one SMILES. Returns (success, message)."""
    result = pick_protonated(smi, target_charge=target_charge)
    if result is None:
        return False, "Dimorphite-DL or RDKit could not produce a protonation variant"
    mol, smi_proto, charge = result
    mol = Chem.AddHs(mol)
    rc = AllChem.EmbedMolecule(mol, randomSeed=42)
    if rc != 0:
        return False, f"EmbedMolecule rc={rc} for {smi_proto}"
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception as e:
        return False, f"MMFF optimize failed: {e}"
    prep = MoleculePreparation()
    prep_mols = prep.prepare(mol)
    if not prep_mols:
        return False, "Meeko returned no preparations"
    pdbqt, ok, msg = PDBQTWriterLegacy.write_string(prep_mols[0])
    if not ok:
        return False, f"PDBQT writer failed: {msg}"
    out_path.write_text(pdbqt)
    return True, f"OK (charge {charge:+d}, SMILES: {smi_proto})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="SMILES file (one per line) or single SMILES")
    ap.add_argument("--output", required=True, help="Output directory for PDBQTs")
    ap.add_argument("--target_charge", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.output); out_dir.mkdir(exist_ok=True, parents=True)
    smis = []
    if Path(args.input).exists():
        for line in open(args.input):
            s = line.strip().split()[0] if line.strip() else None
            if s and not s.startswith("#"):
                smis.append(s)
    else:
        smis = [args.input]

    n_ok = 0; n_fail = 0
    for i, smi in enumerate(smis):
        out = out_dir / f"lig_{i:04d}.pdbqt"
        ok, msg = smiles_to_pdbqt(smi, out, target_charge=args.target_charge)
        if ok:
            print(f"  lig_{i:04d}  {msg}")
            n_ok += 1
        else:
            print(f"  lig_{i:04d}  FAIL: {msg}", file=sys.stderr)
            n_fail += 1
    print(f"\n{n_ok}/{n_ok+n_fail} ligands prepared → {out_dir}/")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
