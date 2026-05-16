#!/usr/bin/env python3
"""Build fasudil cleanly from SMILES, generate 3D, save PDBQT for Vina."""
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from meeko import MoleculePreparation, PDBQTWriterLegacy

# Fasudil at physiological pH: homopiperazine 4-N is protonated (pKa ≈ 8)
# Written directly with [NH2+] to avoid valence ambiguity.
FASUDIL_SMILES_PROTONATED = "O=S(=O)(N1CC[NH2+]CCC1)c1ccnc2ccccc12"

m = Chem.MolFromSmiles(FASUDIL_SMILES_PROTONATED)
if m is None:
    raise SystemExit("SMILES parse failed")
m = Chem.AddHs(m)
print("After protonation:")
print(f"  formula: {Chem.rdMolDescriptors.CalcMolFormula(m)}")
print(f"  formal charge: {Chem.GetFormalCharge(m)}")
print(f"  SMILES: {Chem.MolToSmiles(m)}")

# Embed 3D
AllChem.EmbedMolecule(m, randomSeed=42)
AllChem.MMFFOptimizeMolecule(m)

# Meeko prep
prep = MoleculePreparation()
prep_mols = prep.prepare(m)
pdbqt_str, ok, msg = PDBQTWriterLegacy.write_string(prep_mols[0])
if not ok:
    print(f"Meeko failed: {msg}")
    exit(1)
with open("ligand/fasudil.pdbqt", "w") as f:
    f.write(pdbqt_str)
print(f"\nfasudil.pdbqt written ({len(pdbqt_str)} bytes)")

# Also save 2D image for inspection
Draw.MolToFile(m, "ligand/fasudil_2d.png", size=(400, 400))

# Print summary
with open("ligand/fasudil.pdbqt") as f:
    lines = f.readlines()
n_atoms = sum(1 for l in lines if l.startswith("ATOM"))
n_torsions = sum(1 for l in lines if "BRANCH" in l and "ENDBRANCH" not in l)
print(f"  atoms: {n_atoms}, torsions: {n_torsions}")
