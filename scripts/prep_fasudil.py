"""
Build the physiological cationic form of fasudil (+1) from scratch using SMILES.

Fasudil's diazepane secondary amine (pKa ~9.5) is protonated at pH 7.4.
The isoquinoline N (pKa ~5.5) is mostly neutral and is left alone.
Drug form is fasudil-HCl, net charge +1.

Emit a PDB with explicit Hs for antechamber.
"""
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

# Protonated diazepane N in the middle of the 7-ring; isoquinoline N left neutral.
PROT_SMILES = "C1C[NH2+]CCN(C1)S(=O)(=O)C2=CC=CC3=C2C=CN=C3"
OUT_PDB = Path("/home/anugraha/IDP/inputs/ligand/fasudil_prot.pdb")
OUT_SDF = Path("/home/anugraha/IDP/inputs/ligand/fasudil_prot.sdf")

mol = Chem.MolFromSmiles(PROT_SMILES)
mol = Chem.AddHs(mol)

# 3D embedding + MMFF94s minimisation
params = AllChem.ETKDGv3()
params.randomSeed = 1
AllChem.EmbedMolecule(mol, params)
AllChem.MMFFOptimizeMolecule(mol, maxIters=500, mmffVariant="MMFF94s")

heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
total_q = sum(a.GetFormalCharge() for a in mol.GetAtoms())
print(f"Heavy atoms: {heavy}, total atoms: {mol.GetNumAtoms()}, net charge: {total_q}")
print(f"SMILES (kekulised): {Chem.MolToSmiles(mol, kekuleSmiles=True)}")

# Tag every atom into residue FAS chain L for clean PDB
for a in mol.GetAtoms():
    info = Chem.AtomPDBResidueInfo()
    info.SetResidueName("FAS")
    info.SetResidueNumber(1)
    info.SetChainId("L")
    info.SetName(f" {a.GetSymbol():<3}")
    a.SetMonomerInfo(info)

Chem.MolToPDBFile(mol, str(OUT_PDB))
w = Chem.SDWriter(str(OUT_SDF)); w.write(mol); w.close()
print(f"wrote {OUT_PDB}")
print(f"wrote {OUT_SDF}")
