#!/usr/bin/env python3
"""
Tier 1 in-silico validation of cand 4 (benzothiophene-piperidine-piperazine).

  1. PAINS filter — is it a known assay-interference pattern?
  2. Drug-likeness — Lipinski / QED / MW / logP / TPSA
  3. PubChem similarity search — any prior art?
  4. Re-dock cand 4 into ALL 10 contact-map receptors + 1XQ8 — robustness check
  5. PLIF on each receptor — does the binding mode hold across α-syn conformations?

Output: cand4_validation_report.txt + cand4_validation.csv
"""
import os, sys, json, time, subprocess, requests
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, FilterCatalog
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcCrippenDescriptors, CalcTPSA, CalcNumRotatableBonds, CalcNumHBD, CalcNumHBA

# Set thread caps before heavy imports
for k in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ[k] = "1"

ROOT = Path("/home/anugraha/IDP/docking")
ENSEMBLE = ROOT/"receptor_ensemble"
C09_RECEPTOR = ROOT/"c09_receptor.pdb"
RECEPTOR_1XQ8_PDB = ROOT/"receptors/A_1xq8.pdb"
RECEPTOR_1XQ8_PDBQT = ROOT/"receptors/A_1xq8.pdbqt"
REF_PLIF = json.loads((ROOT/"plif/reference_plif.json").read_text())

CAND4_SMILES = "CC(CC1CCN(C(=O)c2csc3ccccc23)CC1)N1CCNCC1"
CAND4_LABEL = "cand4_benzothiophene"
FASUDIL_SMILES = "O=S(=O)(N1CCNCCC1)c1ccnc2ccccc12"

ALL_INTER = ["Hydrophobic","HBDonor","HBAcceptor","PiStacking",
             "PiCation","CationPi","Anionic","Cationic","VdWContact"]
BOX_SIZE = [22.0, 22.0, 22.0]

print("="*78)
print("Cand 4 Tier 1 in-silico validation")
print("="*78)

# ───────────────────────── 1. PAINS filter ────────────────────────────────────
print("\n[1] PAINS / chemistry filters")
m = Chem.MolFromSmiles(CAND4_SMILES)
print(f"  SMILES: {CAND4_SMILES}")
print(f"  formula: {Chem.rdMolDescriptors.CalcMolFormula(m)}")
print(f"  MW: {CalcExactMolWt(m):.2f}")

cat = FilterCatalog.FilterCatalog(FilterCatalog.FilterCatalogParams())
params = FilterCatalog.FilterCatalogParams()
for f in [params.FilterCatalogs.PAINS, params.FilterCatalogs.PAINS_A, params.FilterCatalogs.PAINS_B,
          params.FilterCatalogs.PAINS_C, params.FilterCatalogs.BRENK, params.FilterCatalogs.NIH]:
    params.AddCatalog(f)
catalog = FilterCatalog.FilterCatalog(params)
entries = catalog.GetMatches(m)
if entries:
    for e in entries:
        print(f"  ⚠️ MATCH: {e.GetDescription()}")
else:
    print(f"  ✓ no PAINS / BRENK / NIH filter hits")


# ───────────────────────── 2. Drug-likeness ────────────────────────────────────
print("\n[2] Drug-likeness (Lipinski / QED / Veber)")
mw = CalcExactMolWt(m)
logp = CalcCrippenDescriptors(m)[0]
hbd = CalcNumHBD(m)
hba = CalcNumHBA(m)
tpsa = CalcTPSA(m)
rot = CalcNumRotatableBonds(m)
qed = QED.qed(m)
print(f"  MW = {mw:.1f}     (Lipinski: ≤500)            {'✓' if mw<=500 else '✗'}")
print(f"  cLogP = {logp:.2f} (Lipinski: ≤5)             {'✓' if logp<=5 else '✗'}")
print(f"  HBD = {hbd}       (Lipinski: ≤5)             {'✓' if hbd<=5 else '✗'}")
print(f"  HBA = {hba}       (Lipinski: ≤10)            {'✓' if hba<=10 else '✗'}")
print(f"  TPSA = {tpsa:.1f}  (Veber: ≤140)             {'✓' if tpsa<=140 else '✗'}")
print(f"  rotB = {rot}      (Veber: ≤10)              {'✓' if rot<=10 else '✗'}")
print(f"  QED = {qed:.3f}   (general drug-likeness ~0-1, higher better)")

# CNS-likeness (Wager 2010 MPO)
# Score: 6 props (clogP, clogD, MW, TPSA, HBD, pKa). Higher = more CNS-suitable.
# Simplified: cLogP ≤ 3, MW ≤ 360, TPSA 40–90, HBD ≤ 1
cns_score = 0
cns_score += 1 if logp <= 3 else 0
cns_score += 1 if mw <= 360 else 0
cns_score += 1 if 40 <= tpsa <= 90 else 0
cns_score += 1 if hbd <= 1 else 0
print(f"  CNS-suitability (simplified Wager): {cns_score}/4  (≥3 = likely CNS-permeable)")


# ───────────────────────── 3. ChEMBL/PubChem similarity search ────────────────
print("\n[3] Prior-art search (PubChem similarity)")
try:
    # Use PubChem's REST API for similarity search
    smi_url = CAND4_SMILES.replace("#", "%23").replace("+", "%2B")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/{smi_url}/cids/JSON?Threshold=80&MaxRecords=10"
    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        data = r.json()
        cids = data.get("IdentifierList", {}).get("CID", [])
        print(f"  PubChem CIDs at ≥80% 2D similarity: {len(cids)}")
        if cids:
            print(f"  top 10 CIDs: {cids[:10]}")
            # For first 3 CIDs, get name + ChEMBL ID if available
            for cid in cids[:3]:
                try:
                    name_r = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName,CanonicalSMILES/JSON", timeout=10)
                    if name_r.status_code == 200:
                        props = name_r.json()["PropertyTable"]["Properties"][0]
                        print(f"    CID {cid}: {props.get('IUPACName','?')[:80]}")
                        print(f"      SMILES: {props.get('CanonicalSMILES','?')}")
                except Exception as e:
                    pass
        else:
            print(f"  ✓ No high-similarity (≥80%) compounds in PubChem — chemistry is novel")
    else:
        print(f"  PubChem query failed: HTTP {r.status_code}")
except Exception as e:
    print(f"  PubChem query exception: {e}")


# ───────────────────────── 4. Re-dock cand 4 into all 11 receptors ─────────────
print("\n[4] Cand 4 robustness — re-dock into ALL 11 receptors")
# Build cand4 PDBQT
print("  building cand4 PDBQT...")
m4 = Chem.MolFromSmiles(CAND4_SMILES)
# Protonate the basic amine (the homopiperazine N not adjacent to S)
for atom in m4.GetAtoms():
    if (atom.GetSymbol()=="N" and not atom.GetIsAromatic()
        and atom.GetFormalCharge()==0
        and atom.GetTotalDegree() < 4
        and atom.GetTotalNumHs() >= 1
        and not any(n.GetSymbol() == "S" for n in atom.GetNeighbors())
        and not any(n.GetSymbol() == "C" and any(nn.GetSymbol()=="O" and nn.GetIsAromatic()==False for nn in n.GetNeighbors()) for n in atom.GetNeighbors())):
        atom.SetFormalCharge(+1)
        break
m4.UpdatePropertyCache(strict=False)
m4 = Chem.AddHs(m4)
rc = AllChem.EmbedMolecule(m4, randomSeed=42)
if rc != 0:
    rc = AllChem.EmbedMolecule(m4, randomSeed=137)
AllChem.MMFFOptimizeMolecule(m4)
from meeko import MoleculePreparation, PDBQTWriterLegacy
prep = MoleculePreparation()
prep_mols = prep.prepare(m4)
pdbqt_str, ok, _ = PDBQTWriterLegacy.write_string(prep_mols[0])
CAND4_PDBQT = ROOT/"cand4_validation.pdbqt"
CAND4_PDBQT.write_text(pdbqt_str)
print(f"  ✓ cand4 PDBQT: {CAND4_PDBQT}  ({len(pdbqt_str)} bytes)")
print(f"  protonation: {Chem.MolToSmiles(Chem.RemoveHs(m4))}")
print(f"  formal charge: {Chem.GetFormalCharge(m4):+d}")

# Get box centers for all 11 receptors
def get_center(pdb, resids={133,135,136}):
    pts = []
    for line in open(pdb):
        if line.startswith("ATOM") and line[12:16].strip()=="CA" and int(line[22:26]) in resids:
            pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(pts).mean(axis=0).tolist()

ensemble_manifest = pd.read_csv(ENSEMBLE/"ensemble_manifest.csv")
receptors = [{"name":"1xq8", "pdbqt":RECEPTOR_1XQ8_PDBQT, "pdb":RECEPTOR_1XQ8_PDB,
              "center":get_center(RECEPTOR_1XQ8_PDB), "weight":1.0}]
for _, r in ensemble_manifest.iterrows():
    pdb = ENSEMBLE/r.receptor
    pdbqt = ENSEMBLE/r.receptor.replace(".pdb",".pdbqt")
    receptors.append({"name":Path(r.receptor).stem, "pdbqt":pdbqt, "pdb":pdb,
                      "center":get_center(pdb), "weight":float(r.weight)})

# Also include c09
receptors.append({"name":"c09", "pdbqt":ROOT/"c09_receptor.pdbqt", "pdb":C09_RECEPTOR,
                  "center":get_center(C09_RECEPTOR), "weight":1.0})

from vina import Vina
import MDAnalysis as mda
import prolif as plf
from meeko import PDBQTMolecule, RDKitMolCreate

print(f"\n  Docking + PLIF across {len(receptors)} receptors:")
print(f"  {'receptor':<15} {'Vina E':>8} {'PLIF T':>8} {'TYR133?':>9} {'GLU130?':>9} {'#contacts':>10}")
print(f"  {'-'*75}")
results = []
for r in receptors:
    out_pose = ROOT/f"cand4_val_{r['name']}.pdbqt"
    try:
        v = Vina(sf_name="vina", cpu=4, verbosity=0, seed=42)
        v.set_receptor(str(r["pdbqt"]))
        v.set_ligand_from_file(str(CAND4_PDBQT))
        v.compute_vina_maps(center=r["center"], box_size=BOX_SIZE)
        v.dock(exhaustiveness=16, n_poses=5)
        v.write_poses(str(out_pose), n_poses=5, overwrite=True)
        e = None
        for line in open(out_pose):
            if line.startswith("REMARK VINA RESULT"):
                e = float(line.split()[3]); break
        # PLIF
        pmol = PDBQTMolecule.from_file(str(out_pose))
        mols = RDKitMolCreate.from_pdbqt_mol(pmol)
        lig = plf.Molecule.from_rdkit(mols[0])
        u = mda.Universe(str(r["pdb"]))
        prot = u.select_atoms("protein")
        prot_mol = plf.Molecule.from_mda(prot)
        fp = plf.Fingerprint(ALL_INTER)
        try:
            fp.run_from_iterable([lig], prot_mol, n_jobs=1, progress=False)
        except TypeError:
            fp.run_from_iterable([lig], prot_mol, progress=False)
        df = fp.to_dataframe()
        ints = set()
        if not df.empty:
            for col in df.columns:
                if df[col].iloc[0]:
                    _, prot_res, itype = col
                    ints.add(f"{str(prot_res).split('.')[0]}|{itype}")
        ref_keys = set(REF_PLIF.keys())
        shared = ints & ref_keys
        only_ref = ref_keys - ints
        only_pose = ints - ref_keys
        num = sum(REF_PLIF[k] for k in shared)
        den = sum(REF_PLIF[k] for k in shared|only_ref) + len(only_pose)
        tani = num/den if den>0 else 0.0
        y133 = any("TYR133" in i for i in ints)
        e130 = any("GLU130" in i for i in ints)
        results.append({"receptor":r["name"], "vina_E":e, "plif_T":tani,
                        "TYR133":y133, "GLU130":e130, "n_contacts":len(ints),
                        "contacts":sorted(ints)})
        print(f"  {r['name']:<15} {e:>8.2f} {tani:>8.3f} {'YES' if y133 else 'no':>9} {'YES' if e130 else 'no':>9} {len(ints):>10}")
    except Exception as exc:
        print(f"  {r['name']:<15}  ERROR: {str(exc)[:50]}")
        results.append({"receptor":r["name"], "vina_E":None, "plif_T":0, "TYR133":False, "GLU130":False, "n_contacts":0, "contacts":[]})

# Save
pd.DataFrame(results).to_csv(ROOT/"cand4_validation.csv", index=False)
print(f"\n  saved: cand4_validation.csv")

# Summary
df = pd.DataFrame(results)
df_valid = df.dropna(subset=["vina_E"])
print(f"\n  Summary across {len(df_valid)} successful dockings:")
print(f"    Vina E: mean {df_valid['vina_E'].mean():.2f}, min {df_valid['vina_E'].min():.2f}, max {df_valid['vina_E'].max():.2f}")
print(f"    PLIF Tanimoto: mean {df_valid['plif_T'].mean():.3f}, max {df_valid['plif_T'].max():.3f}")
print(f"    TYR133 contact: {df_valid['TYR133'].sum()}/{len(df_valid)} receptors")
print(f"    GLU130 contact: {df_valid['GLU130'].sum()}/{len(df_valid)} receptors")


# ───────────────────────── 5. Compare cand 4 to fasudil (head-to-head) ─────────
print("\n[5] Head-to-head: cand 4 vs fasudil — on c09 receptor")
# Use fasudil PDBQT we built earlier
FASUDIL_PDBQT = ROOT/"ligand/fasudil.pdbqt"
if FASUDIL_PDBQT.exists():
    fas_pose = ROOT/"fasudil_on_c09.pdbqt"
    v = Vina(sf_name="vina", cpu=4, verbosity=0, seed=42)
    v.set_receptor(str(ROOT/"c09_receptor.pdbqt"))
    v.set_ligand_from_file(str(FASUDIL_PDBQT))
    v.compute_vina_maps(center=get_center(C09_RECEPTOR), box_size=BOX_SIZE)
    v.dock(exhaustiveness=16, n_poses=5)
    v.write_poses(str(fas_pose), n_poses=5, overwrite=True)
    fas_e = float(next(open(fas_pose)).split()[3]) if False else None
    for line in open(fas_pose):
        if line.startswith("REMARK VINA RESULT"):
            fas_e = float(line.split()[3]); break
    cand4_c09_e = next((r["vina_E"] for r in results if r["receptor"]=="c09"), None)
    cand4_c09_plif = next((r["plif_T"] for r in results if r["receptor"]=="c09"), None)
    print(f"  Vina (c09):  cand 4 = {cand4_c09_e:.2f},  fasudil = {fas_e:.2f}")
    print(f"  Δ E (cand4 - fasudil) = {cand4_c09_e - fas_e:+.2f} kcal/mol  ({'cand4 BETTER' if cand4_c09_e < fas_e else 'fasudil better'})")
    print(f"  PLIF Tanimoto cand 4 = {cand4_c09_plif:.3f}")

print("\n" + "="*78)
print("Tier 1 validation complete. Results: cand4_validation.csv")
print("="*78)
