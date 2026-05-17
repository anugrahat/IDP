#!/usr/bin/env python3
"""
Curated list of published small-molecule α-synuclein binders + similarity
analysis vs our SC4_c09 top-5 candidates (including cand 4).

References:
  - Fasudil: Robustelli, Piana, Shaw 2022 JACS (Kd ~3 mM, dynamic shuttling)
  - Anle138b: Wagner 2013 Acta Neuropathol (PD/MSA clinical candidate, aggregation inhibitor)
  - SynuClean-D: Pujols 2018 Sci Rep (Y-shape aggregation inhibitor)
  - EGCG: Bieschke 2010 PNAS (polyphenol, redirects aggregation)
  - Quercetin / Curcumin: multiple aggregation papers
  - Methylene blue: Levin 2010 (aggregation modulator)
  - Tolcapone: 2017 NMR evidence of α-syn binding
  - Sephin1: Krzyzosiak 2018 (ISR modulator, indirect)
  - Doxycycline: Gonzalez-Lizarraga 2017 (off-pathway inhibitor)
  - ATH434 (PBT434): de novo iron-binding for α-syn aggregation
  - Ligand 23 (Baidya 2024): 4-aminoquinoline + sulfonamide
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw, Descriptors
from pathlib import Path

LITERATURE = {
    # Direct binders to monomeric α-syn (NMR or biophysical evidence)
    "fasudil":        "O=S(=O)(N1CCNCCC1)c1ccnc2ccccc12",
    "tolcapone":      "Cc1ccc(C(=O)c2ccc(O)c(O)c2[N+](=O)[O-])cc1",
    # Aggregation inhibitors with α-syn affinity reported
    "anle138b":       "Brc1cccc(-c2cc(-c3ccc4c(c3)OCO4)[nH]n2)c1",
    "synuclean-D":    "O=C1NC(=O)/C(=C\\c2cc(Cl)cc(Cl)c2O)C1=O",
    "EGCG":           "OC(=O)c1cc(O)c(O)c(O)c1.Oc1cc2c(cc1O)O[C@@H](c1cc(O)c(O)c(O)c1)[C@@H](O)C2",
    "quercetin":      "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
    "curcumin":       "O=C(\\C=C\\c1ccc(O)c(OC)c1)CC(=O)\\C=C\\c1ccc(O)c(OC)c1",
    "methylene blue": "CN(C)c1ccc2nc3ccc(N(C)C)cc3[s+]c2c1",
    "doxycycline":    "C[C@@H]1c2cccc(O)c2C(=O)C2=C(O)[C@@]3(O)C(=O)C(C(N)=O)=C(O)[C@@H](N(C)C)[C@@H]3[C@H](O)[C@@H]12",
    "ATH434":         "OC1=CC=C(C=C1)CC(=O)C2=NC=CC=C2",   # PBT434, iron chelator
    "sephin1":        "O=C(NC1=CC=CC=C1)NC2=NN=CS2",
    # Robustelli 2022 hits beyond fasudil (compounds 2-4 from the JACS paper)
    "ligand_2":       "O=S(=O)(N1CCNCC1)c1ccc(Cl)cc1",
    # Baidya 2024 hit
    "baidya_q23":     "Nc1ccnc2cc(S(=O)(=O)N3CCCCC3)ccc12",
}

# Our top-5 from SC4_c09
import json
mani = json.loads(Path("/home/anugraha/IDP/docking/candidates_v4_full/manifest.json").read_text())
df_res = pd.read_csv("/home/anugraha/IDP/docking/stage5_v2_c09_results.csv")
top5 = df_res.nlargest(5, "SC4_c09")[["id","smiles","SC4_c09","plif_c09","score_c09"]]

# Fingerprints
def fp(s):
    m = Chem.MolFromSmiles(s)
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None

lit_fps = {n: fp(s) for n, s in LITERATURE.items()}
cand_fps = {row.id: fp(row.smiles) for _, row in top5.iterrows()}

# Similarity matrix
print(f"\n══════ Tanimoto similarity: SC4_c09 top-5 vs published α-syn binders ══════")
print(f"\n{'lit binder':<18}", end="")
for cid in cand_fps: print(f"  cand_{cid:>3}", end="")
print(f"  fasudil_ref")
print("  " + "-" * (18 + 12*(len(cand_fps)+1)))
for name, lfp in lit_fps.items():
    print(f"  {name:<16}", end="")
    for cid, cfp in cand_fps.items():
        if lfp and cfp:
            t = DataStructs.TanimotoSimilarity(lfp, cfp)
            print(f"   {t:6.3f}  ", end="")
        else:
            print(f"   {'?':>6}  ", end="")
    # fasudil self-similarity to lit binder (reference column)
    fas_t = DataStructs.TanimotoSimilarity(lit_fps['fasudil'], lfp) if lfp else 0
    print(f"   {fas_t:6.3f}")

# Best lit match per candidate
print(f"\n══════ Best literature match for each top-5 candidate ══════")
for cid, cfp in cand_fps.items():
    best_name, best_t = None, 0
    for name, lfp in lit_fps.items():
        if lfp:
            t = DataStructs.TanimotoSimilarity(lfp, cfp)
            if t > best_t:
                best_t = t; best_name = name
    smi = top5[top5.id==cid].iloc[0].smiles
    print(f"  cand_{cid:>3} (SMILES: {smi[:60]:<60}) → closest to {best_name} (T={best_t:.3f})")

# Save
out = []
for cid, cfp in cand_fps.items():
    row = {"id": cid, "smiles": top5[top5.id==cid].iloc[0].smiles}
    for name, lfp in lit_fps.items():
        row[f"T_{name}"] = DataStructs.TanimotoSimilarity(lfp, cfp) if lfp else None
    out.append(row)
pd.DataFrame(out).to_csv("/home/anugraha/IDP/docking/cand_vs_literature.csv", index=False)
print(f"\nfull matrix → /home/anugraha/IDP/docking/cand_vs_literature.csv")
