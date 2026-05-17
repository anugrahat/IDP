#!/usr/bin/env python3
"""
v3 ligand prep — Level 3 / Gypsum-DL approach.

For each input SMILES:
  1. Enumerate ALL plausible protonation states at pH 6.4-8.4 via Dimorphite-DL
  2. Keep variants with charge in {-1, 0, +1, +2}
  3. Embed 3D for each, optimize with MMFF, write PDBQT via Meeko
  4. Save manifest mapping parent_smiles → list of variant PDBQTs + their charges

The downstream docking pipeline docks EACH variant separately and picks the
best-scoring one per (parent, receptor).  Letting Vina select the protonation
state in the receptor context is the rigorous standard (Ropp 2019, DOCKSTRING).

Run in the reinvent4 env (Python 3.11):
    pixi run -e reinvent python3 docking/ligand_prep_v3.py \
        --input reinvent_work/all_smiles.smi \
        --output docking/candidates_v3
"""
import argparse, sys, json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from dimorphite_dl import protonate_smiles
from meeko import MoleculePreparation, PDBQTWriterLegacy


def enumerate_variants(parent_smi: str, ph_min=6.4, ph_max=8.4, max_variants=20):
    """Return [(variant_smiles, charge, protonation_site_type)]."""
    variants = protonate_smiles(parent_smi, ph_min=ph_min, ph_max=ph_max,
                                max_variants=max_variants)
    out = []
    for v in variants:
        m = Chem.MolFromSmiles(v)
        if m is None: continue
        ch = Chem.GetFormalCharge(m)
        # Drop very high/low charge states (numerical edge cases)
        if abs(ch) > 2: continue
        # Identify the protonated atom(s) for diagnostic labelling
        sites = []
        for a in m.GetAtoms():
            if a.GetFormalCharge() != 0:
                kind = "aromatic" if a.GetIsAromatic() else "aliphatic"
                near_s = any(n.GetSymbol() == "S" for n in a.GetNeighbors())
                sites.append(f"{a.GetSymbol()}({kind}{'+S' if near_s else ''})")
        out.append((v, ch, "+".join(sites) or "neutral"))
    return out


def variant_to_pdbqt(smi: str, out_path: Path) -> bool:
    m = Chem.MolFromSmiles(smi)
    if m is None: return False
    m = Chem.AddHs(m)
    rc = AllChem.EmbedMolecule(m, randomSeed=42)
    if rc != 0:
        # Try a second seed
        rc = AllChem.EmbedMolecule(m, randomSeed=137)
        if rc != 0: return False
    try:
        AllChem.MMFFOptimizeMolecule(m)
    except Exception:
        pass   # MMFF can fail for unusual cations; geometry from Embed is still OK
    prep = MoleculePreparation()
    try:
        prep_mols = prep.prepare(m)
    except Exception:
        return False
    if not prep_mols: return False
    pdbqt, ok, _ = PDBQTWriterLegacy.write_string(prep_mols[0])
    if not ok: return False
    out_path.write_text(pdbqt)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_variants", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.output); out_dir.mkdir(exist_ok=True, parents=True)
    inp = Path(args.input)
    parent_smis = []
    if inp.exists():
        for line in open(inp):
            s = line.strip().split()[0] if line.strip() else None
            if s and not s.startswith("#"):
                parent_smis.append(s)
    else:
        parent_smis = [args.input]

    manifest = []
    for i, parent_smi in enumerate(parent_smis):
        variants = enumerate_variants(parent_smi, max_variants=args.max_variants)
        print(f"\ncand_{i:02d}  parent: {parent_smi}")
        print(f"           variants: {len(variants)}")
        if not variants:
            print(f"           NO VARIANTS PRODUCED — skipping")
            continue
        cand_variants = []
        for j, (vsmi, ch, site) in enumerate(variants):
            pdbqt = out_dir / f"cand{i:02d}_var{j:02d}_chg{ch:+d}.pdbqt"
            ok = variant_to_pdbqt(vsmi, pdbqt)
            status = "OK" if ok else "FAIL"
            print(f"             var{j:02d}  charge {ch:+d}  site={site:<25}  {status}  {vsmi}")
            if ok:
                cand_variants.append({"var_idx": j, "smiles": vsmi, "charge": ch,
                                       "site": site, "pdbqt": str(pdbqt)})
        manifest.append({"cand_idx": i, "parent_smiles": parent_smi, "variants": cand_variants})

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    n_var = sum(len(c["variants"]) for c in manifest)
    print(f"\n[done] {len(manifest)} parents × ~{n_var/len(manifest):.1f} variants = {n_var} PDBQTs")
    print(f"       manifest: {manifest_path}")


if __name__ == "__main__":
    main()
