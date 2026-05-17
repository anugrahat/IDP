#!/usr/bin/env python3
"""
v4 ligand prep — Level 2 with chemistry-aware pKa filter.

Pipeline per parent SMILES:
  1. Enumerate variants via Dimorphite-DL (pH 6.4–8.4)
  2. Filter: keep only variants where every protonated atom is chemically
     realistic at physiological pH (literature pKa).  Specifically:
        ✓ aliphatic amine, not adjacent to S (basic amines, pKa 9-11)
        ✓ neutral, anything
        ✗ aromatic N+ (pyridine/isoquinoline N, pKa ≤5.2)
        ✗ sulfonamide N+ (pKa <0)
        ✗ aniline N+ (aromatic NH2+, pKa ≤5)
  3. For each surviving variant: embed 3D, MMFF, write PDBQT via Meeko.

Output: candidates_v4/manifest.json with one variant per row.

Run in reinvent4 env (py 3.11):
    python3 docking/ligand_prep_v4.py --input X.smi --output candidates_v4
"""
import argparse, sys, json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from dimorphite_dl import protonate_smiles
from meeko import MoleculePreparation, PDBQTWriterLegacy


def site_chemically_reasonable(atom) -> tuple[bool, str]:
    """Return (keep, label) for a protonated atom.  False → unphysical at pH 7.4."""
    if atom.GetFormalCharge() == 0:
        return True, "neutral"
    if atom.GetSymbol() != "N":
        # Anionic O- (deprotonated carboxylate) etc. — accept if -1
        return atom.GetFormalCharge() < 0, f"non-N ({atom.GetSymbol()})"

    # N+ sites
    neighbors = list(atom.GetNeighbors())
    has_S = any(n.GetSymbol() == "S" for n in neighbors)
    has_aromatic_neighbor = any(n.GetIsAromatic() for n in neighbors)

    if atom.GetIsAromatic():
        # Aromatic N+ (pyridine-like): pKa ~5 — unphysical at pH 7.4
        return False, "aromatic-N+ (pKa ~5, unphysical at pH 7.4)"

    if has_S:
        # Sulfonamide N+: pKa <0 — never protonated in water
        return False, "sulfonamide-N+ (pKa <0, never protonated)"

    if has_aromatic_neighbor and atom.GetTotalNumHs() >= 2:
        # Aniline NH2+ — pKa ~4.6
        return False, "aniline-N+ (pKa ~4.6, unphysical at pH 7.4)"

    # Aliphatic amine (primary, secondary, tertiary) → basic, pKa 9-11 → keep
    return True, "aliphatic-N+ (basic, pKa ~9-11, kept)"


def variant_is_acceptable(smi: str) -> tuple[bool, str, int]:
    """Return (keep, reason, total_charge)."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False, "parse failed", 0
    total_charge = Chem.GetFormalCharge(m)
    reasons = []
    for atom in m.GetAtoms():
        if atom.GetFormalCharge() != 0:
            ok, why = site_chemically_reasonable(atom)
            reasons.append((atom.GetIdx(), atom.GetSymbol(), atom.GetFormalCharge(), ok, why))
    bad = [r for r in reasons if not r[3]]
    if bad:
        return False, "; ".join(r[4] for r in bad), total_charge
    return True, "all sites OK", total_charge


def variant_to_pdbqt(smi: str, out_path: Path) -> bool:
    m = Chem.MolFromSmiles(smi)
    if m is None: return False
    m = Chem.AddHs(m)
    if AllChem.EmbedMolecule(m, randomSeed=42) != 0:
        if AllChem.EmbedMolecule(m, randomSeed=137) != 0:
            return False
    try: AllChem.MMFFOptimizeMolecule(m)
    except Exception: pass
    prep = MoleculePreparation()
    try: prep_mols = prep.prepare(m)
    except Exception: return False
    if not prep_mols: return False
    pdbqt, ok, _ = PDBQTWriterLegacy.write_string(prep_mols[0])
    if not ok: return False
    out_path.write_text(pdbqt)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_variants", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.output); out_dir.mkdir(exist_ok=True, parents=True)
    smis = [l.strip().split()[0] for l in open(args.input) if l.strip() and not l.startswith("#")]

    manifest = []
    print(f"{'cand':>5} {'kept/enum':<10} {'charge':>7} {'reason':<40} SMILES")
    print("-"*150)
    for i, parent_smi in enumerate(smis):
        all_vars = protonate_smiles(parent_smi, ph_min=6.4, ph_max=8.4,
                                    max_variants=args.max_variants)
        kept = []
        for j, vsmi in enumerate(all_vars):
            ok, reason, charge = variant_is_acceptable(vsmi)
            if not ok:
                continue
            pdbqt_path = out_dir / f"cand{i:02d}_var{len(kept):02d}_chg{charge:+d}.pdbqt"
            if not variant_to_pdbqt(vsmi, pdbqt_path):
                continue
            kept.append({
                "var_idx": len(kept), "smiles": vsmi, "charge": charge,
                "reason": reason, "pdbqt": str(pdbqt_path)
            })
            print(f"  {i:02d}  {len(kept):>1}/{len(all_vars):<6} {charge:>+5}  {reason:<40} {vsmi}")
        if not kept:
            print(f"  {i:02d}  NONE/{len(all_vars):<6} no acceptable variants — falling back to neutral parent")
            # Force-add neutral
            pdbqt_path = out_dir / f"cand{i:02d}_var00_chg+0.pdbqt"
            if variant_to_pdbqt(parent_smi, pdbqt_path):
                kept.append({"var_idx": 0, "smiles": parent_smi, "charge": 0,
                             "reason": "fallback neutral", "pdbqt": str(pdbqt_path)})
        manifest.append({"cand_idx": i, "parent_smiles": parent_smi, "variants": kept})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(len(c["variants"]) for c in manifest)
    print(f"\n[done] {len(manifest)} parents, {total} variants kept, avg {total/len(manifest):.1f}/cand")
    print(f"       manifest: {out_dir/'manifest.json'}")
    # Summary stats
    counts = [len(c['variants']) for c in manifest]
    print(f"       per-cand: min={min(counts)} median={sorted(counts)[len(counts)//2]} max={max(counts)}")


if __name__ == "__main__":
    main()
