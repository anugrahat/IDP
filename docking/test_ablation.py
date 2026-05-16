"""
Test-driven debugging for the WE → REINVENT ablation pipeline.

Each test captures ONE specific behavior we expect. Run with:
    pytest -x -v test_ablation.py

Tests are ordered roughly by pipeline stage. If an early test fails, fix it
before looking at later ones.
"""
import pytest
import subprocess, json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
import MDAnalysis as mda
import prolif as plf

ROOT = Path("/home/anugraha/IDP/docking")
RECEPTOR_1XQ8_PDB = ROOT / "receptors/A_1xq8.pdb"
ENSEMBLE_DIR = ROOT / "receptor_ensemble"
REF_PLIF_PATH = ROOT / "plif/reference_plif.json"

FASUDIL_NEUTRAL = "O=S(=O)(N1CCNCCC1)c1ccnc2ccccc12"
SAMPLE_ANALOG = "CN1CCN(S(=O)(=O)c2ccnc3ccccc23)CC1"   # from mol2mol output


# ──────────────────────────────── stage 1: LIGAND PREP ─────────────────────────

def _build_protonated(smi: str):
    """
    The protonation function under test.

    PROPER RDKit pattern (per web search):
      - SetFormalCharge(+1) on basic aliphatic N
      - call UpdatePropertyCache()
      - then AddHs() (it will assign the right H count automatically)

    No manual SetNumExplicitHs — that was the bug.
    """
    m = Chem.MolFromSmiles(smi)
    assert m is not None, f"Could not parse SMILES: {smi}"
    # Protonate the first basic aliphatic N (not aromatic, not sulfonamide)
    for atom in m.GetAtoms():
        if (atom.GetSymbol() == "N"
            and not atom.GetIsAromatic()
            and atom.GetFormalCharge() == 0
            and atom.GetTotalDegree() < 4
            and atom.GetTotalNumHs() >= 1
            and not any(n.GetSymbol() == "S" for n in atom.GetNeighbors())):
            atom.SetFormalCharge(+1)
            break
    m.UpdatePropertyCache(strict=False)
    m = Chem.AddHs(m)
    return m


def test_protonate_fasudil_gives_plus_one_charge():
    """Fasudil's homopiperazine N should get +1 charge, valence stays sane."""
    m = _build_protonated(FASUDIL_NEUTRAL)
    assert Chem.GetFormalCharge(m) == +1, \
        f"Fasudil should be +1, got {Chem.GetFormalCharge(m)}"
    # No atom should have illegal valence
    for atom in m.GetAtoms():
        # RDKit raises if illegal, so just sanitize
        pass
    Chem.SanitizeMol(m)


def test_protonate_fasudil_embeds_in_3d():
    """Embedding must succeed (return code 0) and produce one conformer."""
    m = _build_protonated(FASUDIL_NEUTRAL)
    rc = AllChem.EmbedMolecule(m, randomSeed=42)
    assert rc == 0, f"EmbedMolecule failed with code {rc}"
    assert m.GetNumConformers() == 1


def test_protonate_sample_analog_succeeds():
    """A Mol2Mol-generated fasudil analog must prep without valence errors."""
    m = _build_protonated(SAMPLE_ANALOG)
    rc = AllChem.EmbedMolecule(m, randomSeed=42)
    assert rc == 0, f"EmbedMolecule failed for {SAMPLE_ANALOG}, code {rc}"
    Chem.SanitizeMol(m)


def test_protonate_bad_smiles_raises_loudly():
    """Bad SMILES must raise, not silently return None — fail-loud principle."""
    with pytest.raises(AssertionError):
        _build_protonated("not_a_smiles_xxxxxx")


def test_protonate_no_basic_amine_still_works():
    """A molecule with no basic amine (e.g., benzoic acid) must not crash."""
    m = _build_protonated("O=C(O)c1ccccc1")
    assert Chem.GetFormalCharge(m) == 0
    Chem.SanitizeMol(m)


# ────────────────────────────── stage 2: RECEPTOR PDB LOADABILITY ──────────────

def test_1xq8_receptor_loadable_by_mda():
    """1XQ8 PDB must produce an RDKit-convertible MDAnalysis Universe."""
    u = mda.Universe(str(RECEPTOR_1XQ8_PDB))
    prot = u.select_atoms("protein")
    assert len(prot) > 0
    plf.Molecule.from_mda(prot)   # would raise on conversion failure


def test_we_receptor_loadable_by_mda():
    """
    WE receptor PDB must also work — currently FAILS with 'Explicit valence
    for atom # 447 H, 2'.  Fix is: re-protonate the WE receptor PDB.
    """
    pdb = ENSEMBLE_DIR / "receptor_03.pdb"   # the iter28/seg39-family receptor
    assert pdb.exists()
    u = mda.Universe(str(pdb))
    prot = u.select_atoms("protein")
    plf.Molecule.from_mda(prot)   # would raise on conversion failure


# ────────────────────────────── stage 3: REFERENCE PLIF ──────────────────────

def test_reference_plif_exists():
    assert REF_PLIF_PATH.exists()


def test_reference_plif_contains_y133_pistacking():
    """We know from WE that fasudil π-stacks Y133 — must be in the reference."""
    ref = json.loads(REF_PLIF_PATH.read_text())
    assert "TYR133|PiStacking" in ref, \
        f"Reference missing TYR133|PiStacking. Keys: {list(ref.keys())[:10]}"
    assert ref["TYR133|PiStacking"] > 0.3, \
        f"TYR133|PiStacking freq too low: {ref['TYR133|PiStacking']}"


# ────────────────────────────── stage 4: POSE PLIF ──────────────────────────

def _pose_plif(pose_pdbqt: Path, receptor_pdb: Path):
    """Compute PLIF for rank-1 pose. Returns set of "RESNAME###|TYPE" strings."""
    pose_sdf = pose_pdbqt.with_suffix(".pose1_test.sdf")
    r = subprocess.run(
        ["obabel", str(pose_pdbqt), "-O", str(pose_sdf), "-f", "1", "-l", "1"],
        capture_output=True, text=True
    )
    assert r.returncode == 0 and pose_sdf.exists(), f"obabel failed: {r.stderr[-200:]}"
    m = Chem.MolFromMolFile(str(pose_sdf), removeHs=False)
    assert m is not None, f"RDKit could not read {pose_sdf}"
    lig = plf.Molecule.from_rdkit(m)
    u = mda.Universe(str(receptor_pdb))
    prot = u.select_atoms("protein")
    prot_mol = plf.Molecule.from_mda(prot)
    fp = plf.Fingerprint(["Hydrophobic","HBDonor","HBAcceptor","PiStacking",
                          "PiCation","CationPi","Anionic","Cationic","VdWContact"])
    fp.run_from_iterable([lig], prot_mol, progress=False)
    df = fp.to_dataframe()
    ints = set()
    if not df.empty:
        for col in df.columns:
            if df[col].iloc[0]:
                _, prot_res, itype = col
                ints.add(f"{str(prot_res).split('.')[0]}|{itype}")
    return ints


def test_pose_plif_runs_on_1xq8_pose():
    """PLIF for any cand_X docked into 1XQ8 should return ≥1 interaction."""
    pose = ROOT / "ablation_poses/cand06_1xq8.pdbqt"
    assert pose.exists()
    ints = _pose_plif(pose, RECEPTOR_1XQ8_PDB)
    assert len(ints) > 0, "Pose PLIF returned no interactions — pose may be wrong"


def test_pose_plif_runs_on_we_receptor():
    """
    PLIF for cand_X docked into a WE receptor must run without crashing.
    Currently FAILS — 'Explicit valence for atom # 447 H'.
    """
    pose = ROOT / "ablation_poses/cand06_receptor_03.pdbqt"
    assert pose.exists()
    receptor = ENSEMBLE_DIR / "receptor_03.pdb"
    ints = _pose_plif(pose, receptor)
    assert len(ints) > 0, "WE-receptor PLIF returned 0 — likely a silent crash"


# ────────────────────────────── stage 5: TANIMOTO ──────────────────────────

def _tanimoto(pose_keys, ref_plif):
    if not pose_keys: return 0.0
    ref_keys = set(ref_plif.keys())
    shared = pose_keys & ref_keys
    only_ref = ref_keys - pose_keys
    only_pose = pose_keys - ref_keys
    num = sum(ref_plif[k] for k in shared)
    den = sum(ref_plif[k] for k in shared | only_ref) + len(only_pose)
    return num / den if den > 0 else 0.0


def test_tanimoto_known_identical_to_one():
    """Pose ≡ Reference → Tanimoto = 1.0 (perfect match)."""
    ref = {"TYR133|PiStacking": 1.0, "GLU130|Cationic": 0.5}
    pose = {"TYR133|PiStacking", "GLU130|Cationic"}
    t = _tanimoto(pose, ref)
    assert abs(t - 1.0) < 1e-6, f"Expected 1.0, got {t}"


def test_tanimoto_disjoint_is_zero():
    """No shared contacts → Tanimoto = 0.0."""
    ref = {"TYR133|PiStacking": 1.0}
    pose = {"TYR99|Hydrophobic"}
    assert _tanimoto(pose, ref) == 0.0


def test_tanimoto_empty_pose_is_zero():
    assert _tanimoto(set(), {"TYR133|PiStacking": 1.0}) == 0.0


# ────────────────────────────── stage 6: END-TO-END ──────────────────────

def test_fasudil_self_dock_recovers_some_reference_contacts():
    """
    Critical: docking fasudil back into receptor_03 (its OWN bound-family
    receptor) should reproduce SOME of the reference PLIF.  If even fasudil
    can't reproduce itself, the pipeline is fundamentally broken — we'd
    expect ≥1 shared interaction with the reference.
    """
    # This test depends on a fasudil-into-receptor_03 pose existing.
    pose = ROOT / "fasudil_self_dock_receptor_03.pdbqt"
    if not pose.exists():
        pytest.skip(f"fasudil self-dock pose not generated yet ({pose})")
    receptor = ENSEMBLE_DIR / "receptor_03.pdb"
    ints = _pose_plif(pose, receptor)
    ref = json.loads(REF_PLIF_PATH.read_text())
    shared = ints & set(ref.keys())
    assert len(shared) >= 1, (
        f"Fasudil self-dock recovers ZERO reference contacts — pipeline broken.\n"
        f"  Pose contacts: {sorted(ints)}\n"
        f"  Reference top: {list(ref.keys())[:8]}"
    )
