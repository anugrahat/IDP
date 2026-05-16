# IDP Drug Design — α-synuclein × fasudil → REINVENT

WE-MD characterization of fasudil binding to intrinsically disordered α-synuclein, feeding into REINVENT generative ML for novel α-syn binder design.

## Project

UC Davis, ahn lab. Sprint started 2026-05-15. See [`plan.md`](plan.md) for the full plan.

**Stack:**
- WESTPA 2022.11 + Amber pmemd.cuda + NVIDIA MPS (4 × A100 80GB, 48 walkers)
- a99SB-disp + TIP4P-D force field (Robustelli/Piana/Shaw 2018)
- GAFF2 + AM1-BCC for fasudil
- AutoDock Vina + Meeko + RDKit for docking
- ProLIF for interaction fingerprints
- REINVENT 4 (Mol2Mol prior, fasudil-seeded)
- ablation/control infra to compare WE-aware vs WE-free scoring chains

## Key results so far

| Experiment | Result | Path |
|---|---|---|
| WE run | iter 73+/500, ~7.3 ns/walker, helix-1 melting 100% → 57% | `westpa/` |
| Bound pose found | iter 28, seg 39, frame 92 — Y133 π-stack at 3.86 Å | `bound_pose_protein_ligand.pdb` |
| Receptor ensemble | 10 cluster medoids from 7,374 bound frames | `docking/receptor_ensemble/` |
| 1XQ8 vs WE docking control | A_blind 0% / A_biased 65% / C_biased 100% hit rate to Y133/D135/Y136 | `docking/summary.txt` |
| Reference PLIF | TYR133 π-stack 0.63, hydrophobic 0.99 | `docking/plif/reference_plif.json` |
| Test suite | 14/14 passing | `docking/test_ablation.py` |

## Repo layout (committed files)

```
plan.md                      sprint plan + decisions
README.md                    this file

westpa/                      WESTPA config + scripts (data excluded)
  west.cfg                   2D pcoord, MABBinMapper, 48 walkers
  run_WE.sbatch              7-day SLURM job, MPS-shared GPUs
  env.sh                     module loads + conda activation
  init.sh                    fail-CLOSED safety guard against re-init
  westpa_scripts/            pmemd runseg + MDAnalysis pcoord
  bstates/                   basis state (small)

build/                       system builder
  build_complex.tleap        tleap script
  fasudil*.mol2              GAFF2-parameterized ligand
  asyn_extended.pdb          α-syn starting structure (1XQ8-derived)

equilibrate/                 NVT/NPT equilibration scripts

inputs/                      original PDB inputs (1XQ8 etc.)

docking/                     1XQ8-vs-WE control + ablation
  prep_and_dock.py           three-condition docking
  prep_fasudil.py            SMILES→PDBQT (fasudil reference)
  extract_ensemble.py        cluster bound frames → 10 receptors
  compute_reference_plif.py  ProLIF on WE bound ensemble
  ablation_parallel.py       parallel 4-chain ablation
  test_ablation.py           pytest suite (14 tests)
  run_ablation.sbatch        SLURM submission
  summary.txt                A/B/C hit-rate table
  receptor_ensemble/         10 WE-derived receptor PDBs + manifest
  plif/reference_plif.json   fasudil's reference interaction signature
  ablation_scores.csv        per-candidate, per-chain scores

reinvent_work/               REINVENT 4 configs + candidate generation
  smoke_sampling.toml        sanity test config
  mol2mol_fasudil.toml       Mol2Mol on fasudil
  fasudil.smi                seed SMILES
  mol2mol_fasudil.csv        20 fasudil analogues

scripts/                     misc utilities (ligand placement, etc.)
```

## Excluded from git

WE trajectories (`westpa/traj_segs/`, 1.2 TB), AMBER topologies, REINVENT priors (80 MB each), and cloned external repos. See `.gitignore`.

## Reproduce

1. WE: `cd westpa && sbatch run_WE.sbatch`
2. Receptor ensemble: `cd docking && python3 extract_ensemble.py`
3. Reference PLIF: `python3 compute_reference_plif.py`
4. Control experiment: `python3 prep_and_dock.py`
5. Tests: `pytest test_ablation.py -v`
6. Ablation: `sbatch run_ablation.sbatch`

## Tooling notes

- Protonation pattern: `SetFormalCharge` + `UpdatePropertyCache` then `AddHs` (canonical RDKit, per BLOPIG / rdkit-discuss). Dimorphite-DL is the SOTA — TODO for Task #24 full campaign.
- WE receptor PDB prep: strip Hs, re-add via `obabel -p 7.4` (otherwise AMBER H naming confuses MDAnalysis→RDKit conversion).
- Vina runs CPU-only here; Vina-GPU 2.1 cloned (not yet compiled) for the full Task #24 campaign.
