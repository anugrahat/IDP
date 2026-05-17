# IDP Drug Design Plan — α-syn + fasudil → REINVENT

**Goal:** Use weighted-ensemble MD to characterize fasudil binding to α-synuclein, then use that data to drive REINVENT to generate novel α-syn binders that beat fasudil on affinity, residence time, and binding-mode fidelity.

**Sprint start:** 2026-05-15 · **Last updated:** 2026-05-16 21:55

## Status snapshot (2026-05-16 evening)

| Pipeline stage | Status |
|---|---|
| WE production (job 794545, 7-day budget) | running, iter 126/500 |
| 100-seed docking control (paper figure) | ✅ done — A_blind 5.8%, A_biased 88.5%, C_biased 100% hit rate to Y133/D135/Y136 |
| WE-weighted conformational PCA | ✅ done — 10 PCA clusters, **cluster_09 = sole bound state (6.12% bound, 4.8% pop)** |
| Reference PLIF computation | ✅ done — TYR133 PiStacking 0.63, hydrophobic 0.99 |
| 172-candidate REINVENT generation (Mol2Mol high-sim + medium-sim) | ✅ done — Tanimoto 0.25–0.89 to fasudil |
| Stage 5 v2 scoring-chain ablation (10-receptor) | ✅ done — ρ(SC1, SC4) = 0.917, top-5 overlap 2/5 |
| Stage 5 v2 ablation with **single PCA c09 receptor** | ✅ done — **ρ(SC1, SC4_c09) = 0.746, top-5 overlap 1/5** (much sharper signal) |
| Pose-PDBQT-to-RDKit verified via Meeko | ✅ |
| Test suite | 15/15 passing |

### Headline finding for the paper

> **PCA cluster_09** is the single conformational state of α-syn that supports fasudil binding (0% bound in all other 9 clusters, 6.12% bound in C9). Using c09 alone as the WE-aware docking receptor gives a **dramatically sharper "WE matters" signal** than the 10-contact-map ensemble (ρ vs naive 1XQ8: 0.746 vs 0.917; top-5 overlap 1/5 vs 2/5). The PLIF Tanimoto dynamic range is 2× wider (max 0.59 vs 0.38). For Task #24, use c09 as the reward receptor, not the 10-ensemble.

### Notable novel-scaffold hit from c09 ablation

**cand 4** (`CC(CC1CCN(C(=O)c2csc3ccccc23)CC1)N1CCNCC1`) — benzothiophene-amide + piperidine + piperazine. PLIF_c09 = 0.484 (highest in top-5), Vina = −5.88. Tanimoto to fasudil only 0.24 — a **genuine scaffold-hop hit**, not a fasudil analogue. Worth deeper investigation (Task #28).

---

## 0. Current State (snapshot)

| Stage | Status |
|---|---|
| System build (108k atoms, a99SB-disp + TIP4P-D, fasudil at 14 Å from Y133/D135/Y136) | done |
| Equilibration (min → NVT → NPT, density 0.83 → ~1.0) | done |
| WESTPA setup (pmemd.cuda + MPS, 48 walkers, 2D pcoord) | done |
| **WE production run** | **iter 73 / 500, healthy** |
| Disordering check (DSSP on iter 73 sample) | helix-1: 100% → 57.6%, helix-2: 82% → 68%, C-term: 0% (binding region was disordered from start — already valid for docking) |
| REINVENT install | not started |
| Scoring chain | not started |

WE job: `794545` on `sapphire-0`, 7-day budget. ETA to iter 500: ~3 more days at ~11 min/iter.

---

## 1. Pipeline Overview

```
WE (running) ──► receptor ensemble (10-20 frames, weighted)
                                │
                                ▼
                     ┌──────────────────────┐
                     │  REINVENT generator  │ ← seeded on fasudil (Mol2Mol)
                     └──────────────────────┘
                                │  proposes 500 SMILES/round
                                ▼
                ┌────────── SCORING CHAIN ──────────┐
                │  1. ECFP similarity to fasudil    │  cheap, ms
                │  2. Pharmacophore + USRCAT match  │
                │  3. Drug-likeness (QED, Lipinski) │
                │  4. Ensemble docking (Smina/GNINA)│  ~10s/mol
                │  5. PLIF Tanimoto vs fasudil ref  │
                │  6. Pose-persistence MD (top 10%) │  expensive
                └────────────────────────────────────┘
                                │
                                ▼  reward → RL update
                       (repeat 50 rounds)
                                │
                                ▼
                       top-K ranked candidates
                                │
                                ▼  validation
                       mini-WE on top 5 → k_on, k_off, ΔG
                                │
                                ▼
                       wet-lab handoff
```

**Key insight:** WE provides the *map* (receptor ensemble + weights + PLIF target + kinetics baseline) that makes the scoring meaningful for an IDP. REINVENT explores chemistry space using that map.

---

## 2. What To Do Next — Ordered Action List

### Track A — Things that don't need WE to finish (start today)

**A1. Install REINVENT 4 + smoke test** (Task #13)
```bash
# fresh conda env on hpc2
conda create -n reinvent4 python=3.10 -y
conda activate reinvent4
pip install reinvent

# Download priors
reinvent --download-priors

# Smoke test
reinvent run -c examples/sampling/mol2mol_demo.toml
```
Goal: confirm 100-mol generation works on a CPU. ~30 min total.

**A2. Pharmacophore + Lipinski/QED reward** (Task #14)
Pure RDKit, no WE data needed. Reward = +1 if molecule has:
- aromatic ring (isoquinoline-like preferred)
- tertiary basic amine
- 4–7 Å spacer between them
- MW ≤ 500, logP ≤ 5, HBD ≤ 5, HBA ≤ 10
Validate on fasudil itself — should give max reward.

**A3. ECFP + USRCAT fingerprint terms** (Task #20)
ECFP4 Tanimoto similarity to fasudil with a window reward (peak at sim ≈ 0.6 — close enough but not identical; encourages exploration). USRCAT for 3D shape match.

**A4. Set up Smina/GNINA infrastructure** (Task #15)
- Install Smina (CPU) or GNINA (GPU, includes CNN rescoring)
- Write `dock_ensemble.py` wrapper: input = SMILES + N receptor PDBs + box → output = weighted-median score
- Test on fasudil docked into bstate (placeholder receptor) — should give a score in –5 to –8 kcal/mol range

### Track A* — Methodological control (do this FIRST, before any generation)

**A0. Baseline docking control: 1XQ8 vs WE-receptor** (Task #22) — *the figure that justifies WE in the paper*

Three docking conditions, fasudil as ligand, AutoDock Vina or Smina:

| Cond | Receptor | Search box | What it tests |
|---|---|---|---|
| **A_biased** | PDB 1XQ8 (raw, model 1) | Centered on Y133/D135/Y136 region | Does 1XQ8 dock fasudil correctly when we *tell* it the binding site? |
| **A_blind** | PDB 1XQ8 | Whole-protein box | Where does an unbiased docker put fasudil on 1XQ8? |
| **C_biased** | WE-derived bound-pose receptor (fasudil stripped) | Centered on same C-term region | Does docking against a WE-derived receptor reproduce the canonical pose? |

For each: top-20 poses, exhaustiveness 16. Output per condition:
- All pose PDBs (so we can visualize side-by-side)
- Hit rate to Y133/D135/Y136 site (fraction of poses with min-distance < 5 Å to those residues)
- Score distribution
- RMSD of top pose vs WE bound pose

Expected outcome (preregistered):
- A_biased: docker still binds Y133/D135 if forced — but score will be sub-optimal because the C-term is in the wrong conformation
- A_blind: docker prefers helix-1/helix-2 crevices, ignores C-term
- C_biased: recovers Y133/Y136 π-stack + D135 charge contact, score ≲ –6 kcal/mol

Files: `/home/anugraha/IDP/docking/` — receptors, poses, summary table.

### Track B — Needs WE data (start when WE hits iter ~150–200, ~1 day from now)

**B1. Extract receptor ensemble from WE** (Task #12)
- Read west.h5 weights + contact_map aux data
- Filter to "bound" frames: pcoord[0] < 5 Å OR any ring_dist < 6 Å OR d135_charge_dist < 5 Å
- Cluster into 10–20 representatives (k-medoids on contact_map vectors)
- Write each centroid frame to receptor PDB
- Define the binding-site box from the centroid of fasudil position across the ensemble

**B2. Compute fasudil's reference PLIF** (Task #19)
- Run ProLIF on the same bound frames (~hundreds of frames)
- Output: per-residue interaction frequencies (the "interaction signature" for fasudil)
- Save as reference vector for PLIF Tanimoto scoring

**B3. PLIF-similarity reward** (Task #19)
- Plug A4's docking + ProLIF into the scoring chain: for each candidate, dock → top pose → PLIF → Tanimoto vs reference

### Track C — Validation + final campaign

**C1. Disordering recheck at iter 200 and iter 500** (Task #8a)
Re-run the DSSP analysis. Expect helix-1 < 20% by iter 500.

**C2. Contact-map comparison vs Robustelli 2022 / Baidya 2024** (Task #8b)
Compare WE-derived per-residue contact frequencies with published NMR CSPs around C-terminus.

**C3. Kinetics from WE — haMSM analysis** (Task #18)
- Use `msm_we` or `w_kinetics` on west.h5 + traj_segs
- Get: k_on, k_off, MFPT, fasudil residence time τ
- This is the kinetic baseline every REINVENT candidate must match/beat

**C4. First REINVENT campaign — Mol2Mol on fasudil** (Task #16)
- 500 mols × 50 rounds
- Scoring chain: A2 + A3 + A4 + B3
- Output: top 50 ranked candidates with predicted ΔG + PLIF Tanimoto

**C5. Pose-persistence proxy for top 10%** (Task #21)
- For top 10% per round, run 1 ns pmemd.cuda MD from docking pose
- Score by ligand RMSD after 1 ns (low = pose stable = predicted long τ)

**C5.5. Scoring-chain ablation** (Task #23) — *cheap sanity check before committing to the full comparison*

Generate ~100 candidates with minimal scoring. Score them with 4 different chains and look at rank correlation. If WE-derived terms change the top-10, WE pulls its weight in REINVENT. If not, simplify.

**C5.6. Two-campaign comparison: WE-free vs WE-aware REINVENT** (Task #24) — *the paper-figure experiment for the REINVENT case*

Run two identical campaigns differing only in whether the scoring chain uses WE-derived signal. Compare top-50 from each on three orthogonal hold-out metrics (literature-binder similarity, PLIF on held-out WE frames, mini-WE ΔG on top-5). This is what distinguishes "WE → better docking" (Task #22, done) from "WE → better generation."

**C6. Mini-WE validation on top 5 candidates** (Task #17)
- Parameterize each with GAFF2/AM1-BCC
- Run a short WE (~150 iters, 1 day each on a single GPU)
- Compute k_on, k_off, ΔG vs fasudil baseline
- Final ranked list → wet-lab handoff

---

## 3. Decisions Still Open

| Decision | Options | Current lean |
|---|---|---|
| Docker | Smina (CPU) / GNINA (GPU+CNN) / Vina | GNINA if GPU headroom, else Smina |
| Generation mode | Mol2Mol / LibInvent / de novo | Mol2Mol seeded on fasudil (first campaign) |
| Reward weights | TBD by experimentation | start equal; tune after first campaign |
| When to extract receptor ensemble | now (iter 73) / iter 200 / iter 500 | iter 200 (enough disordering, enough sampling) |
| Run a 2nd WE for unbinding rate? | yes / no — currently equilibrium WE | decide after iter 500 based on flux statistics |

---

## 4. Files & Paths (where everything lives)

| | |
|---|---|
| WE root | `/home/anugraha/IDP/westpa/` |
| WE data | `west.h5`, `traj_segs/000NNN/000NNN/seg.nc` |
| Topology | `/home/anugraha/IDP/build/complex.prmtop` |
| Fasudil ligand | `/home/anugraha/IDP/build/fasudil.mol2` (GAFF2 + AM1-BCC) |
| pcoord script | `westpa/westpa_scripts/calc_pcoord.py` |
| env | `westpa/env.sh` (sources amber/24 + openmm_env conda) |
| DSSP check | `westpa/check_disorder.cpptraj` |
| **REINVENT work** | `/home/anugraha/IDP/reinvent/` (to be created) |
| **Scoring code** | `/home/anugraha/IDP/scoring/` (to be created) |
| **Receptor ensemble** | `/home/anugraha/IDP/receptors/` (to be created) |

---

## 5. What "Success" Looks Like

Three concrete deliverables for the sprint:

1. **A WE-derived characterization of α-syn–fasudil binding** — receptor ensemble, contact pattern, residence time, k_on/k_off — that reproduces or extends Robustelli 2022 / Baidya 2024.
2. **A REINVENT-generated ranked list of 50 novel α-syn binder candidates** with predicted ΔG, PLIF similarity to fasudil, and pose-persistence scores.
3. **Mini-WE validation of top 5** with real k_on, k_off, ΔG — handoff-ready for medchem / synthesis.

---

## 6. Daily Check Routine (while WE runs)

```bash
cd /home/anugraha/IDP/westpa
squeue -u anugraha                                   # job alive?
ls traj_segs | tail -3                               # latest iter
grep -iE "error|exception" logs/west-794545-master.log | tail -10   # any errors?
```

If WE stuck: **do not re-init.** `scancel <jobid>; sbatch run_WE.sbatch` (resumes from west.h5). See `memory/feedback_we_restart.md`.

---

## 7. Today's Action

**Right now:** kick off Track A (REINVENT install + pharmacophore reward + docking infra). All three can happen in parallel without touching the running WE.

WE finishes Track B prerequisites in ~1 day. By the time the receptor ensemble is ready, the scoring chain will be sitting there waiting for it.
