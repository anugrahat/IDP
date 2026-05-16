# Pixi setup for this project

Two environments managed by [pixi](https://pixi.sh), pinned to versions that work as of 2026-05-16.

## Quick start

```bash
# Install pixi (one time, if not present)
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc

# Bootstrap both environments + write pixi.lock
cd /home/anugraha/IDP
pixi install

# Run a pinned tool
pixi run -e westpa pytest docking/test_ablation.py -v
pixi run -e reinvent reinvent --version
```

## Two environments, why

| Env | Python | Purpose | Why a separate env |
|---|---|---|---|
| `westpa` | 3.9 | WESTPA propagation, MD analysis, docking, PLIF | WESTPA 2022.11 + parmed 4.3 don't run on py 3.11; Dimorphite-DL pinned to v1.2.4 (last py-3.9-compatible release) |
| `reinvent` | 3.11 | REINVENT 4 generative ML, modern Dimorphite-DL (v2), modern RDKit | REINVENT 4 requires py 3.10+; Dimorphite-DL v2.x requires py 3.10+ |

Both envs share the docking + ProLIF stack at compatible versions.

## Convenience tasks

```bash
pixi run test                  # run pytest in westpa env
pixi run control               # 1XQ8 vs WE control experiment
pixi run ablation              # full scoring-chain ablation
pixi run ctrl100               # 100-seed replication of the control
pixi run prep-receptors        # extract WE receptor ensemble
pixi run ref-plif              # compute fasudil reference PLIF
pixi run mol2mol               # REINVENT Mol2Mol on fasudil
pixi run ligand-prep-v2        # Dimorphite-DL ligand prep (reinvent env)
```

## Version pins (locked in pixi.lock)

| Tool | Version | Why pinned |
|---|---|---|
| WESTPA | 2022.11 | matches our `run_WE.sbatch` configs and west.cfg syntax |
| parmed | 4.3.x | required by WESTPA 2022.11 |
| MDAnalysis | ≥2.2 (westpa), ≥2.10 (reinvent) | older for parmed compat; newer for Meeko ≥0.7 compat |
| Meeko | 0.7.1 | known-good with Vina 1.2.7 + RDKit |
| Vina | 1.2.7 | latest stable Python binding |
| ProLIF | 2.0.3 (py3.9) / 2.1.0 (py3.11) | 2.0.x is the last release w/ py3.9 support |
| Dimorphite-DL | 1.2.4 (py3.9) / 2.0.2 (py3.11) | py3.10+ required for v2 |
| REINVENT | git@main | no PyPI release; pin via lockfile commit |
| RDKit | latest in westpa, 2025.9.6 in reinvent | constrained by Dimorphite-DL 2.x |
| openbabel-wheel | 3.1.1.23 | works around hpc2's missing libxml2 |

## NOT in pixi (use system module or external)

- Amber 24 / pmemd.cuda → `module load amber/24` (system module, CUDA 11.8 wired in)
- CUDA toolkit → comes with `amber/24` module
- SLURM → system
- Vina-GPU 2.1 → external clone (excluded from git), compile against system CUDA when needed
