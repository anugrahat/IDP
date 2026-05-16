#!/bin/bash
#SBATCH --job-name=asyn_collapse
#SBATCH --account=ahnlab
#SBATCH --partition=gpu-ahn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/anugraha/IDP/logs/asyn_collapse-%j.out
#SBATCH --error=/home/anugraha/IDP/logs/asyn_collapse-%j.err

set -euo pipefail
echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
nvidia-smi -L

module load amber/24

cd /home/anugraha/IDP/minimize

# 1) Quick implicit-solvent minimization
pmemd.cuda -O \
  -i  asyn_min_implicit.in \
  -p  /home/anugraha/IDP/build/asyn_vacuum.prmtop \
  -c  /home/anugraha/IDP/build/asyn_vacuum.inpcrd \
  -ref /home/anugraha/IDP/build/asyn_vacuum.inpcrd \
  -o  asyn_min_implicit.out \
  -r  asyn_min_implicit.rst7 \
  -inf asyn_min_implicit.mdinfo

# 2) 10 ns GB-Neck2 collapse at 350 K
pmemd.cuda -O \
  -i  asyn_collapse.in \
  -p  /home/anugraha/IDP/build/asyn_vacuum.prmtop \
  -c  asyn_min_implicit.rst7 \
  -o  asyn_collapse.out \
  -r  asyn_collapse.rst7 \
  -x  asyn_collapse.nc \
  -inf asyn_collapse.mdinfo

# 3) Extract a representative compact frame (smallest Rg in last 5 ns)
module load amber/24
cat > extract_compact.cpptraj <<'EOF'
parm /home/anugraha/IDP/build/asyn_vacuum.prmtop
trajin asyn_collapse.nc 250 1000 1
radgyr @CA out rg_ca.dat mass
go
EOF
cpptraj -i extract_compact.cpptraj > extract_compact.log 2>&1

# Identify min-Rg frame index within the last-5-ns window
python3 - <<'PY'
import numpy as np
data = np.loadtxt("rg_ca.dat", comments="#")
idx = int(np.argmin(data[:,1]))
real_frame = 250 + idx   # offset within full traj
print(f"min Rg = {data[idx,1]:.2f} Å at trajectory frame {real_frame}")
open("min_rg_frame.txt","w").write(str(real_frame))
PY

FRAME=$(cat min_rg_frame.txt)
cat > write_compact_pdb.cpptraj <<EOF
parm /home/anugraha/IDP/build/asyn_vacuum.prmtop
trajin asyn_collapse.nc ${FRAME} ${FRAME} 1
center :1-140 mass origin
trajout /home/anugraha/IDP/build/asyn_compact.pdb pdb
go
EOF
cpptraj -i write_compact_pdb.cpptraj > write_compact_pdb.log 2>&1

echo "Compact α-syn frame written: /home/anugraha/IDP/build/asyn_compact.pdb"
echo "Done at $(date)"
