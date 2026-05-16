# PyMOL visualization for 1XQ8-vs-WE docking control
# Run:  pymol view_in_pymol.pml

bg_color white

# Receptors
load receptors/A_1xq8.pdb, A_1xq8
load receptors/C_we_bound.pdb, C_we_bound

# A_biased poses (top 5)
load poses/A_biased_pose1.pdb, A_biased_top1
load poses/A_biased_pose2.pdb, A_biased_top2
load poses/A_biased_pose3.pdb, A_biased_top3
load poses/A_biased_pose4.pdb, A_biased_top4
load poses/A_biased_pose5.pdb, A_biased_top5

# A_blind poses (top 5)
load poses/A_blind_pose1.pdb, A_blind_top1
load poses/A_blind_pose2.pdb, A_blind_top2
load poses/A_blind_pose3.pdb, A_blind_top3
load poses/A_blind_pose4.pdb, A_blind_top4
load poses/A_blind_pose5.pdb, A_blind_top5

# C_biased poses (top 5)
load poses/C_biased_pose1.pdb, C_biased_top1
load poses/C_biased_pose2.pdb, C_biased_top2
load poses/C_biased_pose3.pdb, C_biased_top3
load poses/C_biased_pose4.pdb, C_biased_top4
load poses/C_biased_pose5.pdb, C_biased_top5

# Style
hide everything
show cartoon, A_1xq8 or C_we_bound
color cyan, A_1xq8
color green, C_we_bound
show sticks, A_biased_top* or A_blind_top* or C_biased_top*
color red, A_biased_top*
color orange, A_blind_top*
color magenta, C_biased_top*

# Highlight the target binding triad
show sticks, (A_1xq8 or C_we_bound) and resi 133+135+136 and not (name N+C+O+CA)
color yellow, (A_1xq8 or C_we_bound) and resi 133+135+136

zoom (A_biased_top* or C_biased_top*) extend 8
