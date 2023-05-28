#!/bin/bash -l
# Job name
#SBATCH --job-name=sub-hemispace1002_python extract_task_blocks sub-hemispace1001
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vayzenb@cmu.edu
# Submit job to cpu queue                
#SBATCH -p cpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
# Job memory request
#SBATCH --mem=48gb
# Time limit days-hrs:min:sec
#SBATCH --time 3-00:00:00
# Exclude
# SBATCH --exclude=mind-1-26,mind-1-30
# Standard output and error log
#SBATCH --output=/user_data/vayzenbe/GitHub_Repos/hemispace/slurm_out/sub-hemispace1002_python extract_task_blocks sub-hemispace1001.out

conda activate fmri
module load fsl-6.0.3  
python python extract_task_blocks sub-hemispace1001 sub-hemispace1002
