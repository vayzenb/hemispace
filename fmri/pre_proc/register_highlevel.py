"""
Register each HighLevel to anat in a parallelized manner

"""
curr_dir = f'/user_data/vayzenbe/GitHub_Repos/hemispace'
import numpy as np
import pandas as pd
import subprocess
import os
import pdb
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
import sys
sys.path.append(curr_dir)

import hemispace_params as params

data_dir = params.data_dir
results_dir = params.results_dir

sub_info = params.sub_info
task_info = params.task_info

suf = ''
thresh = params.thresh
rois = params.rois

runs = params.runs



mni='/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz' #this is the MNI we use for analysis

#extract just sub-109 and sub-hemispace1004 from sub_info
sub_info = sub_info.loc[sub_info['sub'].isin(['sub-109','sub-hemispace1004'])]

for sub in sub_info['sub']:
    sub_dir = f'{data_dir}/{sub}/ses-01'
    for task, cope in zip(task_info['task'], task_info['cope']):
        print(f'Registering {sub} {task} to anat')
        #register each highlevel to anat
        zstat = f'{sub_dir}/derivatives/fsl/{task}/HighLevel{suf}.gfeat/cope{cope}.feat/stats/zstat1.nii.gz'

        out_func = f'{sub_dir}/derivatives/fsl/{task}/HighLevel{suf}.gfeat/cope{cope}.feat/stats/zstat1_reg.nii.gz'

        #check if zstat exists
        if os.path.exists(zstat):
            bash_cmd = f'flirt -in {zstat} -ref {mni} -out {out_func} -applyxfm -init {sub_dir}/anat/anat2stand.mat -interp trilinear'
            subprocess.run(bash_cmd.split(), check=True)
        else:
            print(f'zstat {zstat} does not exist for subject {sub}')