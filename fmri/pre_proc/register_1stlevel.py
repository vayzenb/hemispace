"""
Register each 1stlevel to anat in a parallelized manner

"""
curr_dir = f'/user_data/vayzenbe/GitHub_Repos/hemispace'
import numpy as np
import pandas as pd
import subprocess
import os
import pdb

import sys
sys.path.append(curr_dir)

import hemispace_params as params

sub = sys.argv[1]

data_dir = params.data_dir
results_dir = params.results_dir

sub_info = params.sub_info
task_info = params.task_info

suf = params.suf
thresh = params.thresh
rois = params.rois

runs = params.runs
firstlevel_suf = ''

sub_dir = f'{data_dir}/{sub}/ses-01'

anat = f'{sub_dir}/anat/{sub}_ses-01_T1w_brain.nii.gz'

for task in task_info['task']:
    for run in runs:
        run_dir = f'{sub_dir}/derivatives/fsl/{task}/run-0{run}/1stLevel{firstlevel_suf}.feat'
        filtered_func = f'{run_dir}/filtered_func_data.nii.gz'
        out_func = f'{run_dir}/filtered_func_data_reg.nii.gz'

        #check if run exists
        if os.path.exists(filtered_func):


            bash_cmd = f'flirt -in {filtered_func} -ref {anat} -out {out_func} -applyxfm -init {run_dir}/reg/example_func2standard.mat -interp trilinear'
            print(bash_cmd)
            subprocess.run(bash_cmd.split(), check=True)

        else:
            print(f'run {run} for task {task} does not exist for subject {sub}')
