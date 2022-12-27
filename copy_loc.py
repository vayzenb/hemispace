import os
import shutil
import pandas as pd
import numpy as np
import pdb

curr_dir = '/user_data/vayzenbe/GitHub_Repos/hemispace'
source_dir = '/lab_data/behrmannlab/hemi/Raw'
target_dir = '/lab_data/behrmannlab/vlad/hemispace'
copy_locs = False

og_sub = 'hemispace2001'
conds = ['Face', 'House', 'Object', 'Scramble', 'Word']

ses = '01'

runs = [1,2,3]
#load sub data
sub_data = pd.read_csv(f'{curr_dir}/loc_subs.csv')

for sub, ses in zip(sub_data['ID'], sub_data['Session']):
    
    for run in runs:
        #make path to func
        func_file = f'{target_dir}/{sub}/ses-01/func/{sub}_{ses}_task-loc_run-0{run}_bold.nii.gz'

        #rename to ses-01
        new_func_file = f'{target_dir}/{sub}/ses-01/func/{sub}_ses-01_task-loc_run-0{run}_bold.nii.gz'
        #check if file exists
        if os.path.exists(func_file):
            os.rename(func_file, new_func_file)

        #copy catloc cov from og_sub directory
        #Face, House, Object, Scramble, word
        
        for cond in conds:
            og_cov = f'{target_dir}/sub-{og_sub}/ses-01/covs/catloc_{og_sub}_run-0{run}_{cond}.txt'
            new_cov = f'{target_dir}/{sub}/ses-01/covs/catloc_{sub[-3:]}_run-0{run}_{cond}.txt'
            #check if file exists
            if not os.path.exists(new_cov):
                shutil.copy(og_cov, new_cov)



        


if copy_locs:
    #copy folder from source to target
    for sub, ses in zip(sub_data['ID'], sub_data['Session']):
        source_path = f'{source_dir}/{sub}/{ses}'
        target_path = f'{target_dir}/{sub}/ses-01'
        #print(os.path.exists(source_path), source_path)
        #check if source dir exists
        if os.path.exists(source_path):

            #make target path
            #os.makedirs(target_path, exist_ok=True)



            #copy sub folder from source to target
            print(source_path, target_path)

            shutil.copytree(source_path, target_path,dirs_exist_ok=True)

            #rename folder to ses-01
            #os.rename(f'{target_dir}/{sub}/{ses}', f'{target_dir}/{sub}/ses-01')
