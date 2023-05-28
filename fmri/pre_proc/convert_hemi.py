"""
Copies data from Michael's directory into approriate hemispace directory 

converts timing files into FSL compatible covs
"""
from email import header
import shutil
import os
import glob
import numpy as np
import pandas as pd
import pdb

hemi_subj = [90, 91,92]
subj_list = [1001, 1002, 1003]
runs = list(range(1,4))
study = 'hemispace'

hemi_dir = '/lab_data/behrmannlab/hemi/Raw'
study_dir = f'/lab_data/behrmannlab/vlad/{study}'

"""
copy from michael's

"""
def copy_nii(src_dir, dest_dir, sub,run):
    src_nii = f'{src_dir}/sub-0{sub[1]}_ses-01_task-loc_run-0{run}_bold.nii.gz'
    dest_nii = f'{dest_dir}/sub-{study}{subj_list[sub[0]]}_ses-01_task-loc_run-0{run}_bold.nii.gz'
    shutil.copy(src_nii, dest_nii)

def create_cov(src_dir, dest_dir, sub,run):
    cov_file = f'{src_dir}/sub-0{sub[1]}_ses-01_task-loc_run-0{run}_events.tsv'

    cov = pd.read_csv(cov_file, sep='\t')

    conds = cov.block_type.unique().tolist()

    for cond in conds:
        dest_cov = f'{dest_dir}/covs/catloc_{study}{subj_list[sub[0]]}_run-0{run}_{cond}.txt'
        curr_cov = cov[cov['block_type'] == cond].iloc[:,0:2]
        curr_cov['value'] = np.zeros((len(curr_cov))) + 1

        curr_cov.to_csv(dest_cov, index= False, header =False, sep = '\t')


for sub in enumerate(hemi_subj):
    for run in runs:
        src_dir = f'{hemi_dir}/sub-0{sub[1]}/ses-01/func'
        dest_dir =f'{study_dir}/sub-{study}{subj_list[sub[0]]}/ses-01' 

        #copy the nifti file from hemi to hemispace
        copy_nii(src_dir, dest_dir,sub, run)


        #convert timing file to 3-col cov for FSL
        create_cov(src_dir, dest_dir, sub,run)

        #pdb.set_trace()

        

