"""
Calculate mean activation, cortical volume, summed selectivity, parcel size for each sub 
"""


curr_dir = f'/user_data/vayzenbe/GitHub_Repos/hemispace'
import sys
sys.path.append(curr_dir)

import numpy as np
import pandas as pd
import itertools
from nilearn import image, plotting, datasets, masking
import nibabel as nib
import pdb
import os
import hemispace_params as params

data_dir = params.data_dir
results_dir = params.results_dir

sub_info = params.sub_info
task_info = params.task_info

suf = params.suf
thresh = params.thresh
rois = params.rois


def calc_summary_vals(sub, task, cope, roi,hemi):
    """
    Calculate mean activation for a given sub and roi
    """
    
    #load the functional data
    func = image.load_img(f'{data_dir}/{sub}/ses-01/derivatives/fsl/{task}/HighLevel{suf}.gfeat/cope{cope}.feat/stats/zstat1.nii.gz')
    
    #convert hemi to suffix
    if hemi == 'left':
        hemi = '_left'
    elif hemi == 'right':
        hemi = '_right'
    else:
        hemi = ''

    #load anat mask
    anat_mask = image.load_img(f'{data_dir}/{sub}/ses-01/anat/{sub}_ses-01_T1w_brain_mask{hemi}.nii.gz')


    if roi != 'hemi':
        #load the roi
        roi = image.load_img(f'{data_dir}/{sub}/ses-01/derivatives/rois/parcels/{roi}.nii.gz')
        #binarize the mask and roi
        roi = image.binarize_img(roi, threshold = 0.5, mask_img=anat_mask)
    else:
        roi = anat_mask  
    
    #calculate voxel size
    vox_size = np.prod(func.header.get_zooms())

    #threshold func image
    func = image.threshold_img(func, threshold = thresh)    

    #calc size of roi
    roi_size = np.sum(image.get_data(roi))

    #extract the activation values
    vox_resp = masking.apply_mask(func, roi)
    
    #remove all zeros
    #leave only active values
    vox_resp = vox_resp[vox_resp > 0]

    #calc mean activation
    mean_act = np.mean(vox_resp)

    #count number of voxels
    vox_count = len(vox_resp)
    #convert to mm^3
    cortex_vol = vox_count * vox_size

    #calc summed selectivity
    sum_selec = np.sum(vox_resp)

    return roi_size, mean_act,  cortex_vol, sum_selec


start_sub = 'sub-104'
#check if summary file already exists, else create it
if os.path.exists(f'{results_dir}/hemispace_summary_vals.csv'):
    summary_df = pd.read_csv(f'{results_dir}/hemispace_summary_vals.csv')
else:
    #create empty summary file
    summary_df = pd.DataFrame(columns = ['sub', 'group', 'hemi', 'roi','cond', 'roi_size', 'mean_act', 'volume',  'sum_selec'])

#start sub_info from the line with sub-103
#find index for sub-103 in sub-info
sub_idx = sub_info[sub_info['sub'] == start_sub].index[0]

#start from sub-103
sub_info = sub_info.iloc[sub_idx:]


#for now just create new one every time    
#summary_df = pd.DataFrame(columns = ['sub', 'group', 'hemi', 'roi','cond', 'roi_size', 'mean_act', 'volume',  'sum_selec'])
for sub, group, hemi in zip(sub_info['sub'], sub_info['group'], sub_info['intact_hemi']):
    #check if sub alread has 'sub-' prefix
    if sub[:4] != 'sub-':
        sub = 'sub-' + sub

    #create hemi list
    if hemi == 'both':
        hemis = ['left', 'right']
    else:
        hemis = [hemi]

    for hemi in hemis:
        for roi in rois:
            for task, cond, cope in zip(task_info['task'], task_info['cond'], task_info['cope']):

                #check if task folder exists
                if os.path.exists(f'{data_dir}/{sub}/ses-01/derivatives/fsl/{task}/HighLevel{suf}.gfeat'):
                
                    print(f'Calculating summary values for {sub} {task} {cond} {cope} {hemi} {roi}')
                    roi_size, mean_act,  cortex_vol, sum_selec = calc_summary_vals(sub, task, cope, roi, hemi)

                    #apend to summary df
                    summary_df = summary_df.append({'sub': sub, 'group': group, 'hemi': hemi, 'roi': roi, 'cond': cond, 'roi_size': roi_size, 'mean_act': mean_act, 'volume': cortex_vol, 'sum_selec': sum_selec}, ignore_index = True)

    #save summary df
    summary_df.to_csv(f'{results_dir}/hemispace_summary_vals.csv', index = False)
