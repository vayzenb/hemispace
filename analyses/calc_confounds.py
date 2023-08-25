'''
Computes confound metrics (tsnr, motion) for each subject and saves to csv
'''


import sys
import os
curr_dir = '/user_data/vayzenbe/GitHub_Repos/hemispace'
sys.path.insert(0, curr_dir)
import pandas as pd
import numpy as np
from nilearn import image, plotting, datasets, maskers, glm
#from nilearn.glm import threshold_stats_img
import numpy as np
from nilearn.maskers import NiftiMasker
import nibabel as nib
import hemispace_params as params
import pdb
#hide warnings
import warnings
warnings.filterwarnings("ignore")

data_dir = params.data_dir
results_dir = params.results_dir

sub_info = params.sub_info
task_info = params.task_info

suf = params.suf
thresh = params.thresh
rois = params.rois

#extract only unique tasks
task_info = task_info.drop_duplicates(subset = ['task'])



def extract_tsnr(sub, hemi, task, run):
    sub_dir = f'{data_dir}/{sub}/ses-01'

    anat_mask = image.load_img(f'{data_dir}/{sub}/ses-01/anat/{sub}_ses-01_T1w_brain_mask_{hemi}.nii.gz')
    run_dir = f'{sub_dir}/derivatives/fsl/{task}/run-0{run}/1stLevel.feat'
    
    '''
    Calc tsnr metrics
    '''
    # load filtered func
    filtered_func = f'{run_dir}/filtered_func_data_reg.nii.gz'
    filtered_data = image.load_img(filtered_func)

    #extract data from hemi mask
    masker = NiftiMasker(mask_img=anat_mask)
    masked_data = masker.fit_transform(filtered_data)

    
    del filtered_data
    del masker

    #calculate tsnr
    tsnr = np.mean(masked_data, axis=0) / np.std(masked_data, axis=0)

    return np.nanmean(tsnr)


def extract_roi_tsnr(sub,hemi,roi,task,run):

    sub_dir = f'{data_dir}/{sub}/ses-01'

    anat_mask = image.load_img(f'{data_dir}/{sub}/ses-01/anat/{sub}_ses-01_T1w_brain_mask_{hemi}.nii.gz')
    run_dir = f'{sub_dir}/derivatives/fsl/{task}/run-0{run}/1stLevel.feat'

    roi_img = image.load_img(f'{sub_dir}/derivatives/rois/parcels/{roi}.nii.gz')
    #binarize the mask and roi
    roi_img = image.binarize_img(roi_img, threshold = 0.5, mask_img=anat_mask)
    
    '''
    Calc tsnr metrics
    '''
    # load filtered func
    filtered_func = f'{run_dir}/filtered_func_data_reg.nii.gz'
    filtered_data = image.load_img(filtered_func)

    #extract data from hemi mask
    masker = NiftiMasker(mask_img=roi_img)
    masked_data = masker.fit_transform(filtered_data)

    
    del filtered_data
    del masker

    #calculate tsnr
    tsnr = np.mean(masked_data, axis=0) / np.std(masked_data, axis=0)

    return np.nanmean(tsnr)

def calc_full_confounds():
    summary_df = pd.DataFrame(columns = ['sub', 'group', 'hemi', 'tsnr','rot','trans'])

    for sub,group, hemi in zip(sub_info['sub'],sub_info['group'], sub_info['intact_hemi']):
        
        sub_dir = f'{data_dir}/{sub}/ses-01'
        
        
        if hemi == 'both':
            hemis = ['left','right']
        else:
            hemis = [hemi]
        
        for hemi in hemis:
            anat_mask = image.load_img(f'{data_dir}/{sub}/ses-01/anat/{sub}_ses-01_T1w_brain_mask_{hemi}.nii.gz')
            all_tsnr = []
            all_rot = []
            all_trans = []
            for task in task_info['task']:

                for run in params.runs:
                    run_dir = f'{sub_dir}/derivatives/fsl/{task}/run-0{run}/1stLevel.feat'

                    #check if run exists
                    if os.path.exists(run_dir + '/filtered_func_data_reg.nii.gz'):
                        print(f'Calculating confounds for {sub} {hemi} {task} run {run}')


                        #calculate tsnr
                        tsnr = extract_tsnr(sub, hemi, task, run)
                        
                        #append to list
                        all_tsnr.append(tsnr)

                        '''
                        Calc motion metrics
                        '''

                        #load motion data
                        
                        motion_data = pd.read_csv(f'{run_dir}/mc/prefiltered_func_data_mcf.par', sep = '  ', header = None, names =['rot_x','rot_y','rot_z','trans_x','trans_y','trans_z'])

                        #take absolute value of all columns
                        motion_data = motion_data.abs()

                        #calculate mean of all columns
                        rot = motion_data[['rot_x','rot_y','rot_z']].mean(axis=0)
                        trans = motion_data[['trans_x','trans_y','trans_z']].mean(axis =0)

                        #append to list
                        all_rot.append(np.nanmean(rot))
                        all_trans.append(np.nanmean(trans))

            #append means to summary df
            summary_df = summary_df.append({'sub': sub, 'group': group, 'hemi': hemi, 'tsnr': np.nanmean(all_tsnr), 'rot': np.nanmean(all_rot), 'trans': np.nanmean(all_trans)}, ignore_index = True)

            #save summary df
            summary_df.to_csv(f'{results_dir}/confound/confound_summary.csv', index = False)

def calc_roi_confounds():
    summary_df = pd.DataFrame(columns = ['sub', 'group', 'hemi', 'tsnr','rot','trans'])

    for sub,group, hemi in zip(sub_info['sub'],sub_info['group'], sub_info['intact_hemi']):
        
        sub_dir = f'{data_dir}/{sub}/ses-01'
        
        
        if hemi == 'both':
            hemis = ['left','right']
        else:
            hemis = [hemi]
        
        for hemi in hemis:
            

            for task in task_info['task']:

                if task == 'loc':
                    roi = 'ventral_visual_cortex'
                else:
                    roi = 'dorsal_visual_cortex'

                all_tsnr = []
                all_rot = []
                all_trans = []
                for run in params.runs:
                    run_dir = f'{sub_dir}/derivatives/fsl/{task}/run-0{run}/1stLevel.feat'

                    #check if run exists
                    if os.path.exists(run_dir + '/filtered_func_data_reg.nii.gz'):
                        print(f'Calculating confounds for {sub} {hemi} {roi} {task} run {run}')


                        #calculate tsnr
                        tsnr = extract_roi_tsnr(sub, hemi, roi, task, run)
                        
                        #append to list
                        all_tsnr.append(tsnr)

                        '''
                        Calc motion metrics
                        '''

                        #load motion data
                        
                        motion_data = pd.read_csv(f'{run_dir}/mc/prefiltered_func_data_mcf.par', sep = '  ', header = None, names =['rot_x','rot_y','rot_z','trans_x','trans_y','trans_z'])

                        #take absolute value of all columns
                        motion_data = motion_data.abs()

                        #calculate mean of all columns
                        rot = motion_data[['rot_x','rot_y','rot_z']].mean(axis=0)
                        trans = motion_data[['trans_x','trans_y','trans_z']].mean(axis =0)

                        #append to list
                        all_rot.append(np.nanmean(rot))
                        all_trans.append(np.nanmean(trans))

                print('Saving ', sub, hemi, roi, task)
                #append means to summary df
                summary_df = summary_df.append({'sub': sub, 'task': task, 'group': group, 'hemi': hemi, 'roi':roi, 'tsnr': np.nanmean(all_tsnr), 'rot': np.nanmean(all_rot), 'trans': np.nanmean(all_trans)}, ignore_index = True)

                #save summary df
                summary_df.to_csv(f'{results_dir}/confound/confound_summary_roi.csv', index = False)



calc_roi_confounds()







        




