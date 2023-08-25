'''
Extracts the timeseries of activation for each block of each task
'''

import sys
curr_dir = '/user_data/vayzenbe/GitHub_Repos/hemispace'
sys.path.insert(0, curr_dir)
import pandas as pd
from nilearn import image, plotting, datasets, maskers, glm
#from nilearn.glm import threshold_stats_img
import numpy as np

from nilearn.input_data import NiftiMasker
import nibabel as nib

import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pdb
from scipy.stats import gamma
from scipy.stats import zscore
import warnings
#ignore warnings
warnings.filterwarnings("ignore")

import hemispace_params as params


sub = sys.argv[1]



data_dir = params.data_dir
results_dir = params.results_dir

sub_info = params.sub_info
task_info = pd.read_csv(f'{curr_dir}/task_info_fix.csv')

#extract only loc from task_info
task_info = task_info[task_info['task'] == 'loc']

suf = params.suf
thresh = params.thresh
rois = params.rois

runs = params.runs
firstlevel_suf = ''

sub_dir = f'{data_dir}/{sub}/ses-01'
#create mvpa dir
os.makedirs(f'{sub_dir}/derivatives/mvpa', exist_ok = True)
os.makedirs(f'{sub_dir}/derivatives/snr', exist_ok = True)



def lookup_cov_info(sub,task,cond, run):
    
    print('Creating cov...')
    #remove sub- from sub
    sub = sub.replace('sub-','')
    #get cov name from task file
    cov_name = task_info['cov_name'][(task_info['task'] == task) & (task_info['cond'] == cond)].values[0]
    vols = task_info['vols'][(task_info['task'] == task) & (task_info['cond'] == cond)].values[0]
    tr = task_info['tr'][(task_info['task'] == task) & (task_info['cond'] == cond)].values[0]

    times = np.arange(0, vols*tr, tr)
    
    if task == 'spaceloc':
        cov_file = f'{data_dir}/sub-{sub}/ses-01/covs/SpaceLoc_{sub}_Run{run}_{cov_name}.txt'
    elif task == 'toolloc':
        cov_file = f'{data_dir}/sub-{sub}/ses-01/covs/ToolLoc_{sub}_run{run}_{cov_name}.txt'
    elif task == 'loc':
        cov_file = f'{data_dir}/sub-{sub}/ses-01/covs/catloc_{sub}_run-0{run}_{cov_name}.txt'
    
    #load cov file
    cov = pd.read_csv(cov_file, sep = '\t', header = None, names = ['onset','duration', 'value'])
    #round onset and duration to nearest TR
    cov['onset'] = np.round(cov['onset']/tr)*tr
    cov['duration'] = np.round(cov['duration']/tr)*tr

    #pdb.set_trace()
    final_cov = []
    #loop through rows of cov and create psy
    for i in range(cov.shape[0]):
        #set all but current row to 0
        curr_cov = cov.copy()
        curr_cov['value'] = 0
        curr_cov['value'][i] = 1
        curr_cov = curr_cov.to_numpy()
        
        psy, name = glm.first_level.compute_regressor(curr_cov.T, None, times)
        

        #append 4 s to the start of the psy
        psy = np.append(np.zeros(int(4/tr)), psy)

        #remove last 4 s  of psy
        psy = psy[:-int(4/tr)]

        #append to final cov
        final_cov.append(psy)

    
    return final_cov



#extract hemi info
hemi = sub_info['intact_hemi'][sub_info['sub'] == sub].values[0]

if hemi == 'both':
    hemis = ['left','right']
else:
    hemis = [hemi]

for hemi in hemis:
    #load anat mask
    anat_mask = image.load_img(f'{data_dir}/{sub}/ses-01/anat/{sub}_ses-01_T1w_brain_mask_{hemi}.nii.gz')

    for roi in rois:
        
        #load roi
        if roi != 'hemi':
            #load the roi
            roi_img = image.load_img(f'{sub_dir}/derivatives/rois/parcels/{roi}.nii.gz')
            #binarize the mask and roi
            roi_img = image.binarize_img(roi_img, threshold = 0.5, mask_img=anat_mask)
        else:
            roi_img = anat_mask      
        
        for task,cond in zip(task_info['task'], task_info['cond']):
            print(f'Extracting {sub} {task} {cond} {roi} data')

            if os.path.exists(f'{sub_dir}/derivatives/fsl/{task}/HighLevel.gfeat'):
                snr_list = []
                all_data = []
                for run in runs:
                    run_dir = f'{sub_dir}/derivatives/fsl/{task}/run-0{run}/1stLevel{firstlevel_suf}.feat'
                    filtered_func = f'{run_dir}/filtered_func_data_reg.nii.gz'
                    
                    
                    cov = lookup_cov_info(sub,task,cond, run)
                    
                    #load filtered data
                    filtered_data = image.load_img(filtered_func)
                    #filtered_data = image.clean_img(filtered_data, detrend = False, standardize = True)
                    
                    print('func loaded')
            
                        
                    #extract the activation values
                    masker = maskers.NiftiMasker(mask_img=roi_img)
                    masker.fit(filtered_data)
                    roi_data = masker.transform(filtered_data)
                    #standardize the data
                    roi_data = zscore(roi_data, axis = 0)


                    del filtered_data
                    del masker

                    

                    
                    #extract the values for each cov
                    for i in range(len(cov)):
                        #convert cov to int
                        curr_cov = cov[i].astype(int)
                        #extract roi_data where cov == 1
                        curr_block = roi_data[curr_cov==1,:]

                        #average across time
                        curr_mean = np.mean(curr_block, axis = 0)

                        #sneak in tSNR analysis here
                        snr = curr_mean/np.std(curr_block, axis = 0)
                        snr_list.append(np.nanmean(snr))

                        
                        #append current block data to all data
                        all_data.append(curr_mean)

                #convert all_data to array
                all_data = np.array(all_data)
                

                #save all_data
                np.save(f'{sub_dir}/derivatives/mvpa/{hemi}_{roi}_{task}_{cond}.npy', all_data)
                np.save(f'{sub_dir}/derivatives/snr/{hemi}_{roi}_{task}_{cond}_snr.npy', np.mean(snr_list))
                


                

                    


                    

        

            