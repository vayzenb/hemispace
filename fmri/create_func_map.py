'''
Create 2D heatmap of activation for each subject and group average
'''

curr_dir = f'/user_data/vayzenbe/GitHub_Repos/hemispace' #CHANGE AS NEEEDED CAUSE ITS FOR VLAAAD

import sys
sys.path.insert(0,curr_dir)
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import itertools
from nilearn import image, plotting, datasets
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
import nibabel as nib
import pdb
import os
import hemispace_params as params
import pdb
#hide warning
import warnings
warnings.filterwarnings("ignore")

data_dir = params.data_dir
results_dir = params.results_dir
fig_dir = params.fig_dir

sub_info = params.sub_info
task_info = params.task_info
thresh = params.thresh

suf = params.suf
rois = params.rois
hemis = params.hemis


#load subject info
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')

#extract just patients from group
#sub_info = sub_info[sub_info['group']=='patient']

#load mni mask
mni = load_mni152_brain_mask()
roi_dir = '/user_data/vayzenbe/GitHub_Repos/fmri/roiParcels'



def create_sub_map():
    print('Creating individual subject maps...')

    for task,cond, cope in zip(task_info['task'], task_info['cond'],task_info['cope']):

        if cond == 'word' or task == 'face':
            roi_type = 'ventral_visual_cortex'
        elif cond == 'tool' or cond == 'space':
            roi_type = 'dorsal_visual_cortex'
        
        #load roi
        roi = image.load_img(f'{roi_dir}/{roi_type}.nii.gz')
        #binarize roi
        roi = image.math_img('img > 0', img=roi)
        for sub, code, hemi in zip(sub_info['sub'], sub_info['code'], sub_info['intact_hemi']):
            sub_dir = f'{data_dir}/{sub}/ses-01'

            #check if zstat exists
            zstat_path = f'{sub_dir}/derivatives/fsl/{task}/HighLevel{suf}.gfeat/cope{cope}.feat/stats/zstat1_reg.nii.gz'
            
            if os.path.exists(zstat_path):
                print(f'Processing {sub} {cond}, {roi_type}')

                #create output dir
                os.makedirs(f'{sub_dir}/derivatives/neural_map', exist_ok=True)

                #Load zstat
                zstat = image.load_img(zstat_path)



                #threshold zstat
                zstat = image.threshold_img(zstat, threshold=thresh, two_sided=False)

                #extract unmasked whole_brain data
                whole_brain = zstat.get_fdata()

                #mask zstat with roi
                zstat_masked = image.math_img('img1 * img2', img1=zstat, img2=roi)

                #convert zstat to numpy
                func_np = zstat_masked.get_fdata()

                #binarize func_np
                binary_3dfunc= np.copy(func_np)
                binary_3dfunc[func_np>0] = 1

                #binarize whole brain
                whole_brain[whole_brain>0] = 1



                #average across voxels in z dimension
                func_np = np.transpose(np.max(func_np, axis=2))


                #create binary version of zstat
                binary_func = np.copy(func_np)
                binary_func[binary_func>0] = 1


                #save func np
                np.save(f'{sub_dir}/derivatives/neural_map/{cond}_func.npy', func_np)

                #save binary func
                np.save(f'{sub_dir}/derivatives/neural_map/{cond}_binary.npy', binary_func)

                #save binary 3d func
                np.save(f'{sub_dir}/derivatives/neural_map/{cond}_binary_3d.npy', binary_3dfunc)

                #save whole brain
                np.save(f'{sub_dir}/derivatives/neural_map/{cond}_whole_brain.npy', whole_brain)

            else:
                print(f'{cond} zstat does not exist for subject {sub}')

def create_group_map():
    print('Creating group maps...')

    #extract control subs from sub_info
    control_subs = sub_info[sub_info['group']=='control']

    for task,cond, cope in zip(task_info['task'], task_info['cond'],task_info['cope']):
        print(f'Processing {cond} {task}')
        
        
        n = 0
        func_list = []
        binary_list = []     
        for sub in control_subs['sub']:
            sub_dir = f'{data_dir}/{sub}/ses-01'

            #check if neural map exists
            neural_map_path = f'{sub_dir}/derivatives/neural_map/{cond}_func.npy'

            if os.path.exists(neural_map_path):

                
                #load neural map
                neural_map = np.load(neural_map_path)

                #rescale all values as proportion of max to normalize across subject activation
                neural_map = neural_map/np.max(neural_map)
                
                
                
                #add neural map to list
                func_list.append(neural_map)

                #load binary map
                binary_map = np.load(f'{sub_dir}/derivatives/neural_map/{cond}_binary.npy')

                #add binary map to list
                binary_list.append(binary_map)

        #average func maps across subjects
        func_group = np.nanmean(func_list, axis=0)
        

        #sum binary map
        binary_group = np.nansum(binary_list, axis=0)

        #save func group
        np.save(f'{results_dir}/neural_map/{cond}_func.npy', func_group)

        #save binary group
        np.save(f'{results_dir}/neural_map/{cond}_binary.npy', binary_group)

def create_3d_group_map():
    '''
    Create non-parametric group map for controls
    '''
    print('Creating 3d group maps...')
    #extract control subs from sub_info
    control_subs = sub_info[sub_info['group']=='control']

    for task,cond, cope in zip(task_info['task'], task_info['cond'],task_info['cope']):
        print(f'Processing {cond} {task}')
        n = 0
        func_list = []
        binary_list = []     
        for sub in control_subs['sub']:
            
            
            sub_dir = f'{data_dir}/{sub}/ses-01'

            

            #check if neural map exists
            neural_map_path = f'{sub_dir}/derivatives/neural_map/{cond}_binary_3d.npy'

            if os.path.exists(neural_map_path):
                print(f'path exists for {sub}')
                if n == 0:
                    #load zstat reg
                    zstat_reg = image.load_img(f'{sub_dir}/derivatives/fsl/{task}/HighLevel{suf}.gfeat/cope{cope}.feat/stats/zstat1_reg.nii.gz')
                    affine = zstat_reg.affine
                    header = zstat_reg.header
                    n+=1
                


                #load binary map
                binary_map = np.load(neural_map_path)

                #add binary map to list
                binary_list.append(binary_map)

        
        #pdb.set_trace()
        #sum binary map
        binary_group = np.nansum(binary_list, axis=0)
        np.save(f'{results_dir}/neural_map/{cond}_group_map.npy', binary_group)

        #convert to nifti
        binary_group = nib.Nifti1Image(binary_group, affine, header)
        #save binary group
        nib.save(binary_group,f'{results_dir}/neural_map/{cond}_group.nii.gz')

        #save func group
        #np.save(f'{results_dir}/neural_map/{cond}_func.npy', func_group)

        #save binary group
        


#create_sub_map()
#create_group_map()

create_3d_group_map()


