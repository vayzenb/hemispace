
import pandas as pd
import numpy as np

runs = [1,2]

curr_dir = f'/user_data/vayzenbe/GitHub_Repos/hemispace'
data_dir = f'/lab_data/behrmannlab/vlad/hemispace'
results_dir = f'{curr_dir}/results'
fig_dir = f'{curr_dir}/figures'

suf = ''

sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
task_info = pd.read_csv(f'{curr_dir}/task_info{suf}.csv')



thresh = 2.58 #threshold for zstat at p < 0.01, uncorrected

vox_size = 3

rois = ['hemi','dorsal_visual_cortex', 'ventral_visual_cortex']

hemis = ['left','right']
