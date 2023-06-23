curr_dir = f'/user_data/vayzenbe/GitHub_Repos/hemispace' #CHANGE AS NEEEDED CAUSE ITS FOR VLAAAD

import sys
sys.path.insert(0,curr_dir)


import numpy as np
import pandas as pd

import itertools

import pdb
import os
import hemispace_params as params

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

iter = 10
#number of subs to pull on each resample
n_subs = 4

#point to split posterior and anterior
split = 27

#extract control subs
control_subs = sub_info[sub_info['group']=='control']

#loop through conds
for task,cond, cope in zip(task_info['task'], task_info['cond'],task_info['cope']):
    all_maps = []
    for sub in control_subs['sub']:
        sub_dir = f'{data_dir}/{sub}/ses-01'

        #load data
        all_maps.append(np.load(f'{sub_dir}/derivatives/neural_map/{cond}_binary_3d.npy'))

    
    #convert to numpy
    all_maps = np.array(all_maps)


    for ii in iter:
        #randomly select n_subs
        test_subs = control_subs.sample(n=n_subs,replace=True)

        #remove test_subs from control_subs
        remaining_subs = control_subs[~control_subs['sub'].isin(test_subs['sub'])]

        #get test_maps
        test_maps = all_maps[test_subs.index]

        #get remaining_maps
        remaining_maps = all_maps[remaining_subs.index]

        #sum test maps
        test_maps_sum = np.sum(test_maps,axis=0)

        #calc mean of remaining_maps
        remaining_maps_mean = np.mean(remaining_maps,axis=0)

        #calc diff
        diff = test_maps_mean - remaining_maps_mean

        #calc peak
        peak = np.where(diff==np.max(diff))

        #add to df
        peak_coords.loc[ii,'sub'] = test_subs.iloc[0]['sub']
        peak_coords.loc[ii,'group'] = test_subs.iloc[0]['group']
        peak_coords.loc[ii,'cond'] = cond
        peak_coords.loc[ii,'hemi'] = hemis[peak[0][0]]
        peak_coords.loc[ii,'position'] = 'posterior' if peak[1][0] < split else 'anterior'
        peak_coords.loc[ii,'x'] = peak[1][0]
        peak_coords.loc[ii,'y'] = peak[2][0]









#load peak coords
peak_coords = pd.read_csv(f'{results_dir}/neural_map/peak_coords.csv')



#create empty df to store resampled control data
resample_data = pd.DataFrame()

#resample control data
for ii in range(0,iter):

    for cond in task_info['cond']:


        #Determine number of positions and preferred hemi
        if cond == 'word' or cond == 'tool':
            positions = ['all']
            pref_hemi = 'left'

        elif cond == 'face' or cond == 'space':
            positions = ['posterior','anterior']
            pref_hemi = 'right'
        
        
        for position in positions:
            #get control data for cond and position and preferred hemi
            curr_control_data = control_data[(control_data['cond']==cond) & (control_data['position']==position) & (control_data['hemi']==pref_hemi)]

            #resample control data
            test_control_data = curr_control_data.sample(n=n_subs,replace=False)

            #remove samples subs from curr_control_data
            remaining_control_data = curr_control_data[~curr_control_data['sub'].isin(test_control_data['sub'])]


            #calculuate the distane between each test data and each point in remaining_control_data
            sample_dists = []
            for i in range(len(test_control_data)):
                dists = []
                for j in range(len(remaining_control_data)):
                    dists.append(np.sqrt((test_control_data.iloc[i]['x']-remaining_control_data.iloc[j]['x'])**2 + (test_control_data.iloc[i]['y']-remaining_control_data.iloc[j]['y'])**2))
                
                #append min dist to sample_dists
                sample_dists.append(np.mean(dists))
        
            #calc mean of dists
            mean_dist = np.mean(sample_dists)

            #add mean_dist to resample_data
            resample_data.loc[ii,f'{cond}_{position}'] = mean_dist
            

#save resample_data
resample_data.to_csv(f'{results_dir}/neural_map/resample_data.csv',index=False)




