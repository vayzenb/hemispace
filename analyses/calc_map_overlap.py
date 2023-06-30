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

top_perc = .1#top percentage of voxels to use for overlap calculation
#point to split posterior and anterior
split = 27

positions = ['posterior','anterior']

def calc_full_overlap():
    print('Calculating patient overlap...')

    patient_subs = sub_info[sub_info['group'] == 'patient']
    control_subs = sub_info[sub_info['group'] == 'control']


    summary_df = pd.DataFrame(columns = ['sub','code','group','cond','hemi','dice'])
    n = 0
    for task,cond, cope in zip(task_info['task'], task_info['cond'],task_info['cope']):
        print(f'Processing {cond} {task}')


        #Determine number of positions and preferred hemi
        if cond == 'word' or cond == 'tool':
            pref_hemi = 'left'

        elif cond == 'face' or cond == 'space':
            pref_hemi = 'right'
        
    
        for sub, code, group, hemi in zip(sub_info['sub'], sub_info['code'], sub_info['group'], sub_info['intact_hemi']):
            sub_dir = f'{data_dir}/{sub}/ses-01'

            #check if neural map exists
            neural_map_path = f'{sub_dir}/derivatives/neural_map/{cond}_binary.npy'

            if os.path.exists(neural_map_path):

                #load neural map
                neural_map = np.load(neural_map_path)

                #check if max is not 0
                if np.max(neural_map) != 0:

                    #check if patient and if its the preferred hemi
                    #if not flip the map
                    if hemi != 'both' and hemi != pref_hemi:
                        neural_map = np.flip(neural_map,axis=1)

                    #zero out non-preferred hemi
                    hemi_map = neural_map.copy()
                    if pref_hemi == 'left':
                        hemi_map[:,:int(hemi_map.shape[1]/2)] = 0
                        
                    elif pref_hemi == 'right':
                        hemi_map[:,int(hemi_map.shape[1]/2):] = 0
                
                    dice_list = []
                    #loop through controls and calculate dice coefficient
                    for control_sub in control_subs['sub']:
                        control_sub_dir = f'{data_dir}/{control_sub}/ses-01'
                        control_neural_map_path = f'{control_sub_dir}/derivatives/neural_map/{cond}_binary.npy'

                        if os.path.exists(control_neural_map_path) and sub != control_sub:
                            control_neural_map = np.load(control_neural_map_path)

                            #check if max is not 0
                            if np.max(control_neural_map) != 0:
                                    
                                #zero out non-preferred hemi
                                control_hemi_map = control_neural_map.copy()
                                if pref_hemi == 'left':
                                    control_hemi_map[:,:int(control_hemi_map.shape[1]/2)] = 0

                                elif pref_hemi == 'right':
                                    control_hemi_map[:,int(control_hemi_map.shape[1]/2):] = 0

                                #calculate dice coefficient
                                dice = np.sum(hemi_map*control_hemi_map)*2.0 / (np.sum(hemi_map) + np.sum(control_hemi_map))

                                dice_list.append(dice)
                            
                    #add mean dice to summary df
                    summary_df.loc[n] = [sub,code,group,cond,hemi,np.mean(dice_list)]
                    n+=1

    #save summary df
    summary_df.to_csv(f'{results_dir}/neural_map/full_map_overlap.csv',index=False)

def calc_peak_overlap():
    print('Calculating peak overlap...')

    patient_subs = sub_info[sub_info['group'] == 'patient']
    control_subs = sub_info[sub_info['group'] == 'control']


    summary_df = pd.DataFrame(columns = ['sub','code','group','cond','hemi','dice'])
    n = 0
    for task,cond, cope in zip(task_info['task'], task_info['cond'],task_info['cope']):
        print(f'Processing {cond} {task}')


        #Determine number of positions and preferred hemi
        if cond == 'word' or cond == 'tool':
            pref_hemi = 'left'

        elif cond == 'face' or cond == 'space':
            pref_hemi = 'right'
        
    
        for sub, code, group, hemi in zip(sub_info['sub'], sub_info['code'], sub_info['group'], sub_info['intact_hemi']):
            print(sub)
            sub_dir = f'{data_dir}/{sub}/ses-01'

            #check if neural map exists
            neural_map_path = f'{sub_dir}/derivatives/neural_map/{cond}_func.npy'

            if os.path.exists(neural_map_path):

                #load neural map
                neural_map = np.load(neural_map_path)

        
                #check if patient and if its the preferred hemi
                #if not flip the map
                if hemi != 'both' and hemi != pref_hemi:
                    neural_map = np.flip(neural_map,axis=1)



                for position in positions:
                    #zero out non-preferred hemi
                    hemi_map = neural_map.copy()
                    if pref_hemi == 'left':
                        hemi_map[:,:int(hemi_map.shape[1]/2)] = 0
                    elif pref_hemi == 'right':
                        hemi_map[:,int(hemi_map.shape[1]/2):] = 0

                    #zero out non-preferred position
                    if position == 'posterior':
                        hemi_map[split:,:] = 0
                    elif position == 'anterior':
                        hemi_map[:split,:] = 0
                    
                    #check if max is not 0
                    if np.max(hemi_map) != 0:

                        
                        #extract only top 10% of positive voxels
                        hemi_map[hemi_map < np.percentile(hemi_map[hemi_map>0],90)] = 0

                        dice_list = []
                        #loop through controls and calculate dice coefficient
                        for control_sub in control_subs['sub']:
                            control_sub_dir = f'{data_dir}/{control_sub}/ses-01'
                            control_neural_map_path = f'{control_sub_dir}/derivatives/neural_map/{cond}_binary.npy'

                            if os.path.exists(control_neural_map_path) and sub != control_sub:
                                control_neural_map = np.load(control_neural_map_path)

                                #check if max is not 0
                                if np.max(control_neural_map) != 0:
                                        
                                    #zero out non-preferred hemi
                                    control_hemi_map = control_neural_map.copy()
                                    if pref_hemi == 'left':
                                        control_hemi_map[:,:int(control_hemi_map.shape[1]/2)] = 0

                                    elif pref_hemi == 'right':
                                        control_hemi_map[:,int(control_hemi_map.shape[1]/2):] = 0

                                    #calculate dice coefficient
                                    dice = np.sum(hemi_map*control_hemi_map)*2.0 / (np.sum(hemi_map) + np.sum(control_hemi_map))

                                    dice_list.append(dice)
                            
                        #add mean dice to summary df
                        summary_df.loc[n] = [sub,code,group,cond,hemi,np.mean(dice_list)]
                        n+=1

    #save summary df
    summary_df.to_csv(f'{results_dir}/neural_map/peak_map_overlap.csv',index=False)

calc_full_overlap()
