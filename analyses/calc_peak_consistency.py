'''
Calculates the peak coordinates for each subject and each condition
'''

curr_dir = f'/user_data/vayzenbe/GitHub_Repos/hemispace'

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

iter = 10000
#number of subs to pull on each resample
n_subs = 4

#point to split posterior and anterior
split = 27


def calc_peak_coord():
    print('Calculating peak coordinates...')

    summary_df = pd.DataFrame(columns = ['sub','group','cond','hemi','position','x','y'])
    n = 0
    for task,cond, cope in zip(task_info['task'], task_info['cond'],task_info['cope']):
        print(f'Processing {cond} {task}')


        #Determine number of positions and preferred hemi
        if cond == 'word' or cond == 'tool':
            pref_hemi = 'left'

        elif cond == 'face' or cond == 'space':
            pref_hemi = 'right'
        

        positions = ['posterior','anterior']
        func_list = []
        binary_list = []     
        for sub, group, hemi in zip(sub_info['sub'], sub_info['group'], sub_info['intact_hemi']):
            sub_dir = f'{data_dir}/{sub}/ses-01'

            #check if neural map exists
            neural_map_path = f'{sub_dir}/derivatives/neural_map/{cond}_func.npy'

            if os.path.exists(neural_map_path):

                #load neural map
                neural_map = np.load(neural_map_path)

                #check which hemis the sub has
                if hemi == 'both':
                    curr_hemis = ['left','right']
                else:
                    curr_hemis = [hemi]


                for curr_hemi in curr_hemis:
                    hemi_map = neural_map.copy()
                    if curr_hemi == 'left':
                        hemi_map[:,:int(hemi_map.shape[1]/2)] = 0
                        
                    elif curr_hemi == 'right':
                        hemi_map[:,int(hemi_map.shape[1]/2):] = 0
                    
                    #if hemi is specified, then its a patient
                    #if hemi is also not the preferred hemi, then flip the map
                    if hemi != 'both' and curr_hemi != pref_hemi:
                        hemi_map = np.flip(hemi_map,axis=1)
                        print('flipped for', sub, curr_hemi, cond)

                    #loop through possible positions and extract peak
                    for position in positions:
                        if position == 'posterior':
                            peak = np.max(hemi_map[:split,:])
                        elif position == 'anterior':
                            peak = np.max(hemi_map[split:,:])
                        elif position == 'all':
                            peak = np.max(hemi_map)

                        

                        if peak != 0:
                            #find coordinates of peak
                            peak_coord = np.where(hemi_map == peak)
                            
                            
                            #add to summary df
                            summary_df.loc[n] = [sub,group,cond,curr_hemi,position,peak_coord[0][0],peak_coord[1][0]]
                            n += 1


                        
                    
    #save summary df
    summary_df.to_csv(f'{results_dir}/neural_map/peak_coords.csv',index=False)


def calc_patient_distance():             
    print('Calculating average distance between patient and control peak coordinates...')
    #load peak coords
    peak_coords = pd.read_csv(f'{results_dir}/neural_map/peak_coords.csv')

    #calculate average distance between peak coordinates
    avg_dist = []   

    patient_data = peak_coords[peak_coords['group']=='patient']
    control_data = peak_coords[peak_coords['group']=='control']

    summary_df = pd.DataFrame(columns = ['sub','group','cond','hemi','position','x','y'])
    n = 0
    for cond in task_info['cond']:
        print(f'Processing {cond}')

        #Determine number of positions and preferred hemi
        if cond == 'word' or cond == 'tool':
            pref_hemi = 'left'

        elif cond == 'face' or cond == 'space':
            pref_hemi = 'right'
        
        positions = ['posterior','anterior']
        for position in positions:
            #get patient data for cond and position
            curr_patient_data = patient_data[(patient_data['cond']==cond) & (patient_data['position']==position)]
            #reset index
            curr_patient_data = curr_patient_data.reset_index(drop=True)

            #get control data for cond and position and preferred hemi
            curr_control_data = control_data[(control_data['cond']==cond) & (control_data['position']==position) & (control_data['hemi']==pref_hemi)]

            #compute distance between every patient coordinate with every control coordinate
            #and append as new column in patient data
            for i in range(len(curr_patient_data)):
                curr_dist = []
                for j in range(len(curr_control_data)):
                    curr_dist.append(np.sqrt((curr_patient_data.iloc[i]['x']-curr_control_data.iloc[j]['x'])**2 + (curr_patient_data.iloc[i]['y']-curr_control_data.iloc[j]['y'])**2))

                #append mean dist to patient data
                curr_patient_data.loc[i,'dist'] = np.mean(curr_dist)

 
            #add curr_patient_data to summary_df
            summary_df = summary_df.append(curr_patient_data)
            
    #save summary df
    summary_df.to_csv(f'{results_dir}/neural_map/patient_dists.csv',index=False)


def resample_controls(iter=10000):
    print('Resampling control data...')

    #load peak coords
    peak_coords = pd.read_csv(f'{results_dir}/neural_map/peak_coords.csv')

    #extract control data
    control_data = peak_coords[peak_coords['group']=='control']

    #create empty df to store resampled control data
    resample_data = pd.DataFrame()

    #resample control data
    for ii in range(0,iter):

        for cond in task_info['cond']:


            #Determine number of positions and preferred hemi
            if cond == 'word' or cond == 'tool':
                pref_hemi = 'left'

            elif cond == 'face' or cond == 'space':
                pref_hemi = 'right'
            
            positions = ['posterior','anterior']
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






calc_peak_coord()
calc_patient_distance()
resample_controls()