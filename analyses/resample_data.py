"""
Resample control data
"""


curr_dir = f'/user_data/vayzenbe/GitHub_Repos/hemispace'
import sys
sys.path.append(curr_dir)

import numpy as np
import pandas as pd
import hemispace_params as params
import itertools
import pdb

data_dir = params.data_dir
results_dir = params.results_dir

sub_info = params.sub_info
task_info = params.task_info

suf = params.suf
rois = params.rois
hemis = params.hemis

#load summary file
summary_df = pd.read_csv(f'{results_dir}/hemispace_summary_vals{suf}.csv')

#number of resamples
iter = 10000

#number of subs to pull on each
n_subs = 3

#extract control subs
control_subs = sub_info[sub_info['group'] == 'control']

#make a list of all possible combinations of cond, hemi, and roi
all_combos = list(itertools.product(task_info['cond'], hemis, rois))
all_combos = ['_'.join(list(ele)) for ele in all_combos]

#create dataframe with all possible combinations of condition hemi and roi as columns
roi_size_df = pd.DataFrame(columns =all_combos)
mean_act_df = pd.DataFrame(columns =all_combos)
cortex_vol_df = pd.DataFrame(columns =all_combos)
sum_selec_df = pd.DataFrame(columns =all_combos)

for ii in range(0,iter):
    for hemi in hemis:
        for roi in rois:
            for cond in task_info['cond']:
                #select data that meets cond
                curr_data = summary_df[(summary_df['cond'] == cond) & (summary_df['roi'] == roi) & (summary_df['hemi'] == hemi)]

                #select n_subs random subs
                curr_subs = curr_data.sample(n = n_subs, replace = True)

                #get mean of each value
                roi_size, mean_act,  cortex_vol, sum_selec = curr_subs['roi_size'].mean(), curr_subs['mean_act'].mean(), curr_subs['volume'].mean(), curr_subs['sum_selec'].mean()

                #append to to dataframe
                roi_size_df.loc[ii, f'{cond}_{hemi}_{roi}'] = roi_size
                mean_act_df.loc[ii, f'{cond}_{hemi}_{roi}'] = mean_act
                cortex_vol_df.loc[ii, f'{cond}_{hemi}_{roi}'] = cortex_vol
                sum_selec_df.loc[ii, f'{cond}_{hemi}_{roi}'] = sum_selec

#save each resample
roi_size_df.to_csv(f'{results_dir}/roi_size_resamples{suf}.csv', index=False)
mean_act_df.to_csv(f'{results_dir}/mean_act_resamples{suf}.csv', index=False)
cortex_vol_df.to_csv(f'{results_dir}/cortex_vol_resamples{suf}.csv', index=False)
sum_selec_df.to_csv(f'{results_dir}/sum_selec_resamples{suf}.csv', index=False)






