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



#number of resamples
iter = 10000

#number of subs to pull on each resample
n_subs = 4

rois = params.rois

#make a list of all possible combinations of cond, hemi, and roi
all_combos = list(itertools.product(task_info['cond'], hemis, rois))
all_combos = ['_'.join(list(ele)) for ele in all_combos]

def resample_selectivity():
    """
    Resample selectivity data
    """
    print('resampling selectivity data...')
    results_dir = f'{params.results_dir}/selectivity'

    #load summary file
    summary_df = pd.read_csv(f'{results_dir}/selectivity_summary{suf}.csv')

    #extract only control subs
    summary_df = summary_df[summary_df['group'] == 'control']
    #create dataframe with all possible combinations of condition hemi and roi as columns
    roi_size_df = pd.DataFrame(columns =all_combos)
    mean_act_df = pd.DataFrame(columns =all_combos)
    cortex_vol_df = pd.DataFrame(columns =all_combos)
    sum_selec_df = pd.DataFrame(columns =all_combos)
    sum_selec_norm_df = pd.DataFrame(columns =all_combos)

    

    for ii in range(0,iter):
        
        for hemi in hemis:
            for roi in rois:
                for cond in task_info['cond']:
                    
                    
                    #select data that meets cond
                    curr_data = summary_df[(summary_df['cond'] == cond) & (summary_df['roi'] == roi) & (summary_df['hemi'] == hemi)]

                    #select n_subs random subs
                    curr_subs = curr_data.sample(n = n_subs, replace = True)

                    #get mean of each value
                    roi_size, mean_act,  cortex_vol, sum_selec, sum_selec_norm = curr_subs['roi_size'].mean(), curr_subs['mean_act'].mean(), curr_subs['volume'].mean(), curr_subs['sum_selec'].mean(), curr_subs['sum_selec_norm'].mean()

                    #append to to dataframe
                    roi_size_df.loc[ii, f'{cond}_{hemi}_{roi}'] = roi_size
                    mean_act_df.loc[ii, f'{cond}_{hemi}_{roi}'] = mean_act
                    cortex_vol_df.loc[ii, f'{cond}_{hemi}_{roi}'] = cortex_vol
                    sum_selec_df.loc[ii, f'{cond}_{hemi}_{roi}'] = sum_selec
                    sum_selec_norm_df.loc[ii, f'{cond}_{hemi}_{roi}'] = sum_selec_norm


    #save each resample
    roi_size_df.to_csv(f'{results_dir}/resamples/roi_size_resamples{suf}.csv', index=False)
    mean_act_df.to_csv(f'{results_dir}/resamples/mean_act_resamples{suf}.csv', index=False)
    cortex_vol_df.to_csv(f'{results_dir}/resamples/volume_resamples{suf}.csv', index=False)
    sum_selec_df.to_csv(f'{results_dir}/resamples/sum_selec_resamples{suf}.csv', index=False)
    sum_selec_norm_df.to_csv(f'{results_dir}/resamples/sum_selec_norm_resamples{suf}.csv', index=False)


def generic_resample(data_type, summary_df, data_conds, roi = ''):
    print(f'Resampling {data_type}....')

    if roi != '':
        roi = f'_{roi}'
        
    summary_df = summary_df[summary_df['group'] == 'control']
    resample_df = pd.DataFrame(columns =data_conds)

    for ii in range(0,iter):
        #select n_subs random subs
        curr_subs = summary_df.sample(n = n_subs, replace = True)
        
        for cond in data_conds:
            #get mean of each value
            val= curr_subs[cond].mean()

            #check if val is a numebr
            if np.isnan(val):
                pdb.set_trace()

            #append to to dataframe
            resample_df.loc[ii, f'{cond}'] = val

    #save resample df
    resample_df.to_csv(f'{results_dir}/{data_type}/{data_type}_resamples{roi}{suf}.csv', index=False)
                







def resample_decoding():
    """
    Resample decoding data
    """

    print('resampling decoding data...')

    results_dir = f'{params.results_dir}/decoding'

    #load summary file
    summary_df = pd.read_csv(f'{results_dir}/decoding_summary{suf}.csv')

    #extract only control subs
    summary_df = summary_df[summary_df['group'] == 'control']
    #create dataframe with all possible combinations of condition hemi and roi as columns
    decoding_df = pd.DataFrame(columns =all_combos)


    for ii in range(0,iter):
        
        for hemi in hemis:
            for roi in rois:
                for cond in task_info['cond']:
                    
                    try:
                        #select data that meets cond
                        curr_data = summary_df[(summary_df['cond'] == cond) & (summary_df['roi'] == roi) & (summary_df['hemi'] == hemi)]

                        #select n_subs random subs
                        curr_subs = curr_data.sample(n = n_subs, replace = True)

                        #get mean of each value
                        acc= curr_subs['acc'].mean()

                        
                        

                        #append to to dataframe
                        decoding_df.loc[ii, f'{cond}_{hemi}_{roi}'] = acc
                    except:
                        continue
                        

    #save each resample
    decoding_df.to_csv(f'{results_dir}/resamples/acc_resamples{suf}.csv', index=False)


def resample_neural_map():
    """
    Resample neural 
    """

    print('resampling map data...')

    results_dir = f'{params.results_dir}/neural_map'

    #load summary file
    summary_df = pd.read_csv(f'{results_dir}/full_map_overlap{suf}.csv')

    #extract only control subs
    summary_df = summary_df[summary_df['group'] == 'control']

    
    #create dataframe for each cond
    resample_df = pd.DataFrame(columns =task_info['cond'])

    for ii in range(0,iter):
            
            for cond in task_info['cond']:
                
                #select data that meets cond
                curr_data = summary_df[(summary_df['cond'] == cond)]
    
                #select n_subs random subs
                curr_subs = curr_data.sample(n = n_subs, replace = True)
                
    
                #get mean of each value
                overlap = curr_subs['dice'].mean()
    
                #append to to dataframe
                resample_df.loc[ii, f'{cond}'] = overlap

    
    #save resamples
    resample_df.to_csv(f'{results_dir}/map_overlap_resamples{suf}.csv', index=False)


#resample_selectivity()
#resample_decoding()
#resample_neural_map()
suf = '_roi'
data_type = 'confound'
data_conds = ['tsnr']
summary_df = pd.read_csv(f'{params.results_dir}/{data_type}/{data_type}_summary{suf}.csv')

#extract controls
data_summary = summary_df[summary_df['group'] == 'control']

#replace control tsnr with mean for each sub
for roi in data_summary['roi'].unique():
    for sub in data_summary['sub'].unique():
        mean_tsnr = data_summary[(data_summary['sub'] == sub) & (data_summary['roi'] == roi)]['tsnr'].mean()
        data_summary.loc[(data_summary['sub'] == sub) & (data_summary['roi'] == roi), 'tsnr'] = mean_tsnr

#drop task column
data_summary = data_summary.drop(columns = ['hemi','task', 'rot','trans'])

#drop rows with nans
data_summary = data_summary.dropna()

#drop duplicates
data_summary = data_summary.drop_duplicates()



for roi in data_summary['roi'].unique():
    roi_summary = data_summary[data_summary['roi'] == roi]
    
    generic_resample(data_type, roi_summary, data_conds, roi)