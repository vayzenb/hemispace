curr_dir = f'/user_data/vayzenbe/GitHub_Repos/hemispace'
import numpy as np
import pandas as pd
import subprocess
import os
import pdb

import sys
sys.path.append(curr_dir)

import hemispace_params as params

#import ridgecv
from sklearn.linear_model import RidgeCV
#import stratified shuffle split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

#task = sys.argv[1]
#roi = sys.argv[2]


folds = 20
test_size = 0.2


results_dir = params.results_dir

task_info = params.task_info
def lookup_task_info(cond):
    
    if cond == 'space':
        target = 'space'
        distract = 'feat'
        roi = 'dorsal_visual_cortex'
    elif cond == 'tool':
        target = 'tool'
        distract = 'non_tool'
        roi = 'dorsal_visual_cortex'
    elif cond == 'face':
        target = 'face'
        distract = 'object'
        roi = 'ventral_visual_cortex'
    elif cond == 'word':
        target = 'word'
        distract = 'object'
        roi = 'ventral_visual_cortex'

    return target, distract, roi



summary_df = pd.DataFrame(columns = ['sub', 'group', 'hemi', 'roi','cond', 'acc', 'se'])
for task,cond in zip(task_info['task'], task_info['cond']):

    print(f'Running {task} {cond}...')
    target, distract, roi = lookup_task_info(cond) #get target, distract, and roi for task
    #loop through all subs
    for sub, group, hemi in zip(params.sub_info['sub'],params.sub_info['group'], params.sub_info['intact_hemi']):
        
        mvpa_dir = f'{params.data_dir}/{sub}/ses-01/derivatives/mvpa'
        
        if hemi == 'both':
            hemis = ['left','right']
        else:
            hemis = [hemi]
        
        for hemi in hemis:

            #check if task and cond exist in mvpa dir
            if os.path.exists(f'{mvpa_dir}/{hemi}_{roi}_{task}_{target}.npy'):    
                #print(f'Running {sub} {hemi}_{roi}_{task}_{target} vs {distract}...')            

                #load target and distract data
                target_data = np.load(f'{mvpa_dir}/{hemi}_{roi}_{task}_{target}.npy')
                distract_data = np.load(f'{mvpa_dir}/{hemi}_{roi}_{task}_{distract}.npy')

                #append target and distract data
                data = np.concatenate((target_data, distract_data))

                #remove columns with nans
                data = data[:,~np.isnan(data).any(axis=0)]

                #create labels with 1s for target and 0s for distract
                labels = np.concatenate((np.ones(target_data.shape[0]), np.zeros(distract_data.shape[0])))

                #set up cross validation
                X = data
                y = labels
                
                sss = StratifiedShuffleSplit(n_splits=folds, test_size=test_size, random_state=0)
                sss.get_n_splits(X, y)


                roi_acc = []
                for train_index, test_index in sss.split(X, y):

                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    #pdb.set_trace()
                    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
                    clf.fit(X_train, y_train)   

                    roi_acc.append(clf.score(X_test, y_test))
                    #print(clf.score(X_test, y_test))

                #add data to summary df using concat
                summary_df = pd.concat([summary_df, pd.DataFrame([[sub, group, hemi, roi, cond, np.mean(roi_acc), np.std(roi_acc)]], columns = ['sub', 'group', 'hemi', 'roi','cond', 'acc', 'se'])])
                
                

                #print(f'{sub} {hemi} {roi} {task} {np.mean(roi_acc)}')

            summary_df.to_csv(f'{results_dir}/hemispace_decoding.csv', index=False)




