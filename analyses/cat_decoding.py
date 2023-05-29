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

#task = sys.argv[1]
#roi = sys.argv[2]

task = 'spaceloc'
roi = 'dorsal_visual_cortex'

folds = 20
test_size = 0.2



if task == 'spaceloc':
    target = 'space'
    distract = 'feat'
elif task == 'tooloc':
    target = 'tool'
    distract = 'non_tool'
elif task == 'face':
    target = 'face'
    distract = 'object'
elif task == 'word':
    target = 'word'
    distract = 'object'


all_subs = params.sub_info['sub'].values

all_subs = ['sub-spaceloc1008']
for sub, group, hemi in zip(params.sub_info['sub'],params.sub_info['group'], params.sub_info['intact_hemi']):
    
    mvpa_dir = f'{params.data_dir}/{sub}/ses-01/derivatives/mvpa'

    #load target and distract data
    target_data = np.load(f'{mvpa_dir}/{roi}_{task}_{target}.npy')
    distract_data = np.load(f'{mvpa_dir}/{roi}_{task}_{distract}.npy')

    #append target and distract data
    data = np.concatenate((target_data, distract_data))

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
        clf = RidgeCV()
        clf.fit(X_train, y_train)   

        roi_acc.append(clf.score(X_test, y_test))
        #print(clf.score(X_test, y_test))

    print(f'{sub} {roi} {task} {np.mean(roi_acc)}')





