curr_dir = f'/user_data/vayzenbe/GitHub_Repos/hemispace'
import sys
sys.path.append(curr_dir)

import numpy as np
import pandas as pd
import itertools
from nilearn import image, plotting, datasets, masking
import nibabel as nib
import pdb
import os
import hemispace_params as params

#import ridgeCV
from sklearn.linear_model import RidgeCV

