import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from plotnine import *
import itertools
from nilearn import image, plotting, datasets, masking
import nibabel as nib
import pdb
import os
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template

study='hemispace'

patient_dir = f"/lab_data/behrmannlab/vlad/{study}"

control_dir = f"/lab_data/behrmannlab/vlad/hemispace"
suf = '_roi'
exps = ['spaceloc','toolloc']
copes = [1,1]
p_hemi = ['right', 'left','right'] #the remaining hemi of the sub
c_hemi = ['left', 'right']


c_subs =["hemispace2001", "hemispace2002", "hemispace2003"]

p_subs = ['hemispace1001','hemispace1002','hemispace1003']

#for wang parcels iterate through the numbers you want
parcel_num = list(range(16,25))
parcels = []
for pn in parcel_num:
    parcels.append(f'perc_VTPM_vol_roi{pn}_lh')
    parcels.append(f'perc_VTPM_vol_roi{pn}_rh')


#left is negative, right is positive
mni = load_mni152_brain_mask()
mni_affine = mni.affine

#load hemisubj
control_summary = pd.DataFrame(columns=['subj','task','hemi','selec_spread'])
patient_summary = pd.DataFrame(columns=['subj','task','hemi', 'selec_spread'])

def combine_dorsal(sub,parcels):
    for rp in enumerate(parcels):
        if rp[0] == 0:
            full_parcel = image.get_data(image.load_img(f'{patient_dir}/sub-{sub}/ses-01/derivatives/rois/parcels/{rp}.nii.gz'))
        else:
            curr_parcel = image.get_data(image.load_img(f'{patient_dir}/sub-{sub}/ses-01/derivatives/rois/parcels/{rp}.nii.gz'))
            full_parcel = full_parcel * curr_parcel
    
    return full_parcel
            



def dorsal_selectivity_spread():
    for exp in enumerate(exps):

        '''
        Extract activation spread from patients
        '''
        p_acts = []
        for ss in enumerate(p_subs):
            #set up dirs
            sub_dir = f'{patient_dir}/sub-{ss[1]}/ses-01/derivatives/'
            stat_dir = f'{sub_dir}/fsl/{exp[1]}/HighLevel{suf}.gfeat/cope{copes[exp[0]]}.feat/'

            #load anat
            anat_mask = image.load_img(f'{patient_dir}/sub-{ss[1]}/ses-01/anat/sub-{ss[1]}_ses-01_T1w_brain_mask.nii.gz')
            affine = anat_mask.affine
            hemi_mask = image.get_data(anat_mask)

            #extract just one hemi
            mid = list((np.array((hemi_mask.shape))/2).astype(int)) #find mid point of image
            hemi_mask[hemi_mask>0] = 1 #ensure to mask all of it

            #load and combine dorsal ROIs  
            parcel = combine_dorsal(ss[1], parcel)


            if p_hemi[ss[0]] == 'left':
                hemi_mask[mid[0]:, :, :] = 0 
            else:
                hemi_mask[:mid[0], :, :] = 0 

            mask_size = np.sum(hemi_mask)
            #hemi_mask = nib.Nifti1Image(anat_np, affine)  # create the volume image

            zstat = image.get_data(image.load_img(f'{stat_dir}/stats/zstat1.nii.gz'))
            clust_mask = image.get_data(image.load_img(f'{stat_dir}/cluster_mask_zstat1.nii.gz'))
            clust_mask[clust_mask>0] = 1

            zstat_mask = zstat  * hemi_mask * clust_mask * parcel

            p_spread = np.sum(zstat_mask)/mask_size

            patient_summary = patient_summary.append(pd.Series([ss[1],exp,p_hemi[ss[0]], p_spread], index = patient_summary.columns), ignore_index = True)
            
        
        patient_summary.to_csv('patient_dorsal_summary.csv',index = False)
    
    #p_summary = pd.Series(p_acts, index= exps) #create index for patients

    

    '''
    Extract activation spread from controls
    '''
    act_spread = []
    for ss in enumerate(c_subs):
        #set up dirs
        sub_dir = f'{control_dir}/sub-{ss[1]}/ses-01/derivatives/'
        stat_dir = f'{sub_dir}/fsl/{exp[1]}/HighLevel{suf}.gfeat/cope{copes[exp[0]]}.feat/'

        parcel = combine_dorsal(ss[1], parcel)

        hemi_spread = []
        for hemi in c_hemi:
            anat_mask = image.load_img(f'{control_dir}/sub-{ss[1]}/ses-01/anat/sub-{ss[1]}_ses-01_T1w_brain_mask.nii.gz')
            hemi_mask = image.get_data(anat_mask)
            hemi_mask[hemi_mask>0] = 1 #ensure to mask all of it
                        
            if hemi == 'left':
                hemi_mask[mid[0]:, :, :] = 0 
            else:
                hemi_mask[:mid[0], :, :] = 0 
            
            mask_size = np.sum(hemi_mask)
            #hemi_mask = nib.Nifti1Image(anat_np, affine)  # create the volume image

            zstat = image.get_data(image.load_img(f'{stat_dir}/stats/zstat1.nii.gz'))
            
            clust_mask = image.get_data(image.load_img(f'{stat_dir}/cluster_mask_zstat1.nii.gz'))
            clust_mask[clust_mask>0] = 1

            zstat_mask = zstat *  hemi_mask * clust_mask * parcel
            
            act_val =np.sum(zstat_mask)/mask_size
            #if act_val == 0:
            #    act_val = np.nan
            

            control_summary = patient_summary.append(pd.Series([ss[1],exp,hemi, act_val], index = control_summary.columns), ignore_index = True)
            
        
        control_summary.to_csv('control_dorsal_summary.csv',index = False)
    


def whole_brain_spread():
    for exp in enumerate(exps):

        '''
        Extract activation spread from patients
        '''
        p_acts = []
        for ss in enumerate(p_subs):
            #set up dirs
            sub_dir = f'{patient_dir}/sub-{ss[1]}/ses-01/derivatives/'
            stat_dir = f'{sub_dir}/fsl/{exp[1]}/HighLevel{suf}.gfeat/cope{copes[exp[0]]}.feat/'

            #load anat
            anat_mask = image.load_img(f'{patient_dir}/sub-{ss[1]}/ses-01/anat/sub-{ss[1]}_ses-01_T1w_brain_mask.nii.gz')
            affine = anat_mask.affine
            hemi_mask = image.get_data(anat_mask)
            

            #extract just one hemi
            mid = list((np.array((hemi_mask.shape))/2).astype(int)) #find mid point of image
            hemi_mask[hemi_mask>0] = 1 #ensure to mask all of it

            if p_hemi[ss[0]] == 'left':
                hemi_mask[mid[0]:, :, :] = 0 
            else:
                hemi_mask[:mid[0], :, :] = 0 

            mask_size = np.sum(hemi_mask)
            #hemi_mask = nib.Nifti1Image(anat_np, affine)  # create the volume image

            zstat = image.get_data(image.load_img(f'{stat_dir}/stats/zstat1.nii.gz'))
            clust_mask = image.get_data(image.load_img(f'{stat_dir}/cluster_mask_zstat1.nii.gz'))
            clust_mask[clust_mask>0] = 1

            zstat_mask = zstat  * hemi_mask * clust_mask

            p_spread = np.sum(zstat_mask)/mask_size
            p_acts.append(p_spread)
        
        p_acts = np.array(p_acts)
        patient_summary[exp[1]] = p_acts
    
    #p_summary = pd.Series(p_acts, index= exps) #create index for patients

    

    '''
    Extract activation spread from controls
    '''
    act_spread = []
    for ss in enumerate(c_subs):
        #set up dirs
        sub_dir = f'{control_dir}/sub-{ss[1]}/ses-01/derivatives/'
        stat_dir = f'{sub_dir}/fsl/{exp[1]}/HighLevel{suf}.gfeat/cope{copes[exp[0]]}.feat/'

        hemi_spread = []
        for hemi in c_hemi:
            anat_mask = image.load_img(f'{control_dir}/sub-{ss[1]}/ses-01/anat/sub-{ss[1]}_ses-01_T1w_brain_mask.nii.gz')
            hemi_mask = image.get_data(anat_mask)
            hemi_mask[hemi_mask>0] = 1 #ensure to mask all of it
                        
            if hemi == 'left':
                hemi_mask[mid[0]:, :, :] = 0 
            else:
                hemi_mask[:mid[0], :, :] = 0 
            
            mask_size = np.sum(hemi_mask)
            #hemi_mask = nib.Nifti1Image(anat_np, affine)  # create the volume image

            zstat = image.get_data(image.load_img(f'{stat_dir}/stats/zstat1.nii.gz'))
            
            clust_mask = image.get_data(image.load_img(f'{stat_dir}/cluster_mask_zstat1.nii.gz'))
            clust_mask[clust_mask>0] = 1

            zstat_mask = zstat *  hemi_mask * clust_mask
            
            act_val =np.sum(zstat_mask)/mask_size
            if act_val == 0:
                hemi_spread.append(np.nan)
            else:    
                hemi_spread.append(act_val)

        act_spread.append(hemi_spread)

    act_spread = np.array(act_spread)
    control_summary[f'l{exp[1]}'] = act_spread[:,0]
    control_summary[f'r{exp[1]}'] = act_spread[:,1]

