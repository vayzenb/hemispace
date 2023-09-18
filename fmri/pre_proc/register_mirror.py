'''
Creates Mirror brain for patients and registers ROIs from MNI to each individual subject
'''


curr_dir = '/user_data/vayzenbe/GitHub_Repos/hemispace'
import sys

#insert current directory to path
sys.path.insert(0,curr_dir)

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import itertools
from nilearn import image, plotting, datasets, masking
import nibabel as nib
import pdb
import os
import subprocess
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
import hemispace_params as params

#load fsl on node
#bash_cmd = f'module load fsl-6.0.3'
#subprocess.run(bash_cmd.split(), check = True)

study='hemispace'

study_dir = f"/lab_data/behrmannlab/vlad/{study}"

control_dir = f"/lab_data/behrmannlab/vlad/spaceloc"

sub_info = params.sub_info

#left is negative, right is positive
mni = load_mni152_brain_mask()
mni_affine = mni.affine
parcel_mni='/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz' #this is the MNI we use for both julian and mruczek parcels
anat_mni='/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz' #this is the MNI we use for analysis
parcel_root = "/user_data/vayzenbe/GitHub_Repos/fmri/roiParcels"
parcel_type = ""

parcels = params.rois

#TEMP FOR V1
parcel_mni='/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz' #this is the MNI we use for masks
anat_mni='/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz' #this is the MNI we use for analysis
parcels = ['ventral_visual_cortex','dorsal_visual_cortex']

#exp = 
def create_mirror_brain(sub,hemi):

    print("creating brain mirror", sub)
    sub_dir = f'{study_dir}/{sub}/ses-01/'
    #stat_dir = f'{sub_dir}/fsl/{exp[1]}/HighLevel{suf}.gfeat/cope{copes[exp[0]]}.feat/'

    #load anat
    anat_mask = image.load_img(f'{sub_dir}/anat/{sub}_ses-01_T1w_brain_mask.nii.gz')
    anat = image.load_img(f'{sub_dir}/anat/{sub}_ses-01_T1w_brain.nii.gz')
    anat = image.get_data(anat)
    affine = anat_mask.affine
    hemi_mask = image.get_data(anat_mask)

    #extract just one hemi
    mid = list((np.array((hemi_mask.shape))/2).astype(int)) #find mid point of image

    hemi_mask[hemi_mask>0] = 1 #ensure to mask all of it
    anat_flip = anat
    anat_mirror = anat
    anat_flip =anat_flip[::-1,:, :]

    if hemi == 'left':
        hemi_mask[mid[0]:, :, :] = 0 
        anat_mirror[mid[0]:,:,:] = anat_flip[mid[0]:,:,:]
    else:
        hemi_mask[:mid[0], :, :] = 0 
        anat_mirror[:mid[0],:,:] = anat_flip[:mid[0],:,:]

    anat_mirror = nib.Nifti1Image(anat_mirror, affine)  # create the volume image
    hemi_mask = nib.Nifti1Image(hemi_mask, affine)  # create a mask for just that hemi image
    nib.save(hemi_mask,f'{sub_dir}/anat/{sub}_ses-01_T1w_brain_mask_{hemi}.nii.gz')
    nib.save(anat_mirror,f'{sub_dir}/anat/{sub}_ses-01_T1w_brain_mirrored.nii.gz')
    print('mirror saved to', f'{sub_dir}/anat/{sub}_ses-01_T1w_brain_mirrored.nii.gz')

def create_hemi_mask(sub):
    """
    Creating hemispheric masks for control sub
    """
    print("creating hemisphere mask", sub)
    sub_dir = f'{study_dir}/{sub}/ses-01/'
    #stat_dir = f'{sub_dir}/fsl/{exp[1]}/HighLevel{suf}.gfeat/cope{copes[exp[0]]}.feat/'

    for hemi in ['left','right']:
         #load anat
        anat_mask = image.load_img(f'{sub_dir}/anat/{sub}_ses-01_T1w_brain_mask.nii.gz')
        affine = anat_mask.affine
    
        hemi_mask = image.get_data(anat_mask)
        
        #extract just one hemi
        mid = list((np.array((hemi_mask.shape))/2).astype(int)) #find mid point of image

        hemi_mask[hemi_mask>0] = 1 #ensure to mask all of it

        if hemi == 'left':
            hemi_mask[mid[0]:, :, :] = 0 

        else:
            hemi_mask[:mid[0], :, :] = 0 

        hemi_mask = nib.Nifti1Image(hemi_mask, affine)  # create a mask for just that hemi image
        nib.save(hemi_mask,f'{sub_dir}/anat/{sub}_ses-01_T1w_brain_mask_{hemi}.nii.gz')
                

def register_mni(sub,group):
    '''
    Register to MNI
    '''
    
    print('Registering subj to MNI...', sub)
    anat_dir = f'{study_dir}/{sub}/ses-01/anat/'
    if group == 'patient':
        anat_mirror = f'{anat_dir}/{sub}_ses-01_T1w_brain_mirrored.nii.gz'
    else:
        anat_mirror = f'{anat_dir}/{sub}_ses-01_T1w_brain.nii.gz'
    
    
    anat = f'{anat_dir}/{sub}_ses-01_T1w_brain.nii.gz'


    #pdb.set_trace()
    #create registration matrix for patient to mni
    bash_cmd = f'flirt -in {anat_mirror} -ref {anat_mni} -omat {anat_dir}/anat2stand.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12'
    subprocess.run(bash_cmd.split(), check = True)
    

    #Create mni of patient brain
    bash_cmd = f'flirt -in {anat} -ref {anat_mni} -out {anat_dir}/{sub}_ses-01_T1w_brain_stand.nii.gz -applyxfm -init {anat_dir}/anat2stand.mat -interp trilinear'
    subprocess.run(bash_cmd.split(), check = True)

    

    #create registration matrix for mni to patient
    #use parcel MNI here
    bash_cmd = f'flirt -in {parcel_mni} -ref {anat_mirror} -omat {anat_dir}/mni2anat.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12'
    subprocess.run(bash_cmd.split(), check = True)
    

def register_funcs(sub, exps):
    """
    Register highlevels to MNI
    """
    print("Registering HighLevels to MNI...")
    sub_dir = f'{study_dir}/{sub[1]}/ses-01'
    anat_dir = f'{sub_dir}/anat'
    for exp in enumerate(exps):
        print("Registering ", exp[1])
        stat_dir = f'{sub_dir}/derivatives/fsl/{exp[1]}/HighLevel{suf}.gfeat/cope{copes[exp[0]]}.feat/stats/'
        stat = f'{stat_dir}/zstat1.nii.gz'
        bash_cmd = f'flirt -in {stat} -ref {anat_mni} -out {stat_dir}/zstat1_reg.nii.gz -applyxfm -init {anat_dir}/mirror2stand.mat -interp trilinear'
        subprocess.run(bash_cmd.split(), check = True)

def register_parcels(sub, parcel_dir, parcels):
    """
    Register parcels to subject
    """
    print("Registering parcels for ", sub)
    sub_dir = f'{study_dir}/{sub}/ses-01'
    roi_dir = f'{sub_dir}/derivatives/rois'
    anat_dir = f'{sub_dir}/anat'
    anat = f'{anat_dir}/{sub}_ses-01_T1w_brain.nii.gz'
    os.makedirs(f'{roi_dir}/parcels',exist_ok=True)

    for rp in parcels:
        
        roi_parcel  = f'{parcel_dir}/{rp}.nii.gz'
        bash_cmd = f'flirt -in {roi_parcel} -ref {anat} -out {roi_dir}/parcels/{rp}.nii.gz -applyxfm -init {anat_dir}/mni2anat.mat -interp trilinear'
        subprocess.run(bash_cmd.split(), check = True)

        #binarize
        bash_cmd = f'fslmaths {roi_dir}/parcels/{rp}.nii.gz -bin {roi_dir}/parcels/{rp}.nii.gz'
        subprocess.run(bash_cmd.split(), check = True)


        #load parcel
        #roi_parcel = image.load_img(f'{parcel_dir}/{rp}.nii.gz')

        #resample to anat
        #roi_parcel = image.resample_to_img(roi_parcel, anat, interpolation='nearest')

        #binarize
        #roi_parcel = image.math_img('img>0.1',img=roi_parcel)

        #save
        #nib.save(roi_parcel,f'{roi_dir}/parcels/{rp}.nii.gz')


        #roi_parcel  = f'{parcel_dir}/r{rp}.nii.gz'
        #bash_cmd = f'flirt -in {roi_parcel} -ref {anat} -out {roi_dir}/parcels/r{rp}.nii.gz -applyxfm -init {anat_dir}/parcel2mirror.mat -interp trilinear'
        #subprocess.run(bash_cmd.split(), check = True)
        print(f"Registered {rp}")



#Create mni of patient brain
#bash_cmd = f'flirt -in {anat} -ref {anat_mni} -out {anat_dir}/{sub[1]}_ses-01_T1w_brain_stand.nii.gz -applyxfm -init {anat_dir}/parcel2mirror.mat -interp trilinear'
#subprocess.run(bash_cmd.split(), check = True)


'''
patient_subs=["hemispace1001", "hemispace1002", "hemispace1003","hemispace1004","hemispace1006","hemispace1007", "108", "109"]

control_subs=["hemispace2001", "hemispace2002", "hemispace2003",
"025", "038", "057", "059", "064", "067", "068", "071", "083", "084", "085", 
"087", "088", "093", "094", "095", "096", "097", "103", "104", "105", "106", "107", 
"spaceloc1001", "spaceloc1002", "spaceloc1003", "spaceloc1004", "spaceloc1005", "spaceloc1006",
    "spaceloc1007", "spaceloc1008", "spaceloc1009", "spaceloc1010", "spaceloc1011", "spaceloc1012",
	 "spaceloc2013", "spaceloc2014", "spaceloc2015", "spaceloc2016", "spaceloc2017", "spaceloc2018" ]
'''


""" all_subs = sub_info['sub'].values

all_subs = ['sub-111']
for sub in all_subs:
    if sub[:4] != 'sub-':
        sub = 'sub-' + sub
    
    #ex tract intact hemi of current sub from sub_info
    hemi = sub_info[sub_info['sub']==sub]['intact_hemi'].values[0]
    group = sub_info[sub_info['sub']==sub]['group'].values[0]

    parcel_dir = f'{parcel_root}/{parcel_type}'

    
    if group == 'patient':
        create_mirror_brain(sub,hemi)
    else:
        create_hemi_mask(sub)
    


    #register_mni(sub,group)
    register_parcels(sub, parcel_dir, parcels)
 """

sub = 'sub-111'
hemi = 'left'

create_mirror_brain(sub,hemi)