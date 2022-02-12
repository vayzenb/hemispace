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
import subprocess
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template

#load fsl on node
#bash_cmd = f'module load fsl-6.0.3'
#subprocess.run(bash_cmd.split(), check = True)

study='hemispace'

study_dir = f"/lab_data/behrmannlab/vlad/{study}"

control_dir = f"/lab_data/behrmannlab/vlad/spaceloc"
suf = '_roi'
exps = ['spaceloc','toolloc']
copes = [1,1]
p_hemi = ['right','left','right'] #the remaining hemi of the sub
c_hemi = ['left', 'right']


subj_list = ['hemispace1001','hemispace1002','hemispace1003', 'hemispace2001', 'hemispace2002', 'hemispace2003']
subj_list = ['hemispace1001','hemispace1002','hemispace1003']

subj_list=['hemispace2001', 'hemispace2002', 'hemispace2003']


#left is negative, right is positive
mni = load_mni152_brain_mask()
mni_affine = mni.affine
parcel_mni='/opt/fsl/6.0.3/data/standard/MNI152_T1_1mm_brain.nii.gz' #this is the MNI we use for both julian and mruczek parcels
anat_mni='/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz' #this is the MNI we use for analysis
parcel_root = "/user_data/vayzenbe/GitHub_Repos/fmri/roiParcels/"
parcel_type = "mruczek_parcels/binary"
parcels = ['PPC', 'APC']

#exp = 
def create_mirror_brain(sub):
    print("creating brain mirror", sub[1])
    sub_dir = f'{study_dir}/sub-{sub[1]}/ses-01/'
    #stat_dir = f'{sub_dir}/fsl/{exp[1]}/HighLevel{suf}.gfeat/cope{copes[exp[0]]}.feat/'

    #load anat
    anat_mask = image.load_img(f'{sub_dir}/anat/sub-{sub[1]}_ses-01_T1w_brain_mask.nii.gz')
    anat = image.load_img(f'{sub_dir}/anat/sub-{sub[1]}_ses-01_T1w_brain.nii.gz')
    anat = image.get_data(anat)
    affine = anat_mask.affine
    hemi_mask = image.get_data(anat_mask)

    #extract just one hemi
    mid = list((np.array((hemi_mask.shape))/2).astype(int)) #find mid point of image

    hemi_mask[hemi_mask>0] = 1 #ensure to mask all of it

    if p_hemi[sub[0]] == 'left':
        hemi_mask[mid[0]:, :, :] = 0 
    else:
        hemi_mask[:mid[0], :, :] = 0 

    anat_flip = anat
    anat_mirror = anat
    anat_flip =anat_flip[::-1,:, :]

    anat_mirror[:mid[0],:,:] = anat_flip[:mid[0],:,:]

    anat_mirror = nib.Nifti1Image(anat_mirror, affine)  # create the volume image
    nib.save(anat_mirror,f'{sub_dir}/anat/sub-{sub[1]}_ses-01_T1w_brain_mirrored.nii.gz')
    print('mirror saved to', f'{sub_dir}/anat/sub-{sub[1]}_ses-01_T1w_brain_mirrored.nii.gz')


def register_mni(sub):
    '''
    Register to MNI
    '''
    print('Registering subj to MNI...')
    anat_dir = f'{study_dir}/sub-{sub[1]}/ses-01/anat/'
    if int(sub[1][-4:]) < 2000:
        anat_mirror = f'{anat_dir}/sub-{sub[1]}_ses-01_T1w_brain_mirrored.nii.gz'
    else:
        anat_mirror = f'{anat_dir}/sub-{sub[1]}_ses-01_T1w_brain_brain.nii.gz'
    anat = f'{anat_dir}/sub-{sub[1]}_ses-01_T1w_brain.nii.gz'


    #pdb.set_trace()
    #create registration matrix for patient to mni
    bash_cmd = f'flirt -in {anat_mirror} -ref {anat_mni} -omat {anat_dir}/mirror2stand.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12'
    subprocess.run(bash_cmd.split(), check = True)
    print('created mirror to standard mat')

    #Create mni of patient brain
    bash_cmd = f'flirt -in {anat} -ref {anat_mni} -out {anat_dir}/sub-{sub[1]}_ses-01_T1w_brain_stand.nii.gz -applyxfm -init {anat_dir}/mirror2stand.mat -interp trilinear'
    subprocess.run(bash_cmd.split(), check = True)

    print('Registered patient to MNI')

    #create registration matrix for mni to patient
    #use parcel MNI here
    bash_cmd = f'flirt -in {parcel_mni} -ref {anat_mirror} -omat {anat_dir}/parcel2mirror.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12'
    subprocess.run(bash_cmd.split(), check = True)
    print('Created mni_1mm to subj')

def register_funcs(sub, exps):
    """
    Register highlevels to MNI
    """
    print("Registering HighLevels to MNI...")
    sub_dir = f'{study_dir}/sub-{sub[1]}/ses-01'
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
    print("Registering parcels for ", sub[1])
    sub_dir = f'{study_dir}/sub-{ss[1]}/ses-01'
    roi_dir = f'{sub_dir}/derivatives/rois'
    anat_dir = f'{sub_dir}/anat'
    anat = f'{anat_dir}/sub-{sub[1]}_ses-01_T1w_brain.nii.gz'
    os.makedirs(f'{roi_dir}/parcels',exist_ok=True)

    for rp in parcels:
        
        roi_parcel  = f'{parcel_dir}/l{rp}.nii.gz'
        bash_cmd = f'flirt -in {roi_parcel} -ref {anat} -out {roi_dir}/parcels/l{rp}.nii.gz -applyxfm -init {anat_dir}/parcel2mirror.mat -interp trilinear'
        subprocess.run(bash_cmd.split(), check = True)

        roi_parcel  = f'{parcel_dir}/r{rp}.nii.gz'
        bash_cmd = f'flirt -in {roi_parcel} -ref {anat} -out {roi_dir}/parcels/r{rp}.nii.gz -applyxfm -init {anat_dir}/parcel2mirror.mat -interp trilinear'
        subprocess.run(bash_cmd.split(), check = True)
        print(f"Registered {rp}")



#Create mni of patient brain
#bash_cmd = f'flirt -in {anat} -ref {anat_mni} -out {anat_dir}/sub-{sub[1]}_ses-01_T1w_brain_stand.nii.gz -applyxfm -init {anat_dir}/parcel2mirror.mat -interp trilinear'
#subprocess.run(bash_cmd.split(), check = True)





for ss in enumerate(subj_list):
    
    #create_mirror_brain(ss)
    register_mni(ss)
    register_funcs(ss,exps)
    register_parcels(ss, f'{parcel_root}/{parcel_type}', parcels)
