#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
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

'''
Sub info
'''
#should correspond to the intact hemi of the sub
#should have as many items as there are subs
p_hemi = ['left', 'right','right', 'left'] 
p_subs = ['hemispace1002','hemispace1003','hemispace1006','hemispace1007']

#controls always have left + right
c_hemi = ['left', 'right']
c_subs =["hemispace2001", "hemispace2002", "hemispace2003"]


#whether to norm the selectivity by size of the mask
#set to _norm if yes, _raw if now
file_suf ='_norm'

'''
Select the pathway to extract data from
'''
region = 'dorsal'

if region == 'ventral':
    '''
    Ventral params
    '''
    print('Summed selectivity for ventral...')
    exps = ['loc','loc']
    cond = ['face','word']
    copes = [1,4] #ventral
    parcel_num = list(range(7,16)) #this is for ventral

elif region == 'dorsal':
    '''
    Dorsal params
    '''
    print('Summed selectivity for dorsal...')
    exps = ['spaceloc','toolloc']
    cond = ['space', 'tool','face','word']
    copes = [1,1] #dorsal
    parcel_num = list(range(16,25)) #this is for dorsal






#for wang parcels iterate through the numbers you want

parcels = []
for pn in parcel_num:
    parcels.append(f'perc_VTPM_vol_roi{pn}_lh')
    parcels.append(f'perc_VTPM_vol_roi{pn}_rh')




#left is negative, right is positive
mni = load_mni152_brain_mask()
mni_affine = mni.affine


def combine_rois(sub,parcels):
    #for wang parcels iterate through the numbers you want
   
    for rp in enumerate(parcels):
        curr_parcel = image.get_data(image.load_img(f'{patient_dir}/sub-{sub}/ses-01/derivatives/rois/parcels/{rp[1]}.nii.gz'))
        
        if rp[0] == 0:
            full_parcel = curr_parcel
        else:
            full_parcel = full_parcel+curr_parcel    

    full_parcel[full_parcel>0] = 1
    return full_parcel
            

def combine_ventral(sub):
    #for wang parcels iterate through the numbers you want
    parcel_num = list(range(6,16))
    parcels = []
    for pn in parcel_num:
        parcels.append(f'perc_VTPM_vol_roi{pn}_lh')
        parcels.append(f'perc_VTPM_vol_roi{pn}_rh')

    #combine parcels    
    for rp in enumerate(parcels):
        curr_parcel = image.get_data(image.load_img(f'{patient_dir}/sub-{sub}/ses-01/derivatives/rois/parcels/{rp[1]}.nii.gz'))
        
        if rp[0] == 0:
            full_parcel = curr_parcel
        else:
            full_parcel = full_parcel+curr_parcel    

    full_parcel[full_parcel>0] = 1
    return full_parcel

def pathway_summed_selectivity():
    control_summary = pd.DataFrame(columns=['subj','group','task','hemi','selec_spread'])
    patient_summary = pd.DataFrame(columns=['subj','group','task','hemi', 'selec_spread'])
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
            hemi_mask = image.get_data(image.load_img(f'{patient_dir}/sub-{ss[1]}/ses-01/anat/sub-{ss[1]}_ses-01_T1w_brain_mask_{p_hemi[ss[0]]}.nii.gz'))
            #load and combine dorsal ROIs  
            parcel = combine_rois(ss[1], parcels)

            mask_size = np.sum(hemi_mask)
            

            zstat = image.get_data(image.load_img(f'{stat_dir}/stats/zstat1.nii.gz'))
            clust_mask = image.get_data(image.load_img(f'{stat_dir}/cluster_mask_zstat1.nii.gz'))
            clust_mask[clust_mask>0] = 1

            zstat_mask = zstat  * hemi_mask * clust_mask * parcel


            p_spread = np.sum(zstat_mask)

            if file_suf == '_norm':
                p_spread = p_spread / mask_size
            
            patient_summary = patient_summary.append(pd.Series([ss[1],'patient',cond[exp[0]],p_hemi[ss[0]], p_spread], index = patient_summary.columns), ignore_index = True)
        
        patient_summary.to_csv(f'results/patient_{region}_summary{file_suf}.csv',index = False)
    
    #p_summary = pd.Series(p_acts, index= exps) #create index for patients

        

        '''
        Extract activation spread from controls
        '''
        act_spread = []
        for ss in enumerate(c_subs):
            #set up dirs
            sub_dir = f'{control_dir}/sub-{ss[1]}/ses-01/derivatives/'
            stat_dir = f'{sub_dir}/fsl/{exp[1]}/HighLevel{suf}.gfeat/cope{copes[exp[0]]}.feat/'

            parcel = combine_rois(ss[1], parcels)

            hemi_spread = []
            for hemi in c_hemi:
                anat_mask = image.load_img(f'{control_dir}/sub-{ss[1]}/ses-01/anat/sub-{ss[1]}_ses-01_T1w_brain_mask.nii.gz')
                hemi_mask = image.get_data(image.load_img(f'{patient_dir}/sub-{ss[1]}/ses-01/anat/sub-{ss[1]}_ses-01_T1w_brain_mask_{hemi}.nii.gz'))
                #load and combine dorsal ROIs  
                parcel = combine_rois(ss[1], parcels)

                mask_size = np.sum(hemi_mask)
                
                zstat = image.get_data(image.load_img(f'{stat_dir}/stats/zstat1.nii.gz'))
                clust_mask = image.get_data(image.load_img(f'{stat_dir}/cluster_mask_zstat1.nii.gz'))
                clust_mask[clust_mask>0] = 1

                zstat_mask = zstat  * hemi_mask * clust_mask * parcel
                
                act_val =np.sum(zstat_mask)

                if file_suf == '_norm':
                    act_val = act_val / mask_size
                #if act_val == 0:
                #    act_val = np.nan
                

                control_summary = control_summary.append(pd.Series([ss[1],'control',cond[exp[0]],hemi, act_val], index = control_summary.columns), ignore_index = True)
            
            control_summary.to_csv(f'results/control_{region}_summary{file_suf}.csv',index = False)
    

#%%
def whole_brain_spread():
    control_summary = pd.DataFrame(columns=['subj','group','task','hemi','selec_spread'])
    patient_summary = pd.DataFrame(columns=['subj','group','task','hemi', 'selec_spread'])
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
            hemi_mask = image.get_data(image.load_img(f'{patient_dir}/sub-{ss[1]}/ses-01/anat/sub-{ss[1]}_ses-01_T1w_brain_mask_{p_hemi[ss[0]]}.nii.gz'))
            #load and combine dorsal ROIs  

            mask_size = np.sum(hemi_mask)

            zstat = image.get_data(image.load_img(f'{stat_dir}/stats/zstat1.nii.gz'))
            clust_mask = image.get_data(image.load_img(f'{stat_dir}/cluster_mask_zstat1.nii.gz'))
            clust_mask[clust_mask>0] = 1

            zstat_mask = zstat  * hemi_mask * clust_mask 

            p_spread = np.sum(zstat_mask)

            if file_suf == '_norm':
                p_spread = p_spread / mask_size

            
            
            patient_summary = patient_summary.append(pd.Series([ss[1],'patient',cond[exp[0]],p_hemi[ss[0]], p_spread], index = patient_summary.columns), ignore_index = True)
            
        
        patient_summary.to_csv(f'results/patient_whole_brain_summary{file_suf}.csv',index = False)
    
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
                hemi_mask = image.get_data(image.load_img(f'{patient_dir}/sub-{ss[1]}/ses-01/anat/sub-{ss[1]}_ses-01_T1w_brain_mask_{hemi}.nii.gz'))
                #load and combine dorsal ROIs  
                parcel = combine_rois(ss[1], parcels)

                mask_size = np.sum(hemi_mask)
                
                zstat = image.get_data(image.load_img(f'{stat_dir}/stats/zstat1.nii.gz'))
                clust_mask = image.get_data(image.load_img(f'{stat_dir}/cluster_mask_zstat1.nii.gz'))
                clust_mask[clust_mask>0] = 1

                zstat_mask = zstat  * hemi_mask * clust_mask 
                
                act_val =np.sum(zstat_mask)

                if file_suf == '_norm':
                    act_val = act_val / mask_size
                #if act_val == 0:
                #    act_val = np.nan
                
                control_summary = control_summary.append(pd.Series([ss[1],'control',cond[exp[0]],hemi, act_val], index = control_summary.columns), ignore_index = True)
                
            
            control_summary.to_csv(f'results/control_whole_brain_summary{file_suf}.csv',index = False)
#%%
#pathway_summed_selectivity()
whole_brain_spread()