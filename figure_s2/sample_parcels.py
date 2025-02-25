import os
import nibabel as nib
import numpy as np

os.chdir(os.path.dirname(__file__))

# Define parcellation object 
parc_path = os.path.join(
    '../data/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_3mm.nii.gz'
)
parc_hdr = nib.load(parc_path)
parc_data = parc_hdr.get_fdata().flatten()

# Input
for mod in ['hp', 'pet']:

    # Get unique parcellations
    u_parc = np.unique(parc_data)[1:]   #skip zero
    n_parc = u_parc.shape[0]
    
    if mod not in ['hp']:
        img_pre = ['fdg', 'ho', 'om', 'oe', 'oc', 'gi']
        img_suff = '_smooth'
    else:
        img_pre = ['pyr', 'lac', 'bic', 'lpr', 'bpr', 'lbr']
        img_suff = ''
    
    median_data = np.zeros((n_parc, len(img_pre)))
        
    for pre_idx, pre in enumerate(img_pre):
        
        # Get metabolic data in the same space as atlas
        met_hdr = nib.load(f'../data/{pre}_dfnd_3mm{img_suff}.nii.gz')
        met_data = met_hdr.get_fdata().flatten()
        
        for parc_idx, parc in enumerate(u_parc):
            median_data[parc_idx, pre_idx] = np.median(met_data[parc_data == parc])
    
    np.savez(f'../parcs/{mod}_schaefer200_7net', median_data)
