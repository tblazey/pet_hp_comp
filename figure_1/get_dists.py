import os
import fastkde
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter

os.chdir(os.path.dirname(__file__))

# Inputs
sigma = 14 / 2 / np.sqrt(np.log(2) * 2) / 3 # 14 mm FWHM with 3mm voxels
hp_imgs = ['pyr', 'lac', 'bic', 'lpr', 'bpr', 'lbr']
pet_imgs = ['fdg', 'ho', 'om', 'oe', 'oc', 'gi']
n_pet = len(pet_imgs)
n_hp = len(hp_imgs)

# Load in mask
mask_hdr = nib.load("../data/MNI152_T1_3mm_brain_mask.nii.gz")
mask_data = mask_hdr.get_fdata().flatten() == 1
n_brain = np.sum(mask_data)

# Make arrays for storing data
hp_dist = np.zeros((n_brain, n_hp))
pet_dist = np.zeros((n_brain, n_pet))
pdf_dist = np.zeros((n_brain, n_hp, n_pet))

# Loop through hp images
for i, hp in enumerate(hp_imgs):

    # Load in hp image
    hp_hdr = nib.load(f"../data/{hp}_dfnd_3mm.nii.gz")
    hp_data = hp_hdr.get_fdata()
    
    # Mask and normalize
    hp_masked = hp_data.flatten()[mask_data]
    if hp in ['lpr', 'bpr', 'lbr']:
        hp_masked -= np.mean(hp_masked)
    else:
        hp_masked /= np.mean(hp_masked)
    hp_dist[:, i] = hp_masked
    
    # Loop through pet mages
    for j, pet in enumerate(pet_imgs):
        
        if i == 0:
    
            # Load in pet image
            pet_hdr = nib.load(f"../data/{pet}_dfnd_3mm.nii.gz")
            pet_data = pet_hdr.get_fdata()
            
            # Smooth pet to make it closer to hp resolution
            pet_smooth = gaussian_filter(pet_data, sigma)
            smooth_path = f"../data/{pet}_dfnd_3mm_smooth.nii.gz"
            nib.Nifti1Image(pet_smooth, pet_hdr.affine).to_filename(smooth_path)
            
            # Mask and normalize
            pet_masked = pet_smooth.flatten()[mask_data]
            if pet == 'gi':
                pet_masked -= np.mean(pet_masked)
            else:
                pet_masked /= np.mean(pet_masked)
            pet_dist[:, j] = pet_masked
        
        # Compute pdf between hp and pdf image
        dist = np.stack((hp_dist[:, i], pet_dist[:, j]), axis=1)
        pdf = fastkde.pdf(
            dist[:, 0], dist[:, 1], var_names = [hp, pet], num_points=2**8 + 1)
        
        # Interpolate pdf so that each voxel gets a value
        pdf_vals = pdf.to_numpy().T
        pdf_i = interpn(
            (pdf.coords[hp].data, pdf.coords[pet].data),
            pdf_vals, dist,
            bounds_error=False,
            method='linear',
            fill_value=None
        )
        pdf_dist[:, i, j] = pdf_i
        
# Save data
np.savez(f'../dists/pet_dist_3mm', pet_dist)
np.savez(f'../dists/hp_dist_3mm', hp_dist)
np.savez(f'../dists/hp_pet_dist_3mm', pdf_dist)


