import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.stats import spearmanr


os.chdir(os.path.dirname(__file__))
pet_imgs = ['fdg', 'ho', 'om', 'oef', 'oc', 'gi']
pet_lbls = [
    'CMRglc',
    'CBF',
    r'CMRO$_2$',
    'OEF',
    'CBV',
    'GI',
]
n_pet = len(pet_imgs)
pet_scales = [[0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [0.9, 1.1], [0.5, 3], [-0.25, 0.25]]

# Load in mask
mask_hdr = nib.load("../data/MNI152_T1_3mm_brain_mask.nii.gz")
mask_data = mask_hdr.get_fdata().flatten() == 1

# Load in anatomical underlay
anat_hdr = nib.load("../data/MNI152_T1_3mm_brain.nii.gz")
anat_data = anat_hdr.get_fdata()
anat_data[anat_data < 600] = np.nan

# Load in samples
pet_dist = np.load('../dists/pet_dist_3mm.npz')['arr_0']
pdf_dist = np.abs(np.load('../dists/pet_pdf_dist_3mm.npz')['arr_0'])

# Normalize pdf to absolute max of 1
pdf_dist /= np.max(pdf_dist, axis=0)

# Create figures
fig, ax = plt.subplots(n_pet, n_pet, figsize=(15, 15), gridspec_kw={'wspace':0.2})

# Loop through pet images
for i in range(n_pet):

    # Loop through pet images    
    for j in range(n_pet):
    
        if i > j:
            ax[i, j].axis('off')
            continue
        
        
        if i == j:

            # Get pet data in image format
            pet_img = np.zeros(mask_data.shape)
            pet_img[mask_data] = pet_dist[:, i]
            pet_img[np.logical_not(mask_data)] = np.nan
            pet_img = pet_img.reshape(mask_hdr.shape)

            # Add image to diagonal    
            ax[i, j].matshow(anat_data[30, :, :].T, cmap='gray', origin='lower')
            pet_mat = ax[i, j].matshow(
                pet_img[30, :, :].T,
                cmap='viridis',
                origin='lower',
                alpha=0.75,
                vmin=pet_scales[i][0],
                vmax=pet_scales[i][1]
            )
            
            # Add colorbar 
            cbar = fig.colorbar(
                pet_mat,
                ax=ax[i, j],
                orientation='horizontal',
                fraction=0.05,
                aspect=15,
                ticks=[
                    pet_scales[i][0],
                    (pet_scales[i][1] + pet_scales[i][0]) / 2,
                    pet_scales[i][1]
                ]
            )
            cbar.ax.set_xlabel('WB Norm.', fontweight='bold', fontsize=12)
            
            # Format axis
            ax[i, j].set_ylabel(pet_lbls[i], size=16, weight='bold', labelpad=6)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].spines['right'].set_visible(False)
            ax[i, j].spines['bottom'].set_visible(False)
            ax[i, j].spines['left'].set_visible(False)

        else:
        
            # Create scatter plots with points colored by density
            ax[i, j].grid()
            ax[i, j].scatter(
                pet_dist[:, j],
                pet_dist[:, i],
                s=10,
                cmap='gray_r',
                c=pdf_dist[:, i, j],
                #alpha=pdf_dist[:, i, j]
            )
            
            # Format axis limits/labels
            if pet_imgs[j] == 'gi':
                ax[i, j].set_xlim([-0.3, 0.3])
                ax[i, j].set_xticks([-0.2, 0, 0.2])
            elif pet_imgs[j] == 'oef':
                ax[i, j].set_xlim([.8, 1.2])
            else:
                ax[i, j].set_xlim([.25, 1.75])
            if pet_imgs[i] == "oef":
                ax[i, j].set_ylim([0.8, 1.2])
            else:
                ax[i, j].set_ylim([0.25, 1.75])
                 
            # Add correlation value to plot
            rho = np.round(spearmanr(pet_dist[:, j], pet_dist[:, i]).statistic, 2)
            ax[i, j].annotate(
                rf'$\rho$ = {rho:.2f}',
                [0.1, 0.85],
                fontsize=14,
                fontweight='bold',
                xycoords='axes fraction'
            )
            
        # Add title for top row
        if i == 0:
            ax[i, j].set_title(pet_lbls[j], size=16, weight='bold', pad=10)
              
plt.savefig('./pet_lattice.jpeg', bbox_inches='tight', dpi=300)
plt.close('all')






