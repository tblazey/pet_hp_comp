import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.stats import spearmanr

os.chdir(os.path.dirname(__file__))

# Constants
hp_imgs = ['pyr', 'lac', 'bic', 'lpr', 'bpr', 'lbr']
hp_lbls = [
    r'[1-$^{13}$C]Pyr.',
    r'[1-$^{13}$C]Lac.',
    r'[$^{13}$C]Bic.',
    'LPR',
    'BPR',
    'LBR'
]
n_hp = len(hp_imgs)
hp_scales = [[0.5, 4], [0.5, 2.5], [0.5, 2.5], [-0.75, 0.75], [-0.75, 0.75], [-0.75, 0.75]]

# Load in mask
mask_hdr = nib.load("../data/MNI152_T1_3mm_brain_mask.nii.gz")
mask_data = mask_hdr.get_fdata().flatten() == 1

# Load in anatomical underlay
anat_hdr = nib.load("../data/MNI152_T1_3mm_brain.nii.gz")
anat_data = anat_hdr.get_fdata()
anat_data[anat_data < 600] = np.nan

# Load in samples
hp_dist = np.load('../dists/hp_dist_3mm.npz')['arr_0']
pdf_dist = np.load('../dists/hp_pdf_dist_3mm.npz')['arr_0']

# Normalize pdf to absolute max of 1
pdf_dist = np.abs(pdf_dist) / np.max(pdf_dist, axis=0)

# Create figures
fig, ax = plt.subplots(n_hp, n_hp, figsize=(15, 15), gridspec_kw={'wspace':0.2})

# Loop through hp images
for i in range(n_hp):

    # Loop through pet images    
    for j in range(n_hp):
    
        if i > j:
            ax[i, j].axis('off')
            continue
        
        
        if i == j:
            
            # Get hp data in image format
            hp_img = np.zeros(mask_data.shape)
            hp_img[mask_data] = hp_dist[:, i]
            hp_img[np.logical_not(mask_data)] = np.nan
            hp_img = hp_img.reshape(mask_hdr.shape)

            # Add image to diagonal    
            ax[i, j].matshow(anat_data[30, :, :].T, cmap='gray', origin='lower')
            hp_mat = ax[i, j].matshow(
                hp_img[30, :, :].T,
                cmap='plasma',
                origin='lower',
                alpha=0.75,
                vmin=hp_scales[i][0],
                vmax=hp_scales[i][1]
            )
            
            # Add colorbar 
            cbar = fig.colorbar(
                hp_mat,
                ax=ax[i, j],
                orientation='horizontal',
                fraction=0.05,
                aspect=15,
                ticks=[
                    hp_scales[i][0],
                    (hp_scales[i][1] + hp_scales[i][0]) / 2,
                    hp_scales[i][1]
                ]
            )
            cbar.ax.set_xlabel('WB Norm.', fontweight='bold', fontsize=12)
            
            # Format axis
            ax[i, j].set_ylabel(hp_lbls[i], size=16, weight='bold', labelpad=6)
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
                hp_dist[:, j],
                hp_dist[:, i],
                s=10,
                cmap='gray_r',
                c=pdf_dist[:, i, j],
                #alpha=pdf_dist[:, i, j]
            )
            # Format axis limits/labels
            if hp_imgs[j] in ['lpr', 'bpr', 'lbr']:
                ax[i, j].set_xlim([-0.75, 0.75])
                ax[i, j].set_xticks([-0.5, 0, 0.5])
            else:
                ax[i, j].set_xlim([0, 2.5])
                #ax[i, j].set_xticks(np.arange(0, 4))
            if hp_imgs[i] in ['lpr', 'bpr', 'lbr']:
                ax[i, j].set_ylim([-0.75, 0.75])
                ax[i, j].set_yticks([-0.5, 0, 0.5])
            else:
                ax[i, j].set_ylim([0, 3])
                ax[i, j].set_yticks(np.arange(0, 4))
            
            
            # Add correlation value to plot
            rho = np.round(spearmanr(hp_dist[:, j], hp_dist[:, i]).statistic, 2)
            ax[i, j].annotate(
                rf'$\rho$ = {rho:.2f}',
                [0.1, 0.85],
                fontsize=14,
                fontweight='bold',
                xycoords='axes fraction'
            )
            
        # Add title for top row
        if i == 0:
            ax[i, j].set_title(hp_lbls[j], size=16, weight='bold', pad=10)
              
plt.savefig('./hp_lattice.jpeg', bbox_inches='tight', dpi=300)
plt.close('all')






