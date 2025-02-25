import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.stats import spearmanr

os.chdir(os.path.dirname(__file__))

# PET input
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

# HP input
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

# Load in samples
hp_dist = np.load('../dists/hp_dist_3mm.npz')['arr_0']
pet_dist = np.load('../dists/pet_dist_3mm.npz')['arr_0']
pdf_dist = np.abs(np.load('../dists/hp_pet_dist_3mm.npz')['arr_0'])

# Normalize pdf to absolute max of 1
pdf_dist /= np.max(pdf_dist, axis=0)

# Load in mask
mask_hdr = nib.load("../data/MNI152_T1_3mm_brain_mask.nii.gz")
mask_data = mask_hdr.get_fdata().flatten() == 1

# Load in anatomical underlay
anat_hdr = nib.load("../data/MNI152_T1_3mm_brain.nii.gz")
anat_data = anat_hdr.get_fdata()
anat_data[anat_data < 600] = np.nan

# Create figures
fig, ax = plt.subplots(
    n_hp + 2,
    n_pet + 2,
    figsize=(18, 14),
    gridspec_kw={
        'wspace':0.2,
        'hspace':0.2,
        'width_ratios':[1, 0.005, 1, 1, 1, 1, 1, 1],
        'height_ratios':[1, 0.005, 1, 1, 1, 1, 1, 1]
    }
)
ax[0, 0].axis('off')
for i in range(8):
    ax[i, 1].axis('off')
for i in range(8):
    ax[1, i].axis('off')

# Loop through hp images
for i, hp in enumerate(hp_imgs):

    # Get hp data in image format
    hp_img = np.zeros(mask_data.shape)
    hp_img[mask_data] = hp_dist[:, i]
    hp_img[np.logical_not(mask_data)] = np.nan
    hp_img = hp_img.reshape(mask_hdr.shape)
    
    # Add image to diagonal    
    ax[i + 2, 0].matshow(anat_data[30, :, :].T, cmap='gray', origin='lower')
    hp_mat = ax[i + 2, 0].matshow(
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
        ax=ax[i + 2, 0],
        orientation='horizontal',
        fraction=0.05,
        aspect=15,
        ticks=[
            hp_scales[i][0],
            (hp_scales[i][1] + hp_scales[i][0]) / 2,
            hp_scales[i][1]
        ]
    )
    
    # Format axis
    ax[i + 2, 0].set_ylabel(hp_lbls[i], size=16, weight='bold', labelpad=6)
    ax[i + 2, 0].set_xticks([])
    ax[i + 2, 0].set_yticks([])
    ax[i + 2, 0].spines['top'].set_visible(False)
    ax[i + 2, 0].spines['right'].set_visible(False)
    ax[i + 2, 0].spines['bottom'].set_visible(False)
    ax[i + 2, 0].spines['left'].set_visible(False)

    # Loop through pet images    
    for j, pet in enumerate(pet_imgs):
        
        if i == 0:
        
            # Get pet data in image format
            pet_img = np.zeros(mask_data.shape)
            pet_img[mask_data] = pet_dist[:, j]
            pet_img[np.logical_not(mask_data)] = np.nan
            pet_img = pet_img.reshape(mask_hdr.shape)

            # Add image to diagonal    
            ax[0, j + 2].matshow(anat_data[30, :, :].T, cmap='gray', origin='lower')
            pet_mat = ax[0, j + 2].matshow(
                pet_img[30, :, :].T,
                cmap='viridis',
                origin='lower',
                alpha=0.75,
                vmin=pet_scales[j][0],
                vmax=pet_scales[j][1]
            )
            
            # Add colorbar 
            cbar = fig.colorbar(
                pet_mat,
                ax=ax[0, j + 2],
                orientation='horizontal',
                fraction=0.05,
                aspect=15,
                ticks=[
                    pet_scales[j][0],
                    (pet_scales[j][1] + pet_scales[j][0]) / 2,
                    pet_scales[j][1]
                ]
            )
            
            # Format axis
            ax[0, j + 2].set_title(pet_lbls[j], size=16, weight='bold', pad=6)
            ax[0, j + 2].set_xticks([])
            ax[0, j + 2].set_yticks([])
            ax[0, j + 2].spines['top'].set_visible(False)
            ax[0, j + 2].spines['right'].set_visible(False)
            ax[0, j + 2].spines['bottom'].set_visible(False)
            ax[0, j + 2].spines['left'].set_visible(False)
        
        # Add data to plot
        ax[i + 2, j + 2].grid()
        ax[i + 2, j + 2].scatter(
            pet_dist[:, j],
            hp_dist[:, i],
            c=pdf_dist[:, i, j],
            cmap='gray_r',
            s=10
        )
        
        # Set all axes except gi
        if hp in ['lpr', 'bpr', 'lbr']:
            ax[i + 2, j + 2].set_ylim([-0.75, 0.75])
            ax[i + 2, j + 2].set_yticks([-0.5, 0, 0.5])
        else:
            ax[i + 2, j + 2].set_ylim([0, 2.5])
        if pet == 'gi':
            ax[i + 2, j + 2].set_xlim([-0.3, 0.3])
            ax[i + 2, j + 2].set_xticks([-0.2, 0, 0.2])
        elif pet == 'oef':
            ax[i + 2, j + 2].set_xlim([0.8, 1.2])
        else:
            ax[i + 2, j + 2].set_xlim([.25, 1.75])
        
        # Format plots
        if j > 0:
            ax[i + 2, j + 2].set_yticklabels([])
        else:
            if i >= 3:
                label_pad=-4
            else:
                label_pad=4
            ax[i + 2, j + 2].set_ylabel(
                'WB Norm.', fontweight='bold', size=10, labelpad=label_pad
            )
        if i < 5:
            ax[i + 2, j + 2].set_xticklabels([])
        else:
            ax[i + 2, j + 2].set_xlabel('WB Norm.', fontweight='bold', size=10)
            
        # Add correlation value to plot
        rho = np.round(spearmanr(pet_dist[:, j], hp_dist[:, i]).statistic, 2)
        ax[i + 2, j + 2].annotate(
            rf'$\rho$ = {rho:.2f}',
            [0.5, 0.85],
            fontsize=11,
            fontweight='bold',
            xycoords='axes fraction'
        )

    
plt.savefig('./figure_2.jpeg', bbox_inches='tight', dpi=300)
plt.close('all')






