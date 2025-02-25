import os
import fastkde
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.interpolate import interpn
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA, PLSCanonical
from sklearn.preprocessing import scale

def compute_density(x, y, norm=True):
     
     # Compute KDE
    pdf = fastkde.pdf(x, y, var_names = ['x', 'y'], num_points=2**8 + 1)
        
    # Interpolate KDE at all input points
    pdf_vals = pdf.to_numpy().T
    pdf_i = interpn(
        (pdf.coords['x'].data, pdf.coords['y'].data),
        pdf_vals,
        np.stack((x, y), axis=1),
        bounds_error=False,
        method='linear',
        fill_value=None
    )
    
    if norm is True:
        pdf_i = np.abs(pdf_i) / np.max(pdf_i)
    
    return pdf_i
    
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
pet_scales = [[0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [0.75, 1.25], [0.5, 3], [-0.25, 0.25]]

# HP input
hp_imgs = ['pyr', 'lac', 'bic', 'lac_pyr_resid', 'bic_pyr_resid', 'resid']
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
pet_dist = np.load('../dists/pet_dist_3mm.npz')['arr_0']

# Mean = 0, SD = 1
hp_s = scale(hp_dist)
pet_s = scale(pet_dist)

# Fit canonical pls
n_comp = 2
pls = PLSCanonical(n_components=n_comp)
pls.fit(hp_s, pet_s)

# Get scores and density  
hp_var, pet_var = pls.transform(hp_s, pet_s)
pdf_var_1 = compute_density(pet_var[:, 0], hp_var[:, 0]) 
pdf_var_2 = compute_density(pet_var[:, 1], hp_var[:, 1])

# Create figure grid
fig, ax = plt.subplots(
    2,
    5,
    gridspec_kw={
        'width_ratios':[0.5, 0.2, 1, 1, 0.1],
        'height_ratios':[1, 1],
        'wspace':0.25
    },
    figsize=(14, 8)
)

# Weights for HP
hp_weights = pls.x_rotations_
ax[0, 0].matshow(hp_weights, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
for (i, j), value in np.ndenumerate(hp_weights):
    ax[0, 0].text(
        j, i, f'{np.round(value, 2)}', ha='center', va='center', color='black', fontsize=13
    )
ax[0, 0].set_yticks(np.arange(n_hp), labels=hp_lbls)
comp_labels = [f'Comp. {i + 1}' for i in range(n_comp)]
ax[0, 0].set_xticks(np.arange(n_comp), labels=comp_labels)
ax[0, 0].xaxis.set_ticks_position('bottom')
ax[0, 0].set_xticks(np.arange(-.5, n_comp, 1), minor=True)
ax[0, 0].set_yticks(np.arange(-.5, n_hp, 1), minor=True)
ax[0, 0].grid(which='minor', color='black', linewidth=2)
ax[0, 0].tick_params(which='minor', length=0)
ax[0, 0].set_title('Weights', fontweight='bold', fontsize=16)

# Can. coords for hp
cc_var_dim = list(anat_hdr.shape) + [n_comp]
hp_var_img = np.zeros(cc_var_dim).reshape(-1, n_comp)
hp_var_img[mask_data, :] = hp_var
hp_var_img /= np.max(np.abs(hp_var_img), axis=0)
hp_var_img[np.logical_not(mask_data), :] = np.nan
hp_var_img = hp_var_img.reshape(cc_var_dim)

# HP coord #1
ax[0, 2].matshow(anat_data[32, :, :].T, cmap='gray', origin='lower')
ax[0, 2].matshow(
    hp_var_img[32, :, :, 0].T, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower', alpha=0.75
)
ax[0, 2].set_title('HP Score #1', fontweight='bold', fontsize=16, y=0.975)
ax[0, 2].axis('off')

# HP coord #2
ax[0, 3].matshow(anat_data[32, :, :].T, cmap='gray', origin='lower')
ax[0, 3].matshow(
    hp_var_img[32, :, :, 1].T, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower', alpha=0.75
)
ax[0, 3].set_title('HP Score #2', fontweight='bold', fontsize=16, y=0.975)
ax[0, 3].axis('off')

# Pet weights
pet_weights = pls.y_rotations_
load_mat = ax[1, 0].matshow(pet_weights, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
for (i, j), value in np.ndenumerate(pet_weights):
    ax[1, 0].text(
        j, i, f'{np.round(value, 2)}', ha='center', va='center', color='black', fontsize=13
    )
ax[1, 0].set_yticks(np.arange(n_pet), labels=pet_lbls)
ax[1, 0].set_xticks(np.arange(n_comp), labels=comp_labels)
ax[1, 0].xaxis.set_ticks_position('bottom')
ax[1, 0].set_xticks(np.arange(-.5, n_comp, 1), minor=True)
ax[1, 0].set_yticks(np.arange(-.5, n_pet, 1), minor=True)
ax[1, 0].grid(which='minor', color='black', linewidth=2)
ax[1, 0].tick_params(which='minor', length=0)

# Add loading colorbar
gs = ax[0, 1].get_gridspec()
load_cb_ax = fig.add_subplot(gs[0:2, 1])
load_cb_ax.axis('off')
load_cb = fig.colorbar(
    load_mat,
    ax=load_cb_ax,
    orientation='vertical',
    location='left',
    fraction=0.75/2,
    aspect=10,
    ticks=[-1, 0, 1]
    
)
load_cb.ax.yaxis.set_label_position("right")
load_cb.ax.yaxis.set_ticks_position("right")
load_cb.set_label(label='Weights', weight='bold', size=10)
ax[0, 1].axis('off')
ax[1, 1].axis('off')

# Can. coords for pet
pet_var_img = np.zeros(cc_var_dim).reshape(-1, n_comp)
pet_var_img[mask_data, :] = pet_var
pet_var_img /= np.max(np.abs(pet_var_img), axis=0)
pet_var_img[np.logical_not(mask_data), :] = np.nan
pet_var_img = pet_var_img.reshape(cc_var_dim)

# Pet coord #1
ax[1, 2].matshow(anat_data[32, :, :].T, cmap='gray', origin='lower')
ax[1, 2].matshow(
    pet_var_img[32, :, :, 0].T, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower', alpha=0.75
)
ax[1, 2].set_title('PET Score #1', fontweight='bold', fontsize=16, y=0.975)
ax[1, 2].axis('off')

# Pet coord #2
ax[1, 3].matshow(anat_data[32, :, :].T, cmap='gray', origin='lower')
var_mat = ax[1, 3].matshow(
    pet_var_img[32, :, :, 1].T, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower', alpha=0.75
)
ax[1, 3].set_title('PET Score #2', fontweight='bold', fontsize=16, y=0.975)
ax[1, 3].axis('off')

# Add can. variable colorbar
var_cb_ax = fig.add_subplot(gs[0:2, 4])
var_cb_ax.axis('off')
var_cb = fig.colorbar(
    var_mat,
    ax=var_cb_ax,
    orientation='vertical',
    fraction=0.75,
    aspect=10,
    ticks=[-1, 0, 1]
    
)
var_cb.set_label(label='Norm. Scores', weight='bold', size=10)
ax[0, 4].axis('off')
ax[1, 4].axis('off')

plt.savefig(f'./pls_weights.jpeg', dpi=300, bbox_inches='tight')
plt.close('all')

# Can. var grid
fig_2, ax_2 = plt.subplots(2, 1, gridspec_kw={'hspace':0.4}, figsize=(4, 8))

# First variable
ax_2[0].grid()
ax_2[0].scatter(
    pet_var[:, 0],
    hp_var[:, 0],
    c=pdf_var_1,
    cmap='gray_r',
    s=10
)
ax_2[0].set_ylabel('HP Score', fontweight='bold', fontsize=14)
ax_2[0].set_xlabel('PET Score', fontweight='bold', fontsize=14)
ax_2[0].set_title('PLS Score #1', fontweight='bold', fontsize=16)
ax_2[0].set_xlim([-6, 6])   
ax_2[0].set_xticks(np.arange(-6, 8, 2)) 
ax_2[0].set_yticks(np.arange(-6, 8, 2)) 

rho_1 = np.round(spearmanr(hp_var[:, 0], pet_var[:, 0]).statistic, 2)
ax_2[0].annotate(
    rf'$\rho$ = {rho_1:.2f}', 
    [0.6, 0.1],
    fontsize=14,
    fontweight='bold',
    xycoords='axes fraction'
)

ax_2[1].grid()
ax_2[1].scatter(
    pet_var[:, 1],
    hp_var[:, 1],
    c=pdf_var_2,
    cmap='gray_r',
    s=10
)
ax_2[1].set_xlabel('PET Score', fontweight='bold', fontsize=14)
ax_2[1].set_ylabel('HP Score', fontweight='bold', fontsize=14)
ax_2[1].set_title('PLS Score #2', fontweight='bold', fontsize=16)
ax_2[1].set_xlim([-2, 2])   
ax_2[1].set_ylim([-2, 2])
ax_2[1].set_xticks(np.arange(-2, 3, 1)) 
ax_2[1].set_yticks(np.arange(-2, 3, 1))  

rho_2 = np.round(spearmanr(hp_var[:, 1], pet_var[:, 1]).statistic, 2)
ax_2[1].annotate(
    rf'$\rho$ = {rho_2:.2f}',
    [0.6, 0.1],
    fontsize=14,
    fontweight='bold',
    xycoords='axes fraction'
)

plt.savefig(f'./pls_scatter.jpeg', dpi=300, bbox_inches='tight')
plt.close('all')
