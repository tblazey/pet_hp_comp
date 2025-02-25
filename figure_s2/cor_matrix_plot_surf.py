import os
from brainspace.datasets import load_parcellation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from neuromaps.datasets import fetch_fslr
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from surfplot import Plot

os.chdir(os.path.dirname(__file__))

# PET input
pet_imgs = ['fdg', 'ho', 'om', 'oe', 'oc', 'gi']
pet_lbls = [
    'CMRglc',
    'CBF',
    r'CMRO$_2$',
    'OEF',
    'CBV',
    'GI',
]
n_pet = len(pet_lbls)
pet_scales = [[0.75, 1.25], [0.75, 1.25], [0.75, 1.25], [0.9, 1.1], [0.5, 2.5], [-0.25, 0.25]]

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
n_hp = len(hp_lbls)
hp_scales = [[0.5, 3], [0.5, 2.5], [0.5, 2.5], [-1, 1], [-1, 1], [-1, 1]]

# Load in samples
hp_dist = np.load(f'../parcs/hp_schaefer200_7net.npz')['arr_0']
pet_dist = np.load(f'../parcs/pet_schaefer200_7net.npz')['arr_0']
rot_dist = np.load(f'../parcs/hp_schaefer200_7net_rotated.npz')['arr_0']

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

# Load in surface parcellation
lh_parc, rh_parc = load_parcellation('schaefer', scale=200)
surfaces = fetch_fslr()
lh, rh = surfaces['inflated']

p_values = np.zeros((n_hp, n_pet))
# Loop through hp images
for i, hp in enumerate(hp_imgs):

    # Load rotated samples
    rot_maps = rot_dist[:, :, i]
    n_rot = rot_maps.shape[1]
    
    # Get array values in surface parcellation 
    lh_vals = np.zeros(lh_parc.shape)
    lh_vals[lh_parc > 0] = hp_dist[lh_parc[lh_parc > 0] - 1, i] 

    # Create plot of lateral left hemisphere
    p = Plot(lh, views=['lateral'], zoom=1.5)
    p.add_layer({'left': lh_vals}, cbar=True, cmap='plasma', color_range=hp_scales[i])
    p.add_layer({'left': lh_parc}, cmap='gray', as_outline=True)

    # Get plot as numpy array
    p_render = p.render()
    p_render._check_offscreen()
    x = p_render.to_numpy(transparent_bg=True, scale=(4, 4))
    p_render.close()
    
    # Add surface plot to image
    mat = ax[i + 2, 0].imshow(
        x, cmap='plasma', vmin=hp_scales[i][0], vmax=hp_scales[i][1]
    )
    ax[i + 2, 0].set_ylabel(hp_lbls[i], size=16, weight='bold', labelpad=24)
    ax[i + 2, 0].set_xticks([])
    ax[i + 2, 0].set_yticks([])
    ax[i + 2, 0].spines['top'].set_visible(False)
    ax[i + 2, 0].spines['right'].set_visible(False)
    ax[i + 2, 0].spines['bottom'].set_visible(False)
    ax[i + 2, 0].spines['left'].set_visible(False)
    
    # Add colorbar for surface plot
    divider = make_axes_locatable(ax[i + 2, 0])
    cax = divider.append_axes('bottom', size='7.5%', pad=0.05)
    cbar = fig.colorbar(
        mat,
        ax=cax,
        orientation='horizontal',
        ticks=[
            hp_scales[i][0],
            (hp_scales[i][1] + hp_scales[i][0]) / 2,
            hp_scales[i][1]
        ],
        aspect=13,
        fraction=0.07
    )
    cax.axis('off')

    # Loop through pet images    
    for j, pet in enumerate(pet_imgs):
        
        if i == 0:
        
            # Get array values in surface parcellation 
            lh_vals = np.zeros(lh_parc.shape)
            lh_vals[lh_parc > 0] = pet_dist[lh_parc[lh_parc > 0] - 1, j] 

            # Make surface plot of pet data
            p = Plot(lh, views=['lateral'], zoom=1.5)
            p.add_layer(
                {'left': lh_vals}, cbar=True, cmap='viridis', color_range=pet_scales[j]
            )
            p.add_layer({'left': lh_parc}, cmap='gray', as_outline=True)

            # Current surface plot to numpy array
            p_render = p.render()
            p_render._check_offscreen()
            x = p_render.to_numpy(transparent_bg=True, scale=(4, 4))
            p_render.close()
            
            
            # Add surface plot to figure
            mat = ax[0, j + 2].imshow(
                x, cmap='viridis', vmin=pet_scales[j][0], vmax=pet_scales[j][1]
            )
            ax[0, j + 2].axis('off')
            ax[0, j + 2].set_title(pet_lbls[j], size=16, weight='bold', pad=6)
            divider = make_axes_locatable(ax[0, j + 2])
            cax = divider.append_axes('bottom', size='7.5%', pad=0.05)
            cbar = fig.colorbar(
                mat,
                ax=cax,
                orientation='horizontal',
                ticks=[
                    pet_scales[j][0],
                    (pet_scales[j][1] + pet_scales[j][0]) / 2,
                    pet_scales[j][1]
                ],
                aspect=13,
                fraction=0.07
            )
            cax.axis('off')

        # Add data to plot
        ax[i + 2, j + 2].grid()
        ax[i + 2, j + 2].scatter(
            pet_dist[:, j],
            hp_dist[:, i],
            c='black',
            s=15
        )
        
        # Custom HP axes
        if hp in ['lpr', 'bpr', 'lbr']:
            ax[i + 2, j + 2].set_ylim([-1.25, 1.25])
            ax[i + 2, j + 2].set_yticks([-1, 0, 1])
        elif hp == 'pyr':
            ax[i + 2, j + 2].set_ylim([0, 3.75])
        else:
            ax[i + 2, j + 2].set_ylim([0, 2.75])
            
        # Custom PET aces
        if pet == 'gi':
            ax[i + 2, j + 2].set_xlim([-0.3, 0.3])
            ax[i + 2, j + 2].set_xticks([-0.2, 0, 0.2])
        elif pet == 'oc':
            ax[i + 2, j + 2].set_xlim([0.25, 3.5])
        elif pet == 'oe':
            ax[i + 2, j + 2].set_xlim([0.8, 1.2])
        else:
            ax[i + 2, j + 2].set_xlim([0.5, 1.5])

        
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
            
        # Compute p-value
        rho = spearmanr(pet_dist[:, j], hp_dist[:, i]).statistic
        rho_dist = np.zeros(n_rot)
        for k in range(n_rot):
            rho_dist[k] = spearmanr(pet_dist[:, j], rot_maps[:, k]).statistic
        perm_p = np.sum(np.abs(rho_dist) > np.abs(rho)) / n_rot
        p_values[i, j] = perm_p
       
        # Add correlation value to plot
        if perm_p < 0.01:
            p_weight = 'bold'
        else:
            p_weight = 'normal'
        ax[i + 2, j + 2].annotate(
            rf'$\rho$ = {rho:.2f}',
            [0.1, 0.85],
            fontsize=11,
            fontweight=p_weight,
            xycoords='axes fraction'
        )
           
plt.savefig('figure_s2.jpeg', dpi=300, bbox_inches='tight')
plt.close('all')

# Save p-values to text
p_df = pd.DataFrame({
    'p_value':p_values.flatten(),
    'hp':np.repeat(hp_imgs, n_pet),
    'pet':np.tile(pet_imgs, n_hp)
})
p_df.to_csv('perm_p_values.csv', index=False)




