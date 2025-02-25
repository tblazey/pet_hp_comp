import os
from netneurotools import datasets as nntdata
from neuromaps import images, nulls, transforms
from neuromaps.parcellate import Parcellater
import numpy as np
from surfplot import Plot

os.chdir(os.path.dirname(__file__))

# Constants
n_perm = 10000

for mod in ['hp', 'pet']:

    # Get surface parcellation in gifti format for neuromaps
    surf_lbl_path = nntdata.fetch_schaefer2018('fslr32k')['200Parcels7Networks'] 
    surf_gii = images.dlabel_to_gifti(surf_lbl_path)
    surf_parc = Parcellater(surf_gii, 'fsLR')
    surf_parc.parcellation[0].to_filename('/tmp/hemi0.gii')
    surf_parc.parcellation[1].to_filename('/tmp/hemi1.gii')
    
    # Load sample data
    samp_data = np.load(f'../parcs/{mod}_schaefer200_7net.npz')['arr_0']
    
    # Get rotated data
    rotate_data = np.zeros((200, n_perm, samp_data.shape[-1]))
    for i in range(samp_data.shape[-1]):
    
        rotate_data[:, :, i] = nulls.alexander_bloch(
            samp_data[:, i],
            atlas='fsLR',
            density='32k',
            n_perm=n_perm,
            parcellation=['/tmp/hemi0.gii', '/tmp/hemi1.gii'],
            seed=123
    )
    
    # Save data
    np.savez(f'../parcs/{mod}_schaefer200_7net_rotated', rotate_data)