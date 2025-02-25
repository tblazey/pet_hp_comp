import os
import fastkde
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.interpolate import interpn

os.chdir(os.path.dirname(__file__))

# Load in samples
for mod in ['pet', 'hp']:

    samp_dist = np.load(f'../dists/{mod}_dist_3mm.npz')['arr_0']
    n_dist = samp_dist.shape[-1]
    
    pdf_dist = np.zeros((samp_dist.shape[0], n_dist, n_dist))
     
    for i in range(n_dist):
    
        for j in range(i, n_dist):
        
            # Compute KDE
            pdf = fastkde.pdf(
                samp_dist[:, i], samp_dist[:, j], var_names = ['x', 'y'], num_points=2**8 + 1
            )
            
            # Interpolate KDE at all input points
            pdf_vals = pdf.to_numpy().T
            pdf_i = interpn(
                (pdf.coords['x'].data, pdf.coords['y'].data),
                pdf_vals,
                np.stack((samp_dist[:, i], samp_dist[:, j]), axis=1),
                bounds_error=False,
                method='linear',
                fill_value=None
            )
            pdf_dist[:, i, j] = pdf_i
            pdf_dist[:, j, i] = pdf_i
    
    np.savez(f'../dists/{mod}_pdf_dist_3mm.npz', pdf_dist)






