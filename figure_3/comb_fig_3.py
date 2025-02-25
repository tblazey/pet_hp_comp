import os
import matplotlib.pyplot as plt
import matplotlib.image as img

os.chdir(os.path.dirname(__file__))

pls_maps = img.imread('./pls_weights.jpeg')
pls_points = img.imread('./pls_scatter.jpeg')

fig, ax = plt.subplots(
    1, 2, figsize=(12, 6), width_ratios=[3, 1], gridspec_kw={'wspace':0.05}
)

ax[0].imshow(pls_maps)
ax[0].axis('off')
ax[0].annotate(
    'A)', [-0.035, 1.05], fontsize=14, fontweight='bold', xycoords='axes fraction'
)
ax[0].annotate(
    'B)', [0.3, 1.05], fontsize=14, fontweight='bold', xycoords='axes fraction'
)
ax[1].imshow(pls_points)
ax[1].axis('off')
ax[1].annotate(
    'C)', [-0.05, 1.0], fontsize=14, fontweight='bold', xycoords='axes fraction'
)

plt.savefig('figure_3.jpeg', bbox_inches='tight', dpi=300)
plt.close('all')
