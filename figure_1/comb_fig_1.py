import os
import matplotlib.pyplot as plt
import matplotlib.image as img

os.chdir(os.path.dirname(__file__))

hp_lat = img.imread('./hp_lattice.jpeg')
pet_lat = img.imread('./pet_lattice.jpeg')

fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'wspace':0.1})

ax[0].imshow(hp_lat)
ax[0].axis('off')
ax[0].annotate(
    'A)', [-0.035, 1.025], fontsize=14, fontweight='bold', xycoords='axes fraction'
)
ax[1].imshow(pet_lat)
ax[1].axis('off')
ax[1].annotate(
    'B)', [-0.05, 1.025], fontsize=14, fontweight='bold', xycoords='axes fraction'
)

plt.savefig('./figure_1.jpeg', bbox_inches='tight', dpi=300)
plt.close('all')
