#!/bin/tcsh -f

# Compute KDE between modalities and then within
python3 get_dists.py
python3 within_mod_kde.py

# Make scatterplot matricies for each modality
python3 hp_matrix_plot.py
python3 pet_matrix_plot.py

# Combine figures
python3 comb_fig_1.py

