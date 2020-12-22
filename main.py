"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import os
from raw_data_to_csv import mat_to_df_raw_data
from graph import plot_band_graph

dict_df_raw_dta = mat_to_df_raw_data();
print(dict_df_raw_dta)

os.system("pause")