# -*- coding: utf-8 -*-
"""

Copyright:
Wei-Long Zheng and Bao-Liang Lu
Center for Brain-like Computing and Machine Intelligence, Department of Computer Science and Engineering, Shanghai Jiao Tong University, China
Key Laboratory of Shanghai Education Commission for Intelligent Interaction and Cognitive Engineering, Shanghai Jiao Tong University, China
Brain Science and Technology Research Center, Shanghai Jiao Tong University, China

@author: Aurelien
"""

import os
from traitement_data import mat_to_df_raw_data
from traitement_data import mat_to_df_perclos_label
from traitement_data import df_5band
from graph import plot_band_graph

'''
# On récupère tout les samples des 23 participants
dict_df_raw_data = mat_to_df_raw_data()

# On récupère les noms de fichiers/échantillons
samples_headers = list(dict_df_raw_data)

# On plot toutes les données et on sauvegarde dans des fichiers pdf
for sample in samples_headers :
    df_sample_to_plot = dict_df_raw_data[sample]
    plot_band_graph(df_sample_to_plot, sample)
'''

df_5band()

os.system("pause")
