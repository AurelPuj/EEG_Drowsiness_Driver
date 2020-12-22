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
from raw_data_to_csv import mat_to_df_raw_data
from raw_data_to_csv import mat_to_df_perclos_label
from graph import plot_band_graph

mat_to_df_perclos_label()

'''
dict_df_raw_dta = mat_to_df_raw_data();
print(dict_df_raw_dta)
'''

os.system("pause")