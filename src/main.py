# -*- coding: utf-8 -*-
"""

Copyright:
Wei-Long Zheng and Bao-Liang Lu
Center for Brain-like Computing and Machine Intelligence, Department of Computer Science and Engineering,
Shanghai Jiao Tong University, ChinaKey Laboratory of Shanghai Education Commission for Intelligent Interaction
and Cognitive Engineering, Shanghai Jiao Tong University, China Brain Science and Technology Research Center,
Shanghai Jiao Tong University, China

@author: Aurelien
"""

from traitement_data import df_concat, df_5band, stat_study
from model import train_ml, train_dl

#stat_study("..\\DataBase\\SEED-VIG\\Dataset_Classification.csv")
#df_concat()
#train_ml()
train_dl()