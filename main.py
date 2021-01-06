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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from traitement_data import mat_to_df_raw_data
from traitement_data import mat_to_df_perclos_label
from traitement_data import df_5band
from graph import plot_band_graph

df_5band()

file_path = "..\\DataBase\\SEED-VIG\\5Bands_Perclos_Csv\\psd_LDS_11_20151024_night.csv"
dataset = pd.read_csv(file_path, sep=";")
print(dataset.describe())

X = dataset.drop(['perclos'], axis=1)
y = dataset['perclos']

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(regressor.score(X_test, y_test))

os.system("pause")
