"""
Ã‰diteur de Spyder

Ceci est un script temporaire.
"""

import scipy.io 
import pandas as pd
import matplotlib.pyplot as plt
file_name = "..\\DataBase\\SEED-VIG\\Raw_Data\\1_20151124_noon_2.mat"

#On récupère les données stockées dans le fichier mat
mat = scipy.io.loadmat(file_name) 

#On extrait les data de chaque éléctrodes 
raw_data = mat['EEG'][0][0][0] # Data of each band

#On crée un tableau afin de stocker les différents étiquettes des canaux de l'EEG
band_headers = []

for band in mat['EEG'][0][0][1][0] :
    band_headers.append(band[0])

#On crée un dictoinnaire afin d'associer chaque valeur à chaque séquence de données mesurées
data_dict={}

#On initialise chaque cannaux avec un tableau afin de mesurée celui-ci
for i in range (0,len(band_headers)):
    data_dict[band_headers[i]] = []

#On rempli les tableau avec les données
for sample in raw_data:
    for i in range(0,sample.size):
        data_dict[band_headers[i]].append(sample[i])

#On affiche chaque cannal sur un subplot
fig,axs = plt.subplots(6,3)
row = 0
color=['red','darkblue','orange','purple','darkgreen','brown','lightgreen','lightblue','magenta','orangered','salmon','blue']
for column,band in enumerate(data_dict) :
    if column%3 == 0 and column>0 :
        row+=1
    axs[row,column%3].plot(data_dict[band],color=color[column%12])
    axs[row,column%3].set_title(band,color=color[column%12])
border=[0.05,0.95,0.6,0.2]
fig.subplots_adjust(border[0],border[0],border[1],border[1],border[-1],border[-2])
plt.show()


import os
os.system("pause")