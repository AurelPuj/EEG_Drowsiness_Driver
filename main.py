"""
Ã‰diteur de Spyder

Ceci est un script temporaire.
"""
import os
import scipy.io 
import pandas as pd
import matplotlib.pyplot as plt
Path = "..\\DataBase\\SEED-VIG\\Raw_Data"
Path_csv = "..\\DataBase\\SEED-VIG\\EEG_csv"

listeFichiers = []
for (repertoire, sousRepertoires, fichiers) in os.walk(Path):
    listeFichiers.extend(fichiers)
    
listeCsv = []
for (repertoire, sousRepertoires, fichiers) in os.walk(Path_csv):
    listeCsv.extend(fichiers)


for file_name in listeFichiers :
    file_csv = file_name.replace(".mat",".csv")
    csv_path = Path_csv+"\\" + file_csv
    
    if file_csv not in listeCsv :
        #On récupère les données stockées dans le fichier mat
        data_mat = scipy.io.loadmat(Path + "\\" + file_name)
        
        #On extrait les data de chaque éléctrodes 
        raw_data = data_mat['EEG'][0][0][0] # Data of each band
        
        #On crée un tableau afin de stocker les différents étiquettes des canaux de l'EEG
        band_headers = []
        
        for band in data_mat['EEG'][0][0][1][0] :
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
        
        #On crée un dataframe dans lequel on stock le dictionnaire -> plus rapide que de crée le data frame direcctment
        df = pd.DataFrame(data_dict)
        df.to_csv(csv_path, sep=";", index=False)
        print(df)
    else :
        df = pd.read_csv(csv_path, sep=";")
        print(df)

'''
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
'''

os.system("pause")

