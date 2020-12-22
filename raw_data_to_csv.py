# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 11:31:01 2020

@author: Aurelien
"""

import os
import scipy.io 
import pandas as pd

def mat_to_df_raw_data () :
    Path = "..\\DataBase\\SEED-VIG\\Raw_Data"
    Path_csv = "..\\DataBase\\SEED-VIG\\EEG_csv"
    dict_df = {}
    
    
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
            
            file_name.replace(".mat",".csv")
        
        else :
            df = pd.read_csv(csv_path, sep=";")
            
        dict_df[file_name.replace(".mat","")]=df
    return dict_df

