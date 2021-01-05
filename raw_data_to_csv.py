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
import scipy.io 
import pandas as pd

def mat_to_df_raw_data () :
    Path = "..\\DataBase\\SEED-VIG\\Raw_Data"
    Path_csv = "..\\DataBase\\SEED-VIG\\EEG_csv"
    dict_df = {}
    
    # On charge l'ensemble des fichiers .mat existant
    listeFichiers = []
    for (repertoire, sousRepertoires, fichiers) in os.walk(Path):
        listeFichiers.extend(fichiers)
    
    # On charge l'ensemble des fichiers .csv existant 
    listeCsv = []
    for (repertoire, sousRepertoires, fichiers) in os.walk(Path_csv):
        listeCsv.extend(fichiers)
    
    
    for file_name in listeFichiers :
        file_csv = file_name.replace(".mat",".csv")
        csv_path = Path_csv+"\\" + file_csv
        
        if file_csv not in listeCsv :
            # On récupère les données stockées dans le fichier mat 
            data_mat = scipy.io.loadmat(Path + "\\" + file_name)
            
            # On extrait les data de chaque éléctrodes 
            raw_data = data_mat['EEG'][0][0][0] # Data of each band
            
            # On crée un tableau afin de stocker les différents étiquettes des canaux de l'EEG
            band_headers = []
            
            for band in data_mat['EEG'][0][0][1][0] :
                band_headers.append(band[0])
            
            # On crée un dictoinnaire afin d'associer chaque valeur à chaque séquence de données mesurées
            data_dict={}
            
            # On initialise chaque cannaux avec un tableau afin de mesurée celui-ci
            for i in range (0,len(band_headers)):
                data_dict[band_headers[i]] = []
            
            # On rempli les tableau avec les données
            for sample in raw_data:
                for i in range(0,sample.size):
                    data_dict[band_headers[i]].append(sample[i])
            
            # On crée un dataframe dans lequel on stock le dictionnaire -> plus rapide que de crée le data frame direcctment
            df = pd.DataFrame(data_dict)
            df.to_csv(csv_path, sep=";", index=False)
            
            file_name.replace(".mat",".csv")
        
        else :
            # Si le fichier csv existe déjà on charge directement les données depuis celui-ci
            df = pd.read_csv(csv_path, sep=";")
            
        dict_df[file_name.replace(".mat","")]=df
        
    return dict_df

def mat_to_df_perclos_label ():
    
    Path = "..\\DataBase\\SEED-VIG\\EEG_Feature_2Hz\\1_20151124_noon_2.mat"
    data_mat = scipy.io.loadmat(Path)
    print(data_mat.keys())
    
    print(data_mat['psd_movingAve'][0][0])
    print(len(data_mat['psd_movingAve'][0][0]))
    
    
    
    
    