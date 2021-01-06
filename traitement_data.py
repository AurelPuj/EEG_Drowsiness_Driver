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


def mat_to_df_raw_data():

    path = "..\\DataBase\\SEED-VIG\\Raw_Data"
    path_csv = "..\\DataBase\\SEED-VIG\\EEG_csv"
    dict_df = {}
    
    # On charge l'ensemble des fichiers .mat existant
    list_files = []
    for (repertoire, sousRepertoires, file) in os.walk(path):
        list_files.extend(file)
    
    # On charge l'ensemble des fichiers .csv existant 
    list_csv = []
    for (repertoire, sousRepertoires, file) in os.walk(path_csv):
        list_csv.extend(file)

    for file_name in list_files:
        file_csv = file_name.replace(".mat", ".csv")
        csv_path = path_csv+"\\" + file_csv
        
        if file_csv not in list_csv:
            # On récupère les données stockées dans le fichier mat 
            data_mat = scipy.io.loadmat(path + "\\" + file_name)
            
            # On extrait les data de chaque éléctrodes 
            raw_data = data_mat['EEG'][0][0][0] # Data of each band
            
            # On crée un tableau afin de stocker les différents étiquettes des canaux de l'EEG
            band_headers = []
            
            for band in data_mat['EEG'][0][0][1][0]:
                band_headers.append(band[0])
            
            # On crée un dictoinnaire afin d'associer chaque valeur à chaque séquence de données mesurées
            data_dict = {}
            
            # On initialise chaque cannaux avec un tableau afin de mesurée celui-ci
            for i in range(0, len(band_headers)):
                data_dict[band_headers[i]] = []
            
            # On remplie les tableau avec les données
            for sample in raw_data:
                for i in range(0, sample.size):
                    data_dict[band_headers[i]].append(sample[i])

            # On crée un dataframe dans lequel on stock le dictionnaire -> plus rapide
            df = pd.DataFrame(data_dict)
            df.to_csv(csv_path, sep=";", index=False)
            
            file_name.replace(".mat", ".csv")
        
        else:
            # Si le fichier csv existe déjà on charge directement les données depuis celui-ci
            df = pd.read_csv(csv_path, sep=";")
            print(file_name)
            
        dict_df[file_name.replace(".mat", "")] = df


def df_5band():
    path = "..\\DataBase\\SEED-VIG\\EEG_Feature_5Bands"
    path_raw_csv = "..\\DataBase\\SEED-VIG\\EEG_csv"
    path_csv = "..\\DataBase\\SEED-VIG\\5Bands_Perclos_Csv"
    path_perclos = "..\\DataBase\\SEED-VIG\\perclos_labels"
    dict_df = {}

    # On charge l'ensemble des fichiers .mat existant
    list_files = []
    for (repertoire, sousRepertoires, file) in os.walk(path):
        list_files.extend(file)

    # On charge l'ensemble des fichiers .csv existant
    list_csv = []
    for (repertoire, sousRepertoires, file) in os.walk(path_csv):
        list_csv.extend(file)

    for file_name in list_files[0:2]:
        file_csv = file_name.replace(".mat", ".csv")
        csv_raw_path = path_raw_csv + "\\" + file_csv
        csv_path = path_csv + "\\" + file_csv

        if file_csv not in list_csv:

            # Si le fichier csv existe déjà on charge directement les données depuis celui-ci
            df_raw = pd.read_csv(csv_raw_path, sep=";")
            band_headers = list(df_raw)

            # On récupère les données stockées dans le fichier mat
            data_mat = scipy.io.loadmat(path + "\\" + file_name)
            data_mat_perclos = scipy.io.loadmat(path_perclos + "\\" + file_name)

            header_signal_analyse_component = [
                'psd_movingAve',
                'psd_LDS',
                'de_movingAve',
                'de_LDS'
            ]
            data_dict = {}

            for signal_analyse_component in header_signal_analyse_component:
                data_signal_analyse_component = data_mat[signal_analyse_component]
                data_dict[signal_analyse_component] = {}
                for index, band_data in enumerate(band_headers):

                    data_dict[signal_analyse_component][band_data] = []
                    for band_tab in data_signal_analyse_component[index] :
                        data_dict[signal_analyse_component][band_data].append(band_tab)

                data_dict[signal_analyse_component]['perclos'] = []
                for perclos in data_mat_perclos['perclos']:
                    data_dict[signal_analyse_component]['perclos'].append(perclos)

                # On crée un dataframe dans lequel on stock le dictionnaire -> plus rapide
                df = pd.DataFrame(data_dict[signal_analyse_component])
                df.to_csv(path_csv + "\\" + signal_analyse_component + "_" + file_csv, sep=";", index=False)

                data_dict[signal_analyse_component] = [pd.DataFrame(data_dict[signal_analyse_component])]

            file_name.replace(".mat", ".csv")
            df = pd.read_csv((path_csv + "\\psd_LDS_11_20151024_night.csv"), sep=";")
            print(df['FT7'][0])
            os.system("pause")

        else:
            # Si le fichier csv existe déjà on charge directement les données depuis celui-ci

            dict_df[file_name.replace(".mat", "")] = df


def mat_to_df_perclos_label():

    path = "..\\DataBase\\SEED-VIG\\perclos_labels\\1_20151124_noon_2.mat"
    data_mat = scipy.io.loadmat(path)
    print(data_mat.keys())

    print(data_mat['psd_movingAve'].shape)
