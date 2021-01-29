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

import os
import scipy.io
import pandas as pd


def mat_to_df_raw_data():

    path = "..\\DataBase\\SEED-VIG\\Raw_Data"
    path_csv = "..\\DataBase\\SEED-VIG\\EEG_csv"
    
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


def df_5band():

    path = "..\\DataBase\\SEED-VIG\\EEG_Feature_5Bands"
    path_raw_csv = "..\\DataBase\\SEED-VIG\\EEG_csv"
    path_csv = "..\\DataBase\\SEED-VIG\\5Bands_Perclos_Csv"
    path_perclos = "..\\DataBase\\SEED-VIG\\perclos_labels"
    path_json = "..\\DataBase\\SEED-VIG\\EEG_json"

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
        csv_raw_path = path_raw_csv + "\\" + file_csv
        csv_path = path_csv + "\\" + file_csv

        if file_csv not in list_csv:

            df_raw = pd.read_csv(csv_raw_path, sep=";")
            band_headers = list(df_raw)
            waves = ['delta', 'theta', 'alpha', 'beta', 'gamma']

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
                for index, band_data in enumerate(band_headers):
                    for index_waves, wave in enumerate(waves):
                        column_header = band_data+"_"+wave+"_"+signal_analyse_component
                        data_dict[column_header] = []
                        for band_tab in data_signal_analyse_component[index]:
                            data_dict[column_header].append(band_tab[index_waves])

                data_dict['perclos'] = []
                for perclos in data_mat_perclos['perclos']:
                    data_dict['perclos'].append(perclos[0])

            df = pd.DataFrame(data_dict)
            df.to_csv(path_csv + "\\" + file_csv, sep=";", index=False)

            file_json = path_json + "\\" + file_name.replace(".mat", ".json")
            df.to_json(file_json, orient="records")
