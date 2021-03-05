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
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
import seaborn as sns


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
            print(data_dict['perclos'])
            df = pd.DataFrame(data_dict)
            df.to_csv(path_csv + "\\" + file_csv, sep=";", index=False)

            file_json = path_json + "\\" + file_name.replace(".mat", ".json")
            df.to_json(file_json, orient="records")


def df_concat():

    path_raw_csv = "..\\DataBase\\SEED-VIG\\EEG_csv"
    path_csv = "..\\DataBase\\SEED-VIG\\5Bands_Perclos_Csv"
    path = "..\\DataBase\\SEED-VIG"
    df_total = None

    # On charge l'ensemble des fichiers .csv existant
    list_csv = []
    for (repertoire, sousRepertoires, file) in os.walk(path_csv):
        list_csv.extend(file)

    list_raw_csv = []
    for (repertoire, sousRepertoires, file) in os.walk(path_raw_csv):
        list_raw_csv.extend(file)

    list_files = []
    for (repertoire, sousRepertoires, file) in os.walk(path):
        list_files.extend(file)

    if "Dataset_Regression.csv" not in list_files or "Dataset_Classification.csv" not in list_files:

        for file_name in list_csv:

            csv_path = path_csv + "\\" + file_name
            df_5band = pd.read_csv(csv_path, sep=";")

            if df_total is None:
                column_header = df_5band.columns
                df_total = pd.DataFrame(columns=column_header)

            df_total = pd.concat([df_total, df_5band], sort=False)

        df_total.to_csv("..\\DataBase\\SEED-VIG\\Dataset_Regression.csv", sep=";", index=False)

        for i, perclos in enumerate(df_total['perclos']):
            if perclos < 0.3:
                df_total['perclos'][i] = int(0)
            elif perclos < 0.7:
                df_total['perclos'][i] = int(1)
            else :
                df_total['perclos'][i] = int(2)

        df_total.to_csv("..\\DataBase\\SEED-VIG\\Dataset_Classification.csv", sep=";", index=False)

    if "Dataset_Raw.csv" not in list_files:
        df_raw_total = None
        for file_name in list_raw_csv:

            file_raw_path = path_raw_csv + "\\" + file_name
            df_raw = pd.read_csv(file_raw_path, sep=";")

            if df_raw_total is None:
                column_raw_header = df_raw.columns
                df_raw_total = pd.DataFrame(columns=column_raw_header)

            df_total = pd.concat([df_total, df_raw], sort=False)
        df_total.to_csv("..\\DataBase\\SEED-VIG\\Dataset_Raw.csv", sep=";", index=False)



def stat_study(file):

    dataset = pd.read_csv(file, sep=";")
    #path = ".\\stat"

    # On charge l'ensemble des fichiers .csv existant
    list_file = []
    for (repertoire, sousRepertoires, file) in os.walk(path):
        list_file.extend(file)

    if "numpy_stat.txt" in list_file:
        os.remove(path+"\\numpy_stat.txt")
        print("Stat deleted !")

    file_object = open(path + '\\numpy_stat.txt', 'w')
    print("\n------CALCUL DE LA MOYENNE DE CHAQUE COLONNE-----\n")
    file_object.write("\n------CALCUL DE LA MOYENNE DE CHAQUE COLONNE-----\n")
    for col in dataset.columns:
        file_object.write("\n-------------" + col + "-------------")
        file_object.write("\nMoyenne : " + str(round(np.mean(dataset[col]), 2)))
        file_object.write("\nMediane : " + str(round(np.median(dataset[col]), 2)))
        file_object.write("\nQ1 : " + str(round(np.percentile(dataset[col], 25))))
        file_object.write("\nQ3  : " + str(round(np.percentile(dataset[col], 75))))
        file_object.write("\nVariance : " + str(round(np.var(dataset[col]), 2)))
        file_object.write("\nEcart type : " + str(round(np.std(dataset[col]), 2)) + "\n\n")
        #sns.catplot(data=dataset, x='perclos', y=col, kind="box")
        #plt.show()
    file_object.close()
    sns.displot(dataset['perclos'])
    plt.savefig("Label Distribution.png")


def add_raw_label():

    path_raw_csv = "../../Database/SEED-VIG/EEG_csv"
    path_5band_csv = "../../Database/SEED-VIG/5Bands_Perclos_Csv"
    path_raw_label = "../../Database/SEED-VIG/Raw_Data_Labelized/"

    # On charge l'ensemble des fichiers .csv existant
    list_raw_csv = []
    for (repertoire, sousRepertoires, file) in os.walk(path_raw_csv):
        list_raw_csv.extend(file)

    list_raw_label = []
    for (repertoire, sousRepertoires, file) in os.walk(path_raw_label):
        list_raw_label.extend(file)

    for file_name in list_raw_csv:

        if file_name not in list_raw_label:
            df_raw = pd.read_csv((path_raw_csv+"/"+file_name), sep=";")
            df_band = pd.read_csv((path_5band_csv+"/"+file_name), sep=";")

            label_columns = []

            for i, perclos in enumerate(df_band['perclos']):
                for j in range(0, 1600):
                    if perclos < 0.3:
                        label_columns.append(1)
                    elif 0.3 < perclos < 0.7:
                        label_columns.append(2)
                    elif perclos > 0.7:
                        label_columns.append(3)
            df_raw = df_raw.assign(label=label_columns)
            df_raw.to_csv(path_raw_label+file_name, sep=";", index=False)
            print("{} created !".format(file_name))



