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

from data_process import df_concat, df_5band, stat_study, add_raw_label
from model import train_ml, train_dl
import keyboard  # using module keyboard
from filter import process
import tensorflow as tf
import pandas as pd
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
'''
menu = input("-------------------Menu-------------------\n1: étude stat \n2: Créer csv complet\n3: Entrainer ML\n"
             "4: Entrainer DL\n5: Créer Raw + Label\nChoix :    ")
print(menu)
if menu == '1':
    stat_study("../../Database/SEED-VIG/Dataset_Classification.csv")
if menu == '2':
    df_concat()
if menu == '3':
    train_ml()
if menu == '4':
    train_dl()
if menu == '5':
    add_raw_label()
'''

file_path = "../../Database/SEED-VIG/Raw_Data_Labelized/1_20151124_noon_2.csv"
dataset = pd.read_csv(file_path, sep=";")

dataset = dataset.drop(['label'], axis=1)
data = dataset[:1599]

process(data)

file_path = "../../Database/SEED-VIG/5Bands_Perclos_Csv/1_20151124_noon_2.csv"
dataset = pd.read_csv(file_path, sep=";")
print(dataset["FT7_delta_psd_LDS"][0])