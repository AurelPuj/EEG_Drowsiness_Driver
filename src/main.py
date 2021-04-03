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

from model import train_ml, train_dl, train_rf
from filter import process, filter_raw, psd_raw, process_bpci_data,filter_api
import pandas as pd
from pyOpenBCI import OpenBCICyton
import requests
import json
import numpy as np
import keras

menu = input("-------------------Menu-------------------\n1: étude stat \n2: Créer csv complet\n3: Entrainer ML\n"
             "4: Entrainer DL\n5: Créer Raw + Label\n6: Process signal\n7: Filter Raw\n8: psd Raw\n"
             "9: Autoencoder\nChoix :    ")

print(menu)
if menu == '1':
    stat_study("../../Database/SEED-VIG/Dataset_Classification.csv")

if menu == '2':
    df_concat()

if menu == '3':
    train_rf()

if menu == '4':
    train_dl()

if menu == '5':
    add_raw_label()

if menu == '6':
    file_path = "../../Database/SEED-VIG/Dataset_Raw.csv"
    dataset = pd.read_csv(file_path, sep=";")
    process(dataset)

if menu == '7':
    file_path = "../../Database/SEED-VIG/Dataset_Raw.csv"
    dataset = pd.read_csv(file_path, sep=";")
    filter_raw(dataset)

if menu == '8':
    psd_raw(dataset)


def print_raw(sample):
    print(sample.channels_data)


if menu == '9':
    board = OpenBCICyton(port=None, daisy=False)
    board.start_stream(print_raw)

if menu == '10':
    file_path = "../../Database/SEED-VIG/data/eeg.csv"
    dataset = pd.read_csv(file_path, sep=",")
    process_bpci_data(dataset)

if menu =='11':
    with open('../../Database/SEED-VIG/test.json') as json_data:
        data_dict = json.load(json_data)

if menu == '12':

    SCALE_FACTOR_EEG = (4500000)/24/(2**23-1) #uV/count

    data = {}

    data['FT7'] = []
    data['FT8'] = []
    data['T7'] = []
    data['CP1'] = []
    data['CP2'] = []
    data['T8'] = []
    data['O2'] = []
    data['O1'] = []

    def process_predict(sample):

        if len(data['FT7']) < 1000:
            data['FT7'].append(np.array(sample.channels_data[0]) * SCALE_FACTOR_EEG)
            data['FT8'].append(np.array(sample.channels_data[1]) * SCALE_FACTOR_EEG)
            data['T7'].append(np.array(sample.channels_data[2]) * SCALE_FACTOR_EEG)
            data['CP1'].append(np.array(sample.channels_data[3]) * SCALE_FACTOR_EEG)
            data['CP2'].append(np.array(sample.channels_data[4]) * SCALE_FACTOR_EEG)
            data['T8'].append(np.array(sample.channels_data[5]) * SCALE_FACTOR_EEG)
            data['O2'].append(np.array(sample.channels_data[6]) * SCALE_FACTOR_EEG)
            data['O1'].append(np.array(sample.channels_data[7]) * SCALE_FACTOR_EEG)
            if len(data['FT7'])%250 == 0:
                json_data = {
                    'FT7' : data['FT7'][-250:],
                    'FT8': data['FT8'][-250:],
                    'T7': data['T7'][-250:],
                    'CP1': data['CP1'][-250:],
                    'CP2': data['CP2'][-250:],
                    'T8': data['T8'][-250:],
                    'O2': data['O2'][-250:],
                    'O1': data['O1'][-250:],
                }
                requests.post('http://0.0.0.0:5000/compute_psd', json=json_data)
                requests.post('http://0.0.0.0:5000/store_raw', json=json_data)

        elif len(data['FT7']) == 1000:
            requests.post('http://0.0.0.0:5000/predictdl', json=data)

            data['FT7'] = []
            data['FT8'] = []
            data['T7'] = []
            data['CP1'] = []
            data['CP2'] = []
            data['T8'] = []
            data['O2'] = []
            data['O1'] = []


    board = OpenBCICyton(port='/dev/ttyUSB0', daisy=True)
    board.start_stream(process_predict)

if menu == '13':

    SCALE_FACTOR_EEG = 4500000 / 24 / (2 ** 23 - 1)  # uV/count

    data = {}

    data['FT7'] = []
    data['FT8'] = []
    data['T7'] = []
    data['CP1'] = []
    data['CP2'] = []
    data['T8'] = []
    data['O2'] = []
    data['O1'] = []

    deep_model = keras.models.load_model("../api/models/DL_CNNLSTM.h5")
    deep_model.load_weights("../api//models/DL_CNNLSTMweights.h5")


    def process_predict(sample):

        if len(data['FT7']) < 1000:
            data['FT7'].append(np.array(sample.channels_data[0]) * SCALE_FACTOR_EEG)
            data['FT8'].append(np.array(sample.channels_data[1]) * SCALE_FACTOR_EEG)
            data['T7'].append(np.array(sample.channels_data[2]) * SCALE_FACTOR_EEG)
            data['CP1'].append(np.array(sample.channels_data[3]) * SCALE_FACTOR_EEG)
            data['CP2'].append(np.array(sample.channels_data[4]) * SCALE_FACTOR_EEG)
            data['T8'].append(np.array(sample.channels_data[5]) * SCALE_FACTOR_EEG)
            data['O2'].append(np.array(sample.channels_data[6]) * SCALE_FACTOR_EEG)
            data['O1'].append(np.array(sample.channels_data[7]) * SCALE_FACTOR_EEG)


        elif len(data['FT7']) == 1000:
            requests.post('http://0.0.0.0:5000/predict', json=data)

            data['FT7'] = []
            data['FT8'] = []
            data['T7'] = []
            data['CP1'] = []
            data['CP2'] = []
            data['T8'] = []
            data['O2'] = []
            data['O1'] = []



    board = OpenBCICyton(port='/dev/ttyUSB0', daisy=True)
    board.start_stream(process_predict)
