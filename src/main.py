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

from data_process import df_concat, df_5band, stat_study, add_raw_label,add_raw_label_regress, mat_to_df_raw_data

from model import train_ml, train_dl, train_rf, train_voting, train_pca, train_lda
from filter import process, filter_raw, psd_raw, process_bpci_data,filter_api
import pandas as pd
from pyOpenBCI import OpenBCICyton
import requests
import json
import numpy as np
import keras

SCALE_FACTOR_EEG = (4500000) / 12 / (2 ** 23 - 1)  # uV/count

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
        if len(data['FT7']) % 125 == 0:
            json_data = {
                'raw': [
                    data['FT7'],
                    data['FT8'],
                    data['T7'],
                    data['CP1'],
                    data['CP2'],
                    data['T8'],
                    data['O2'],
                    data['O1']
                ]
            }
            requests.post('http://0.0.0.0:5000/compute_psd', json=json_data)
            requests.post('http://0.0.0.0:5000/store_raw', json=json_data)

    elif len(data['FT7']) == 1000:
        requests.post('http://0.0.0.0:5000/predictml', json=data)
        data['FT7'] = []
        data['FT8'] = []
        data['T7'] = []
        data['CP1'] = []
        data['CP2'] = []
        data['T8'] = []
        data['O2'] = []
        data['O1'] = []


board = OpenBCICyton(port='/dev/ttyUSB1', daisy=True)
board.start_stream(process_predict)
