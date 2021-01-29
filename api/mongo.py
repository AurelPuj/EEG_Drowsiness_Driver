# -*- coding: utf-8 -*-
"""
Copyright:
Wei-Long Zheng and Bao-Liang Lu
Center for Brain-like Computing and Machine Intelligence, Department of Computer Science and Engineering, Shanghai Jiao Tong University, China
Key Laboratory of Shanghai Education Commission for Intelligent Interaction and Cognitive Engineering, Shanghai Jiao Tong University, China
Brain Science and Technology Research Center, Shanghai Jiao Tong University, China
@author: Aurelien
"""

from pymongo import MongoClient
import pandas as pd


def init_mongo_db():

    client = MongoClient("localhost", 27017)
    collection = client["eeg_raw_data"]

    file_path = "..\\DataBase\\SEED-VIG\\5Bands_Perclos_Csv\\10_20151125_noon.csv"
    data = pd.read_csv(file_path, sep=";")

    data.reset_index(inplace=True)
    data_dict = data.to_dict("records")  # Insert collection
    collection.insert_many(data_dict)


