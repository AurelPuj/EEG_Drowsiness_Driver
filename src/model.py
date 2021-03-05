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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from traitement_data import df_5band, mat_to_df_raw_data
import joblib
import keras
from keras.layers import Conv2D, Dense, Flatten
import tensorflow as tf


def train_ml():

    # on prépare les données
    print("Raw data to csv")
    mat_to_df_raw_data()
    print("EEG_5_band to csv")
    df_5band()

    # on charge le dataset du 10_20151125_noon.csv
    file_path = "..\\DataBase\\SEED-VIG\\Dataset_Regression.csv"
    dataset = pd.read_csv(file_path, sep=";")
    print(dataset.describe())

    # on sépare les données entre caractériqtique et les données à prédire
    X = dataset.drop(['perclos'], axis=1)
    y = dataset['perclos']

    print(X)
    print(y)

    # on sépare le tout en un ensemble d'entrainement et un de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model_name = 'LinearRegression'
    # on crée le modèle et on l'entraine
    model = LinearRegression()
    model.fit(X_train, y_train)

    # on affiche ensuite l(accuracy et enfin on sauvegarde le modèle entrainé
    print(model.score(X_test, y_test))

    joblib.dump(model, './api/models/'+model_name+'.pkl')

    model_columns = list(X.columns)
    joblib.dump(model_columns, './api/models/columns.pkl')


def train_dl():
    print("Training Deep learning")
    file_path = "../../Database/SEED-VIG/Raw_Data_Labelized/1_20151124_noon_2.csv"
    dataset = pd.read_csv(file_path, sep=";")

    x = dataset.drop(['label'], axis=1)
    y = dataset['label']

    model = keras.models.Sequential()
    model.add(Conv2D(filters=32, kernel_size=1, input_shape=(dataset.shape[1], dataset.shape[0], 1)))
    model.add(Flatten())
    model.add(Dense(1))
    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    model.fit(x_train, y_train,
              epochs=1,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=['history'], batch_size=8)
