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
import numpy as np
import sklearn.preprocessing

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

    data = dataset.drop(['label'], axis=1).to_numpy()
    label = dataset['label'].values.tolist()
    y = []
    for i in range(885):
        y.append(label[i*1600])

    X = data.reshape(885,1600,17,1)
    y = np.array(y)

    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(max(y) + 1))
    y = label_binarizer.transform(y)

    print(X.shape)
    print(y.shape)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = keras.models.Sequential()
    model.add(Conv2D(filters=32, kernel_size=1, input_shape=(1600,17,1)))
    model.add(Flatten())
    model.add(Dense())
    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])

    model.fit(X, y,
              epochs=1,
              verbose=1,
              validation_data=(x_test, y_test)
              ,metric=['accuracy'])
    history = model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10, verbose=2)