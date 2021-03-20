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
from data_process import df_5band, mat_to_df_raw_data
import joblib
import keras
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, Conv1D, MaxPooling1D
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K

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

def create_model(optimizer='adam'):

    K.clear_session()

    model = keras.models.Sequential()
    model.add(Conv1D(filters=32, kernel_size=1, input_shape=(1, 170), activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.7))

    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.7))

    model.add(Conv1D(filters=128, kernel_size=1, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.7))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))

    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

def train_dl():

    tf.config.experimental.set_memory_growth

    print("Training Deep learning")
    file_path = "../../Database/SEED-VIG/psdDeRaw.csv"
    dataset = pd.read_csv(file_path, sep=";")

    data = dataset.drop(['label'], axis=1).to_numpy()
    label = dataset['label'].to_numpy()

    onehot = LabelBinarizer()
    onehot.fit(label)
    y = onehot.transform(label)

    print(data.shape)
    print(y)

    X = data.reshape(20355, 1, 170)
    y = np.array(y)

    print(X.shape)
    print(y.shape)

    #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = keras.models.Sequential()
    model.add(Conv1D(filters=32, kernel_size=1, input_shape=(1, 170), activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.7))

    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.7))

    model.add(Conv1D(filters=128, kernel_size=1, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.7))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))

    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    history = model.fit(X, y, validation_split=0.3, epochs=30, verbose=1)
    plt.plot(history.history['val_accuracy'])
    plt.show()
    plt.figure()
    plt.plot(history.history['val_loss'])
    plt.show()