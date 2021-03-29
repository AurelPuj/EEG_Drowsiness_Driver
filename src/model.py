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
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, Conv1D, MaxPooling1D, ConvLSTM2D,\
    TimeDistributed, LSTM, Bidirectional, UpSampling2D, BatchNormalization
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN
import pickle


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
    from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix
    import seaborn as sn

    tf.compat.v1.disable_v2_behavior()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    print("Training Deep learning")
    file_path = "../../Database/SEED-VIG/filterRaw.csv"
    dataset = pd.read_csv(file_path, sep=";")
    sleep_data = dataset[dataset['label'] == 2]

    noise_data = sleep_data+np.random.normal(0, .1, sleep_data.shape)
    noise_data['label'] = 2
    dataset = pd.concat([dataset,noise_data], ignore_index=True)

    awake_data = dataset[dataset['label'] == 0]
    noise_data = awake_data + np.random.normal(0, .1, awake_data.shape)
    noise_data['label'] = 0
    dataset = pd.concat([dataset, noise_data], ignore_index=True)

    data = dataset[['FT7', 'FT8', 'T7', 'CP1', 'CP2', 'T8', 'O2', 'O1']]
    print(data.columns)
    data = data.to_numpy()
    label = dataset['label']
    onehot = LabelBinarizer()
    onehot.fit(label)
    label = onehot.transform(label)

    print(data.shape)

    X = data.reshape(-1, 4, 1, 400, 8)
    y = []

    for i in range(X.shape[0]):
        y.append(label[i * 1600])
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    model = keras.models.Sequential()
    model.add(ConvLSTM2D(filters=128, kernel_size=(1,1), activation='relu', padding = "same",  input_shape=(4, 1, 400, 8)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding = "same"))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=100, callbacks=[callback], verbose=1)

    model.save("../api/models/DL_CNNLSTM.h5")
    model.save_weights("../api/models/DL_CNNLSTMweights.h5")


    y_pred = model.predict(x_test, verbose=1)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, range(3), range(3))
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.1)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.xlabel("Predictions")
    plt.ylabel("True labels")

    plt.show()

    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['loss'])
    plt.show()
