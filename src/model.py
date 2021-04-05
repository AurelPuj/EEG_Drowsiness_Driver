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
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, ConvLSTM2D, BatchNormalization
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

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


    data = dataset.drop('label', axis=1)
    print(data.columns)
    label = dataset['label'].round()

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=42)

    data = None
    label = None
    

    data_train = x_train
    data_train['label'] = y_train

    x_train = None
    y_train = None

    sleep_data = data_train[data_train['label'] == 2]

    noise_data = sleep_data + np.random.normal(0, .1, sleep_data.shape)
    noise_data['label'] = 2
    data_train = pd.concat([data_train, noise_data], ignore_index=True)

    awake_data = data_train[data_train['label'] == 0]
    noise_data = awake_data + np.random.normal(0, .1, awake_data.shape)
    noise_data['label'] = 0
    data_train = pd.concat([data_train, noise_data], ignore_index=True)

    x_train = data_train.drop('label', axis=1).to_numpy()
    x_test = x_test.to_numpy()
    y_train = data_train['label']

    onehot = LabelBinarizer()
    onehot.fit(y_train)

    y_train = onehot.transform(y_train)
    y_test = onehot.transform(y_test)

    x_train = x_train.reshape(-1, 4, 4, 250, 8)
    output_train = []

    x_test = x_test.reshape(-1, 4, 4, 250, 8)
    output_test = []

    for i in range(x_train.shape[0]):
        output_train.append(y_train[i * 1000])
    output_train = np.array(output_train)

    for i in range(x_train.shape[0]):
        output_test.append(y_test[i * 1000])
    output_test = np.array(output_test)

    y_train = None
    y_test = None

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    model = keras.models.Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(2,2), activation='relu', padding = "same",  input_shape=(4, 4, 250, 8)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding = "same"))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])

    history = model.fit(x_train, output_train, validation_data=(x_test, output_test), batch_size=32, epochs=100, callbacks=[callback], verbose=1)

    model.save("../api/models/DL_CNNLSTM.h5")
    model.save_weights("../api/models/DL_CNNLSTMweights.h5")

    y_pred = model.predict(x_test, verbose=1)
    y_pred = np.argmax(y_pred, axis=1)
    output_test = np.argmax(output_test, axis=1)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(output_test, y_pred)
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

def train_rf():
    from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix
    import seaborn as sn

    # on prépare les données
    print("Raw data to csv")
    mat_to_df_raw_data()
    print("EEG_5_band to csv")
    df_5band()

    # on charge le dataset du 10_20151125_noon.csv
    file_path = "../../Database/SEED-VIG/dataset.csv"
    dataset = pd.read_csv(file_path, sep=";")
    print(dataset.describe())

    # on sépare les données entre caractériqtique et les données à prédire
    X = dataset.drop(['label'], axis=1)
    y = dataset['label']

    print(X)
    print(y)

    # on sépare le tout en un ensemble d'entrainement et un de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model_name = 'RandomForest'
    param_grid = {
        'n_estimators': [1000]
    }  # Create a based model

    rf = RandomForestClassifier()  # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)

    '''
    'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
    param_grid = {'n_estimators': [50]}
    # Define which metric will be used
    score = 'precision'
    # Create a based model
    rf = RandomForestClassifier()  # Instantiate the grid search model
    # 4)  Train (Fit) the best model with training data
    model = GridSearchCV(estimator=rf, param_grid=param_grid, cv=4,
                                     scoring='%s_macro' % score, verbose=2)'''

    grid_search.fit(X_train, y_train)

    best_grid = grid_search.best_estimator_

    print("  ------------------------------------  ")
    print("BEST Configuration is  ==== ", best_grid)
    print("  ------------------------------------  ")


    # on affiche ensuite l(accuracy et enfin on sauvegarde le modèle entrainé
    print(best_grid.score(X_test, y_test))

    y_pred = best_grid.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(best_grid, '../api/models/'+model_name+'.pkl')

    model_columns = list(X.columns)
    joblib.dump(model_columns, '../api/models/columns.pkl')

    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, range(3), range(3))
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.1)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.show()

    from sklearn import metrics
    metrics.plot_roc_curve(best_grid, X_test, y_test)
    plt.show()

def train_voting():

    from sklearn.ensemble import VotingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from numpy import mean

    # on prépare les données
    print("Raw data to csv")
    mat_to_df_raw_data()
    print("EEG_5_band to csv")
    df_5band()

    # on charge le dataset du 10_20151125_noon.csv
    file_path = "../../Database/SEED-VIG/dataset.csv"
    dataset = pd.read_csv(file_path, sep=";")
    print(dataset.describe())

    # on sépare les données entre caractériqtique et les données à prédire
    X = dataset.drop(['label'], axis=1)
    y = dataset['label']

    print(X)
    print(y)

    # on sépare le tout en un ensemble d'entrainement et un de test

    models = list()
    models.append(('rf1', RandomForestClassifier(n_estimators=100)))
    models.append(('rf2', RandomForestClassifier(n_estimators=250)))
    models.append(('rf3', RandomForestClassifier(n_estimators=500)))

    # define the voting ensemble
    model = VotingClassifier(estimators=models, voting='hard')

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

    print(mean(scores))

def train_pca():

    from numpy import mean
    from numpy import std

    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.decomposition import PCA
    # on prépare les données
    print("Raw data to csv")
    mat_to_df_raw_data()
    print("EEG_5_band to csv")
    df_5band()

    # on charge le dataset du 10_20151125_noon.csv
    file_path = "../../Database/SEED-VIG/dataset.csv"
    dataset = pd.read_csv(file_path, sep=";")
    print(dataset.describe())

    # on sépare les données entre caractériqtique et les données à prédire
    X = dataset.drop(['label'], axis=1)
    y = dataset['label']

    print(X)
    print(y)

    # on sépare le tout en un ensemble d'entrainement et un de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def get_models():
        models = dict()
        for i in range(5, 10):
            steps = [('pca', PCA(n_components=i*10)), ('svc', RandomForestClassifier(n_estimators=100))]
            models[str(i*10)] = Pipeline(steps=steps)
        return models

    def evaluate_model(model, X, y):
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        return scores

    models = get_models()
    # evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, X, y)
        results.append(scores)
        names.append(name)
        print('>%s %.3f' % (name, mean(scores)))

def train_lda():

    from numpy import mean
    from numpy import std

    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.decomposition import PCA
    # on prépare les données

    # on charge le dataset du 10_20151125_noon.csv
    file_path = "../../Database/SEED-VIG/dataset.csv"
    dataset = pd.read_csv(file_path, sep=";")
    print(dataset.describe())

    # on sépare les données entre caractériqtique et les données à prédire
    X = dataset.drop(['label'], axis=1)
    y = dataset['label'].round()

    print(X)
    print(y)
    lda = LDA()
    X = lda.fit_transform(X, y)

    # on sépare le tout en un ensemble d'entrainement et un de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=50)  # Instantiate the grid search model
    rf.fit(X_train, y_train)

    print(rf.score(X_test,y_test))