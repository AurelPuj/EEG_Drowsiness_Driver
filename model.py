# -*- coding: utf-8 -*-
"""

Copyright:
Wei-Long Zheng and Bao-Liang Lu
Center for Brain-like Computing and Machine Intelligence, Department of Computer Science and Engineering, Shanghai Jiao Tong University, China
Key Laboratory of Shanghai Education Commission for Intelligent Interaction and Cognitive Engineering, Shanghai Jiao Tong University, China
Brain Science and Technology Research Center, Shanghai Jiao Tong University, China

@author: Aurelien
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from traitement_data import df_5band
import joblib


def train_ml():

    df_5band()

    file_path = "..\\DataBase\\SEED-VIG\\5Bands_Perclos_Csv\\10_20151125_noon.csv"
    dataset = pd.read_csv(file_path, sep=";")
    print(dataset.describe())

    X = dataset.drop(['perclos'], axis=1)
    y = dataset['perclos']

    print(X)
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(model.score(X_test, y_test))

    joblib.dump(model, 'model.pkl')

    model_columns = list(X.columns)
    joblib.dump(model_columns, 'model_columns.pkl')
