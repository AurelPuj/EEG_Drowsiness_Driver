# -*- coding: utf-8 -*-
"""
Copyright:
Wei-Long Zheng and Bao-Liang Lu
Center for Brain-like Computing and Machine Intelligence, Department of Computer Science and Engineering, Shanghai Jiao Tong University, China
Key Laboratory of Shanghai Education Commission for Intelligent Interaction and Cognitive Engineering, Shanghai Jiao Tong University, China
Brain Science and Technology Research Center, Shanghai Jiao Tong University, China
@author: Aurelien
"""

import os
from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_pymongo import PyMongo
import pickle
import sys
import keras
from filter import filter_api, bandpower, filter_band, filter_low, process
from flask import Flask, render_template
import numpy as np

# Create API and load ML Algo
app = Flask("MyEEG")
print("lancement de l'api")

# Initialise the database

app.config["MONGO_URI"] = 'mongodb://' + os.environ['MONGODB_USERNAME'] + ':' + os.environ['MONGODB_PASSWORD'] + '@' \
                          + os.environ['MONGODB_HOSTNAME'] + ':27017/' + os.environ['MONGODB_DATABASE']
mongo = PyMongo(app)
db = mongo.db

path_model = "models/"
list_model = []
for (repertoire, sousRepertoires, file) in os.walk(path_model):
    list_model.extend(file)
db.model.drop()
db.stream.drop()
db.raw.drop()
db.psd.drop()

db.stream.insert_one({"data": {}, 'state': 0})
db.raw.insert_one({"raw": [[], [], [], [], [], [], [], []]})
db.psd.insert_one({'psd': [[], [], [], [], [], [], [], []]})

ml_model = joblib.load(path_model+"RandomForest.pkl")

deep_model = keras.models.load_model(path_model+"DL_CNNLSTM.h5")
deep_model.load_weights(path_model+"DL_CNNLSTMweights.h5")


@app.route('/predict', methods=['POST'])  # Your API endpoint URL would consist /predict
def predict():

    json = request.json
    query = pd.get_dummies(pd.DataFrame(json))

    # On charge les columns pour vérfier celle de la requêtes
    json_data = {}
    data = db.model.find({'name': 'columns'})
    for i in data:
        json_data = i
    model_columns = json_data['columns']

    query = query.reindex(columns=model_columns, fill_value=0)


    # On charge le modèle
    json_data = {}
    data = db.model.find({'name': 'RandomForest'})
    for i in data:
        json_data = i
    pikled_model = json_data['RandomForest']
    model_loaded = pickle.loads(pikled_model)

    prediction = list(model_loaded.predict(query))
    return jsonify({'prediction': prediction})

@app.route('/predictdl', methods=['POST'])  # Your API endpoint URL would consist /predict
def predictdl():

    db.stream.drop()
    json = request.json
    df = pd.get_dummies(pd.DataFrame(json))
    df = filter_api(df)
    df = df.to_numpy().reshape(-1, 8, 1, 125, 8)
    prediction = deep_model.predict(df)
    max_prediction = max(enumerate(prediction[0]), key=(lambda x: x[1]))

    db.stream.insert_one({"data": json, 'state': max_prediction[0]})

    return jsonify(1)

@app.route('/predictml', methods=['POST'])  # Your API endpoint URL would consist /predict
def predictml():

    db.stream.drop()
    json = request.json
    df = pd.get_dummies(pd.DataFrame(json))
    df = process(df)
    prediction = ml_model.predict(df).tolist()

    db.stream.insert_one({"data": json, 'state': prediction[0]})

    return jsonify(1)

@app.route('/store_raw', methods=['POST'])  # Your API endpoint URL would consist /predict
def store_raw():
    json = request.json['raw']
    raw = []
    db.raw.drop()

    for i, data in enumerate(json):
        signal = filter_band(data, 250, 1, 50).tolist()
        raw.append(signal)

    db.raw.insert_one({'raw' : raw})

    return jsonify(1)


@app.route('/getraw', methods=['GET'])  # Your API endpoint URL would consist /predict
def getraw():
    _dict = db.raw.find()
    raw = None

    for data in _dict :
        raw = data['raw']

    if raw != None:
        return jsonify(raw)
    else:
        return jsonify(3)


@app.route('/compute_psd', methods=['POST'])  # Your API endpoint URL would consist /predict
def compute_psd():
    json = request.json['raw']
    db.psd.drop()
    psd = []

    for i, data in enumerate(json):
        psd.append([])
        signal = np.array(filter_band(data, 250, 0.5, 31))

        psd[i].append(bandpower(signal, [0.5, 4], 'welch', None, relative=True))
        psd[i].append(bandpower(signal, [4, 8], 'welch', None, relative=True))
        psd[i].append(bandpower(signal, [8, 14], 'welch', None, relative=True))
        psd[i].append(bandpower(signal, [14, 31], 'welch', None, relative=True))

    db.psd.insert_one({'psd': psd})

    return jsonify(1)


@app.route('/getstate', methods=['GET'])  # Your API endpoint URL would consist /predict
def getstate():
    _dict = db.stream.find()
    state = None

    for data in _dict :
        state = data['state']

    if state != None:
        if state == 0:
            message = ["Awake", "bg-success", str(state+1)]
        if state == 1:
            message = ["Falling in sleep", "bg-warning", str(state+1)]
        if state == 2:
            message = ["Sleep", "bg-danger", str(state+1)]

        return jsonify(message)
    else:
        return jsonify(3)


@app.route('/getpsd', methods=['GET'])  # Your API endpoint URL would consist /predict
def getpsd():
    _dict = db.psd.find()
    psd = None

    for data in _dict :
        psd = data['psd']

    if psd != None:
        return jsonify(psd)
    else:
        return jsonify(3)

@app.route("/")
def home():
   return render_template('index.html')


@app.route('/model')
def find_model():
    _dict = db.model.find()

    item = {}
    data = []

    for dict_data in _dict:
        item = {
            'name': dict_data['name']
        }
        data.append(item)

    return jsonify(
        status=True,
        model_list=data
    )

if __name__ == "__main__":
    ENVIRONMENT_DEBUG = os.environ.get("APP_DEBUG", True)
    ENVIRONMENT_PORT = os.environ.get("APP_PORT", 5000)
    app.run(host='0.0.0.0', port=ENVIRONMENT_PORT, debug=ENVIRONMENT_DEBUG)
