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

for file_plk in list_model:
    if file_plk != 'columns.pkl':
        model = joblib.load(path_model+file_plk)
        plk_model = pickle.dumps(model)
        model_name = file_plk.replace(".pkl", "")
        db.model.insert_one({model_name: plk_model, 'name': model_name})
    else:
        columns = joblib.load(path_model+file_plk)
        db.model.insert_one({file_plk.replace(".pkl", ""): columns, 'name': file_plk.replace(".pkl", "")})


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
    data = db.model.find({'name': 'LinearRegression'})
    for i in data:
        json_data = i
    pikled_model = json_data['LinearRegression']
    model_loaded = pickle.loads(pikled_model)

    prediction = list(model_loaded.predict(query))
    return jsonify({'prediction': prediction})


@app.route("/")
def hello():
    return "Welcome to machine learning model APIs for predict EEG drowsiness!"


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
