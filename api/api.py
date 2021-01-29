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

# Create API and load ML Algo
app = Flask("MyEEG")
print("lancement de l'api")
model = joblib.load('models/LinearRegression.pkl')
print('Model loaded')
model_columns = joblib.load('models/columns.pkl')
print('Model columns loaded')

# Initialise the database
app.config['MONGO_DBNAME'] = 'eeg_raw_data'
app.config["MONGO_URI"] = 'mongodb://' + os.environ['MONGODB_USERNAME'] + ':' + os.environ['MONGODB_PASSWORD'] + '@' + os.environ['MONGODB_HOSTNAME'] + ':27017/' + os.environ['MONGODB_DATABASE']
mongo = PyMongo(app)
db = mongo.db


@app.route('/predict', methods=['POST'])  # Your API endpoint URL would consist /predict
def predict():
    if model:
        try:
            json = request.json
            query = pd.get_dummies(pd.DataFrame(json))
            query = query.reindex(columns=model_columns, fill_value=0)
            prediction = list(model.predict(query))
            return jsonify({'prediction': prediction})

        except:
            return "An error occur"

    else:
        print('Train the model first')
        return 'No model here to use'


@app.route("/")
def hello():
    return "Welcome to machine learning model APIs!"


@app.route('/find')
def find():
    _dict = db.todo.find()

    item = {}
    data = []

    for dict_data in _dict:
        item = {
            'id': str(dict_data['_id']),
            'data': dict_data['data']
        }
        data.append(item)

    return jsonify(
        status=True,
        data=data
    )


@app.route('/upload', methods=['POST'])
def create_data():
    dict_data = request.get_json(force=True)
    item = {
        'data': dict_data['data']
    }
    db.todo.insert_one(item)

    return jsonify(
        status=True,
        message='To-do saved successfully!'
    ), 201


if __name__ == "__main__":
    ENVIRONMENT_DEBUG = os.environ.get("APP_DEBUG", True)
    ENVIRONMENT_PORT = os.environ.get("APP_PORT", 5000)
    app.run(host='0.0.0.0', port=ENVIRONMENT_PORT, debug=ENVIRONMENT_DEBUG)
