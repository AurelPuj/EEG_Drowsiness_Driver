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

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import traceback


def start_api():

    app = Flask("MyEEG")

    model = joblib.load('../api/LinearRegression.pkl')
    print('Model loaded')

    model_columns = joblib.load('../api/columns.pkl')
    print('Model columns loaded')

    @app.route('/predict', methods=['POST'])  # Your API endpoint URL would consist /predict
    def predict():
        if model:
            try:
                json_ = request.json
                query = pd.get_dummies(pd.DataFrame(json_))
                query = query.reindex(columns=model_columns, fill_value=0)
                prediction = list(model.predict(query))
                return jsonify({'prediction': prediction})

            except:
                return jsonify({'trace': traceback.format_exc()})

        else:
            print('Train the model first')
            return 'No model here to use'

    @app.route("/")
    def hello():
        return "Welcome to machine learning model APIs!"

    app.run(debug=True)

