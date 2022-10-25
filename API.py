from flask import Flask, render_template, request,jsonify
from pickle5 import pickle
import pandas as pd


app = Flask(__name__)

@app.route('/')
def Scoring():  # put application's code here
    data = pd.read_csv(
            'C:/Users/akabe/PycharmProjects/flaskProject/data/rt.csv', sep=',')
    model = pickle.load(open('C:/Users/akabe/PycharmProjects/flaskProject/data/logistic_model', 'rb'))
    data_bis = data.copy()
    data_bis['prediction'] = model.predict(data)
    data_bis['score_probabilite'] = model.predict_proba(data)[:, 1]
    CUSTOMER_ID = list(data_bis['SK_ID_CURR'])
    PREDICTION = list(data_bis['prediction'])
    VALEUR_PREDICTION = list(data_bis['score_probabilite'])

    return jsonify({
        'CUSTOMER_ID': CUSTOMER_ID,
        'CUSTOMER_SCORING_NEW': {
        'PREDICTION': PREDICTION,
        'SEUIL_DECISION': VALEUR_PREDICTION
        }

    })

if __name__ == '__main__':
    app.run(debug=True)

