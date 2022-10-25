from flask import Flask, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)


def SCORING(ID_CLIENT):
    data = pd.read_csv(
        "./X_test_data_Dashboard_V1.csv", sep=',')
    # load model
    model = pickle.load(open('./logistic_model_V1.pkl', 'rb'))
    data_bis = data.drop(columns=['SK_ID_CURR'])
    ID_CLIENT = int(ID_CLIENT)
    # probabilite et prediction
    data['model_proba_client'] = model.predict_proba(data_bis)[:, 1]
    data['model_prediction'] = np.where(data['model_proba_client'] > 0.52, 1, 0)
    data['model_decision_credit'] = np.where(data['model_prediction'] > 0, "Acceptation_credit", "Refus_credit")
    ID_probabilite = data.loc[data['SK_ID_CURR'] == ID_CLIENT, 'model_proba_client']
    ID_prediction = data.loc[data['SK_ID_CURR'] == ID_CLIENT, 'model_prediction']
    ID_decision_credit = data.loc[data['SK_ID_CURR'] == ID_CLIENT, 'model_decision_credit']
    probabilite = list(ID_probabilite)
    prediction = list(ID_prediction)
    decision_credit = list(ID_decision_credit)
    return probabilite, prediction, decision_credit


@app.route('/get/<ID_CLIENT>', methods=['GET'])
def api_all(ID_CLIENT):
        probabilite, prediction, decision_credit = SCORING(ID_CLIENT)
        books = {'ID_CLIENT': ID_CLIENT,
                 'probabilite_dossier': probabilite,
                 'prediction_dossier': prediction,
                 'decision_credit_dossier': decision_credit
                 }
        return jsonify(books)

if __name__ == '__main__':
    app.run()




