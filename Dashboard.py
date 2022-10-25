import streamlit as st
import pandas as pd
import numpy as np
import joblib
import streamlit.components.v1 as components
import shap
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff

X_test_data = pd.read_csv("C:/Users/akabe/PycharmProjects/pythonProject/donnees/X_test_data_b.csv", sep=',')
shap_data = pd.read_csv("C:/Users/akabe/PycharmProjects/pythonProject/donnees/X_test_data_b.csv", sep=',')
application_train = pd.read_csv("C:/Users/akabe/PycharmProjects/pythonProject/donnees/application_train.csv", sep=',')
previous_application = pd.read_csv("C:/Users/akabe/PycharmProjects/pythonProject/donnees/previous_application.csv", sep=',')
credit_card_balance = pd.read_csv("C:/Users/akabe/PycharmProjects/pythonProject/donnees/credit_card_balance.csv", sep=',')
Qualite_dossier = pd.read_csv("C:/Users/akabe/PycharmProjects/pythonProject/donnees/Qualite_dossier.csv", sep=',')

#Chargement des donnees shap
Shap_values = joblib.load('C:/Users/akabe/PycharmProjects/pythonProject/donnees/Shap_values')
explainer = joblib.load('C:/Users/akabe/PycharmProjects/pythonProject/donnees/explainer')

#Chargement de l'image
image = Image.open('C:/Users/akabe/PycharmProjects/pythonProject/donnees/pret_a_depenser.png')

#Chargement du modele
model = joblib.load('C:/Users/akabe/PycharmProjects/pythonProject/donnees/logistic_model')

#calcul de la proba Client
seuil_decision = 0.52
X_test_data['proba_client'] = model.predict_proba(X_test_data)[:, 1]
X_test_data['prediction'] = np.where(X_test_data['proba_client']>seuil_decision, 1, 0)

#je recodifie les proba_client
X_test_data['proba_client_bis'] = np.where(X_test_data['proba_client']> seuil_decision,X_test_data['proba_client'],X_test_data['proba_client']-0.52)

#Creation des fonctions

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

@st.cache
def make_prediction(ID):
    donnees_client = X_test_data.loc[X_test_data['SK_ID_CURR'] == ID]
    probabilite = donnees_client['proba_client']
    prediction = donnees_client['prediction']
    return probabilite, prediction

st.markdown('---')
# Sidebar Configuration
st.sidebar.image(image)
st.sidebar.markdown('# Visualisation de la demande de credit pour un client' )

st.sidebar.markdown('---')
form = st.sidebar.form(key="my_form")
ID = form.number_input(label="Veuillez entrer un ID client")
submit = form.form_submit_button(label="Decision suite à la prediction")
# Declare a form to receive a movie's review
st.sidebar.write('Developpé par Yannick AKABE')
st.sidebar.write('Projet 7 OPENCLASSROOM')

#elements du dashboard
st.title("TABLEAU DE BORD - CREDIT ")



# make prediction from the input text
probabilite, prediction = make_prediction(ID)
st.header("Situation suite à une demande de credit")
st.subheader("Evaluation du dossier")
st.markdown("cette premiere partie permet d'evaluer la qualité du dossier et de la classer parmi 5 categories\
            en fonction de la gauge. Une gauge inférieur à 0 indiquera un dossier pour lequel on un refus du credit\
            tandis qu'une gauge superieure à 0 indiquera un dossier avec une acceptation du credit")
if submit:
    donnees_client = X_test_data.loc[X_test_data['SK_ID_CURR'] == ID]
    proba = list(donnees_client.loc[donnees_client['SK_ID_CURR'] == ID, "proba_client_bis"])
    #proba[0]
    figa = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value= -0.01,
        mode="gauge",
        title={'text': "Seuil probabilité de remboursement"},
        delta={'reference': 0},
        gauge={'axis': {'range': [-1, 1]},
                'steps': [
                    {'range': [-1, -.76], 'color': "darkred"},
                    {'range': [-.75, -.50], 'color': "red"},
                    {'range': [-.49, 0], 'color': "lightblue"},
                    {'range': [0, .49], 'color': "blue"},
                    {'range': [.5, 1], 'color': "darkblue"}],
                   'threshold': {'line': {'color': "grey", 'width': 5}, 'thickness': 0.25, 'value': -0.01}}))
    st.plotly_chart(figa)
    st.write("Le seuil metier de separation entre les mauvais et bons dossiers est 0.52.")
    st.write("la probabilite de remboursement pour ce dossier est : ",probabilite)
    if proba[0] < .26:
        st.write("le dossier est tope tres mauvais dossier")
    if 0.26 <= proba[0] < .53:
        st.write("le dossier est tope mauvais dossier")
    if 0.53 <= proba[0] < .76:
        st.write("le dossier est tope dossier acceptable")
    if 0.76 <= proba[0] < .89:
        st.write("le dossier est tope bon dossier")
    if 0.89 <= proba[0] :
        st.write("le dossier est tope tres bon dossier")

    # make prediction from the input text
    probabilite, prediction = make_prediction(ID)


    # afficher les resultats de la prediction et la proba
    st.header("Decision de credit")
    if int(prediction) == 1:
        st.write("Credit accordé")
        # Let us display the Raw Data into our Web App
    else:
        st.write("Credit refusé")

    # Afficher le graphique shap local et shap global - shap de l'individu moyen de la classe
    st.header("Analyse des predictions du modele")
    st.subheader("modele local")
    # Shap values
    shap_data = shap_data.reset_index(drop=True)
    shap_index = shap_data.index[shap_data['SK_ID_CURR'] == ID]
    shap_local = st_shap(
        shap.force_plot(explainer.expected_value[1], Shap_values[1][shap_index, :], shap_data.iloc[shap_index, :]))
    st.subheader('modele global')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(Shap_values, shap_data, feature_names=shap_data.columns, max_display=10)
    st.pyplot(fig)



    # Afficher les caractéristiques individuelles du client - ddn - nombre de credit
    st.header("Caracteristiques individuelles du client")

    #je renomme les champs qui m'interesse dans les tables d'interet
    application_train.rename(columns={'CODE_GENDER': 'Sexe',
                                'NAME_CONTRACT_TYPE': 'Type_contrat_credit',
                                'CNT_CHILDREN': 'Nombre_enfants',
                                'NAME_INCOME_TYPE': 'Sources_revenus',
                                'NAME_EDUCATION_TYPE': 'Niveau_education',
                                'NAME_FAMILY_STATUS': 'Statut_familial',
                                'OCCUPATION_TYPE': 'Emploi',
                                'DAYS_BIRTH': 'Nombre_jours_naissance',
                                'DAYS_EMPLOYED': 'Duree_emploi',
                                'AMT_INCOME_TOTAL': 'Revenus',
                                'AMT_CREDIT': 'Montant_credit',
                                'AMT_ANNUITY': 'Annuite_mensuel_credit'}, inplace=True)

    info_client1 = application_train[['Sexe','Type_contrat_credit','Nombre_enfants',
                                      'Sources_revenus','Niveau_education','Statut_familial',
                                      'Emploi','Nombre_jours_naissance','Duree_emploi',
                                      'Revenus','Montant_credit','Annuite_mensuel_credit','SK_ID_CURR']]

    info_client1['Sexe'] = info_client1['Sexe'].map({'M': 'Homme', 'F': 'Femme'})
    info_client1['Age_en_annee'] = round(info_client1['Nombre_jours_naissance'] / -365)
    info_client1['Duree_emploi_en_annee'] = round(info_client1["Duree_emploi"] / -365)
    info_client1['taux_endettement_en_%'] = (info_client1["Annuite_mensuel_credit"]/info_client1["Revenus"])*100
    info_client1.drop(columns=['Nombre_jours_naissance'],inplace=True)

    previous_application.rename(columns={'NAME_CONTRACT_TYPE': 'Type_contrat_credit',
                                'AMT_CREDIT': 'Montant_credit',
                                'AMT_ANNUITY': 'Annuite_mensuel_credit',
                                'NAME_CONTRACT_STATUS': 'statut_contrat_credit',
                                'DAYS_TERMINATION': 'date_fin_remboursement',
                                'NAME_PAYMENT_TYPE': 'type_paiement'}, inplace=True)

    info_client2 = previous_application[['Type_contrat_credit', 'Montant_credit', 'Annuite_mensuel_credit',
                                      'statut_contrat_credit', 'type_paiement','SK_ID_CURR','date_fin_remboursement']]
    info_client2['date_fin_remboursement'] = round(info_client2['date_fin_remboursement'] / -365)

    caracteristiques_indiv = X_test_data.loc[X_test_data['SK_ID_CURR'] == ID, ['CNT_CHILDREN','CODE_GENDER','AMT_INCOME_TOTAL']]
    caracteristiques_indiv1 = info_client1.loc[info_client1['SK_ID_CURR'] == ID]
    caracteristiques_indiv2 = info_client2.loc[(info_client2['SK_ID_CURR'] == ID) & ((info_client2['statut_contrat_credit'] == 'Approved'))]
    st.subheader("Description de l'individu")
    st.write(caracteristiques_indiv1)
    st.subheader("historique de credit accepté")
    st.write(caracteristiques_indiv2)

    #test des comparaisons
    if 'type' not in st.session_state:
        st.session_state['type'] = 'Categorical'

    df = pd.read_csv("C:/Users/akabe/PycharmProjects/pythonProject/donnees/X_test_data_b.csv", sep=',')

    types = {'Categorical': ['CODE_GENDER','NAME_CONTRACT_TYPE'], 'Numerical': ['CNT_CHILDREN','AMT_INCOME_TOTAL']}
    column = st.selectbox('Selectionner un champ', types[st.session_state['type']])

    def handle_click(new_type):
        st.session_state.type = new_type


    type_of_column = st.radio("Quelle type de champ voulez vous analyser?", ['Categorical', 'Numerical'])
    change = st.button('Change', on_click=handle_click, args=[type_of_column])

    #st.session_state['type'] = st.radio("Quelle type de champ voulez vous analyser?", ['Categorical', 'Numerical'])

    if st.session_state['type']=='Categorical':
        dist = pd.DataFrame(df[column].value_counts()).head()
        st.bar_chart(dist)
    else:
        st.table(df[column].describe())




