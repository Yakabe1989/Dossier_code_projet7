import streamlit as st
import pandas as pd
import numpy as np
import pickle
import streamlit.components.v1 as components
import shap
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
import requests, json
import plotly.express as px
import plotly.figure_factory as ff





# Chargement des donnees shap
Shap_values = pickle.load(open('./Shap_values_Dashboard', 'rb'))

#Chargement de mes ID
valeur_ID = pickle.load(open('./LISTE_ID_Dashboard.pkl', 'rb'))
#chargement image
image = Image.open('./pret_a_depenser.png')
#chargement Columns
Columns = pickle.load(open('./LISTE_COLUMNS_Dashboard.pkl', 'rb'))


@st.cache
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


#Appel de l'API
@st.cache
def Gauge(ID):
    API_URL = "https://donnees-dashboard.herokuapp.com/"
    response = requests.get(API_URL) #Chargement des données
    content = json.loads(response.content.decode('utf-8'))
    Decision = list(pd.Series(content['decision']).values)
    ID_client = list(pd.Series(content['SK_ID_CURR']).values)
    probabilite_client = list(pd.Series(content['probabilite_client']).values)
    prediction = list(pd.Series(content['prediction']).values)
    proba_gauge = list(pd.Series(content['proba_gauge']).values)
    decision = list(pd.Series(content['decision']).values)
    appreciation = list(pd.Series(content['appreciation']).values)
    tab_valeur = pd.DataFrame([probabilite_client,prediction,proba_gauge,decision,ID_client,appreciation]).transpose()
    tab_valeur.columns=['probabilite_client','prediction','proba_gauge','decision','ID_client','appreciation']
    proba_id = tab_valeur.loc[tab_valeur['ID_client'] == ID,'probabilite_client']
    prediction_id = tab_valeur.loc[tab_valeur['ID_client'] == ID, 'prediction']
    proba_gauge_id = tab_valeur.loc[tab_valeur['ID_client'] == ID, 'proba_gauge']
    decision_id = tab_valeur.loc[tab_valeur['ID_client'] == ID, 'decision']
    appreciation_id = tab_valeur.loc[tab_valeur['ID_client'] == ID, 'appreciation']
    return proba_id, prediction_id, proba_gauge_id, decision_id, appreciation_id



def Caracteristiques_client(ID):
    API_URL = "https://donnees-dashboard.herokuapp.com/"
    response = requests.get(API_URL)  # Chargement des données
    content = json.loads(response.content.decode('utf-8'))
    df = pd.DataFrame(content)
    valeur = ['SK_ID_CURR','Niveau_education','Age_en_annee_brut',
    'Statut_familial',
    'Emploi',
    'Duree_emploi_en_annee_brut',
    'taux_endettement_modelisation_en_%',
    'Revenus_brut_avant_modelisation',
    'Revenus_brut_apres_modelisation',
    'Montant_credit_avant_modelisation',
    'Montant_credit_apres_modelisation',
    'Annuite_mensuel_credit_avant_modelisation',
    'Annuite_mensuel_credit_apres_modelisation',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3',
    'prediction']
    df = df[valeur]
    df.loc[df["Duree_emploi_en_annee_brut"] < 0, "Duree_emploi_en_annee_brut"] = 0
    Caracteristiques = df.loc[df['SK_ID_CURR']==ID]
    return Caracteristiques


def Caracteristiques_client_similaires(ID):
    API_URL = "https://donnees-dashboard.herokuapp.com/"
    response = requests.get(API_URL)  # Chargement des données
    content = json.loads(response.content.decode('utf-8'))
    df = pd.DataFrame(content)
    valeur = ['SK_ID_CURR','Niveau_education','Age_en_annee_brut',
    'Statut_familial',
    'Emploi',
    'Duree_emploi_en_annee_brut',
    'taux_endettement_modelisation_en_%',
    'Revenus_brut_avant_modelisation',
    'Revenus_brut_apres_modelisation',
    'Montant_credit_avant_modelisation',
    'Montant_credit_apres_modelisation',
    'Annuite_mensuel_credit_avant_modelisation',
    'Annuite_mensuel_credit_apres_modelisation',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3',
    'prediction']
    df = df[valeur]
    df.loc[df["Duree_emploi_en_annee_brut"] < 0, "Duree_emploi_en_annee_brut"] = 0
    decision = list(df.loc[df['SK_ID_CURR'] == ID, 'prediction'])
    Caracteristiques = df.loc[df['prediction'] == decision[0]]
    return Caracteristiques.describe()




def Caracteristiques_client_opposes(ID):
    API_URL = "https://donnees-dashboard.herokuapp.com/"
    response = requests.get(API_URL)  # Chargement des données
    content = json.loads(response.content.decode('utf-8'))
    df = pd.DataFrame(content)
    valeur = ['SK_ID_CURR','Niveau_education','Age_en_annee_brut',
    'Statut_familial',
    'Emploi',
    'Duree_emploi_en_annee_brut',
    'taux_endettement_modelisation_en_%',
    'Revenus_brut_avant_modelisation',
    'Revenus_brut_apres_modelisation',
    'Montant_credit_avant_modelisation',
    'Montant_credit_apres_modelisation',
    'Annuite_mensuel_credit_avant_modelisation',
    'Annuite_mensuel_credit_apres_modelisation',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3',
    'prediction']
    df = df[valeur]
    df.loc[df["Duree_emploi_en_annee_brut"] < 0, "Duree_emploi_en_annee_brut"] = 0
    decision = list(df.loc[df['SK_ID_CURR'] == ID,'prediction'])
    Caracteristiques = df.loc[df['prediction'] != decision[0]]
    return Caracteristiques.describe()


def graphique(ID,champs):
        API_URL = "https://donnees-dashboard.herokuapp.com/"
        response = requests.get(API_URL)  # Chargement des données
        content = json.loads(response.content.decode('utf-8'))
        df = pd.DataFrame(content)
        valeur = list(pd.Series(content[champs]).values)
        prediction = list(pd.Series(content['prediction']).values)
        SK_ID_CURR = list(pd.Series(content['SK_ID_CURR']).values)
        tab_valeur = pd.DataFrame(
            [valeur, prediction, SK_ID_CURR]).transpose()
        tab_valeur.columns = ['valeur', 'prediction', 'SK_ID_CURR']
        # Add histogram data
        x1 = tab_valeur.loc[tab_valeur['prediction'] == 1, 'valeur']
        x2 = tab_valeur.loc[tab_valeur['prediction'] == 0, 'valeur']
        valeur_id = list(tab_valeur.loc[tab_valeur['SK_ID_CURR'] == ID, 'valeur'])
        # Group data together
        hist_data = [x1, x2]
        group_labels = ['Credit accordé', "credit refusé"]

        # Create distplot with custom bin_size
        fig = ff.create_distplot(
            hist_data, group_labels)
        fig.add_vline(x=valeur_id[0], line_width=3, line_dash="dash", line_color="green")
        graphique = st.plotly_chart(fig, use_container_width=True)
        return graphique





#PARTIE 1 - CONFIGURATION SIDERBAR
st.markdown('---')
# Sidebar Configuration
st.sidebar.image(image)
st.sidebar.markdown('# Visualisation de la demande de credit pour un client' )



#PARTIE 2 - ELEMENTS DU DASHBOARD
#st.set_page_config(page_title='Application de scoring credit',
#                       page_icon='random',
#                       layout='centered',
#                       initial_sidebar_state='auto')

st.title("TABLEAU DE BORD - SCORING CREDIT ")
# make prediction from the input text
st.header("Situation suite à une demande de credit")
st.subheader("Evaluation du dossier")

IDS = valeur_ID
ID = st.sidebar.selectbox('Select SK_ID from list:', IDS, key=18)
st.write('Vous avez selectionné le dossier: ', ID)


if st.sidebar.checkbox('Evaluation du dossier', key=18):
    # appel des valeurs de l'API via la fonction
    proba_id, prediction_id, proba_gauge_id, decision_id, appreciation_id = Gauge(ID)
    proba = list(proba_gauge_id)
    figa = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=proba[0],
        mode="gauge",
        title={'text': "Categorisation du dossier"},
        delta={'reference': 0},
        gauge={'axis': {'range': [-1, 1]},
               'steps': [
                   {'range': [-1, -.5], 'color': "darkred"},
                   {'range': [-0.5, 0], 'color': "red"},
                   {'range': [0, .52], 'color': "lightblue"},
                   {'range': [.52, .75], 'color': "blue"},
                   {'range': [.75, 1], 'color': "darkblue"}],
               'threshold': {'line': {'color': "grey", 'width': 5}, 'thickness': 0.25, 'value': proba[0]}}))
    st.plotly_chart(figa)
    st.write(appreciation_id)

if st.sidebar.checkbox('Decision du credit', key=18):
    st.header("Decision de credit")
    proba_id, prediction_id, proba_gauge_id, decision_id, appreciation_id = Gauge(ID)
    if int(prediction_id) == 1:
        st.write("Credit accordé avec un seuil de decision de ",proba_id)
    else:
        st.write("Credit refusé avec un seuil de decision de ",proba_id)



if st.sidebar.checkbox("Features importance", key=38):
    # Afficher le graphique shap local
    st.header("Analyse des predictions du modele")
    data = pd.read_csv("./X_test_data_Dashboard.csv", sep=',')
    shap_data = data.reset_index(drop=True)
    shap_index = list(shap_data.index[shap_data['SK_ID_CURR'] == ID])


    st.subheader("modele local")
    fig2, ax = plt.subplots(nrows=1, ncols=1)
    shap.waterfall_plot(shap.Explanation(values=Shap_values[1][shap_index[0]],
                                                  base_values=0.407, data=shap_data.iloc[shap_index[0]]))
    st.pyplot(fig2)

    with st.expander("Description du modèle local"):
        st.write("""
                ce graphique présente les variables discriminantes pour l'acceptation ou
                le refus de credit pour un individu. en rouge, les champs favorisant une acceptation de credit.
                En bleu, ceux favorisant un refus.
            """)

if st.sidebar.checkbox("Caractéristiques clients", key=38):
    st.subheader("Description des caracteristiques du client")
    st.write(Caracteristiques_client(ID))

if st.sidebar.checkbox("Caractéristiques moyennes des décisions similaires", key=38):
    st.subheader("Description des caracteristiques moyennes des clients ayant la meme décision")
    st.write(Caracteristiques_client_similaires(ID))

if st.sidebar.checkbox("Caractéristiques moyennes des décisions opposés", key=38):
    st.subheader("Description des caracteristiques moyennes des clients ayant une decision opposée")
    st.write(Caracteristiques_client_opposes(ID))


if st.sidebar.checkbox("Analyse visuelle", key=38):
    st.subheader('comparaison visuelle des individus avec credits acceptés et refusés')
    API_URL = "https://donnees-dashboard.herokuapp.com/"
    response = requests.get(API_URL)  # Chargement des données
    content = json.loads(response.content.decode('utf-8'))
    #Colonne = list(content.keys())
    Colonne = ['EXT_SOURCE_1','EXT_SOURCE_2',
    'EXT_SOURCE_3','Age_en_annee_brut',
    'Annuite_mensuel_credit_apres_modelisation','Annuite_mensuel_credit_avant_modelisation',
    'AMT_ANNUITY','AMT_CREDIT','AMT_GOODS_PRICE','AMT_INCOME_TOTAL',
    'DAYS_BIRTH','DAYS_EMPLOYED','Duree_emploi_en_annee_brut','INCOME_PER_PERSON',
    'Montant_credit_apres_modelisation','Montant_credit_avant_modelisation','Nombre_enfants']
    champs = st.selectbox(
        'Quel champ voulez vous analyser?',
        Colonne)
    graphique(ID,champs)


st.sidebar.write('Developpé par Yannick AKABE')
st.sidebar.write('Projet 7 OPENCLASSROOM')





