import calendar
import joblib
import pycaret
import seaborn as sns
from pycaret.regression import *
from sklearn.model_selection import StratifiedKFold
from joblib import load
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit_authenticator as stauth
from PIL import Image
import time
import datetime as dt
from datetime import datetime
import streamlit.components.v1
import io
import plotly.express as px
import yaml
from yaml.loader import SafeLoader
#################################

st.set_page_config(layout="wide")

st.session_state['authenticated'] = False
hashed_passwords = stauth.Hasher(['abc', 'test']).generate()

# Chargement de la configuration à partir d'un fichier YAML
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Création d'un objet d'authentification
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Vérification de l'état d'authentification de l'utilisateur
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# Si l'utilisateur n'est pas authentifié, afficher les champs de connexion
if not st.session_state['authenticated']:
    # st.title("Page de connexion")
    # # Défilement du texte en utilisant des styles CSS personnalisés
    # st.markdown(
    #     """
    #     <style>
    #         .marquee {width: 100%; overflow: hidden; white-space: nowrap;}
    #         .marquee p {display: inline-block;animation: marquee 10s linear infinite;}
    #         @keyframes marquee {
    #             0% { transform: translateX(100%); }
    #             100% { transform: translateX(-100%); }
    #         }
    #     </style>
    #     <div class="marquee">
    #         <p style="font-weight: bold; font-size: 20px;color:red;">Veuillez vous authentifié(e) pour avoir accès à l'application"</p>
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )
    col1, col2 = st.columns(2)
    with col1 :
        name, authentication_status, username = authenticator.login("S\'authentifier", 'main')
    # with col2:
    #     login = Image.open('images/login2.jpg')
    #     # Redimensionner l'image (par exemple, à 50% de sa taille d'origine)
    #     largeur, hauteur = login.size
    #     nouvelle_largeur = largeur // 2  # Vous pouvez ajuster la taille comme vous le souhaitez
    #     nouvelle_hauteur = hauteur // 2
    #     login_redimensionnee = login.resize((nouvelle_largeur, nouvelle_hauteur))
    #     # Afficher l'image redimensionnée dans Streamlit
    #     st.image(login_redimensionnee)
    if authentication_status:
        st.session_state['authenticated'] = True
        # col2.empty()

    #########################     PAGE D'ACCUEIL            #########################

# Si l'utilisateur est authentifié, afficher les fonctionnalités de l'application

if st.session_state['authenticated']:
    authenticator.logout('Se déconnecter', 'sidebar')
    # Affiche le texte pendant 5 secondes
    text_container = st.empty()
    text_container.success(f'Welcome *{st.session_state["name"]}*')
    time.sleep(5)
    # Supprime le texte après 5 secondes
    text_container.empty()

    # ajout du sidebar (une sidebar est une colonne ( : bar) placée sur la droite ou la gauche de la page principale)
    st.sidebar.title("STRATEGIES")
    add_selectbox = st.sidebar.selectbox("Choisir la stratégie",("PAGE D'ACCUEIL","SEGMENTATION", "PREVISIONS DE VENTE"), key=1)
#####////////////////////////////////////////////////////////////////////
    # taaral = Image.open('images/taaral.jpg')
    # BMP = Image.open('images/BMP.jpg')
    # st.sidebar.image(BMP)

    # col1,col2,col3,col4, col5, col6 = st.columns(6)
    # with col1 :
    #     accueil_button = st.button("Page d'accueil", key=103)
    # with col2 :
    #     segmentation_button = st.button("Segmentation", key=101)
    # with col3:
    #     Prévisions_button = st.button("Prévisions", key=102)
    # st.write("-----")


    if add_selectbox == 'PAGE D\'ACCUEIL' :

        # Défilement du texte en utilisant des styles CSS personnalisés
        st.markdown(
            """
            <style>
                .marquee {width: 100%; overflow: hidden; white-space: nowrap;}
                .marquee p {display: inline-block;animation: marquee 10s linear infinite;}
                @keyframes marquee {
                    0% { transform: translateX(100%); }
                    100% { transform: translateX(-100%); }
                }
            </style>
            <div class="marquee">
                <p style="font-weight: bold; font-size: 20px; color:blue;">BIENVENU(E) A TAARAL, "ku ñuulul dangaa xess te ñépp a rafet"</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Utilisation de balises HTML et CSS pour personnaliser l'apparence
        html_template = """
        <div style="background-color: #B9D6F5; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: #333; font-family: 'Arial', sans-serif; align="center";>BIENVENU(E) SUR L'APPLICATION WEB</h2>
            <p style="font-size: 18px; color: #666;">MARKETING PREDICTIF</p>
        </div>
        """
        # Afficher le contenu HTML personnalisé dans Streamlit
        st.markdown(html_template, unsafe_allow_html=True)

        st.write("")
        col1, col2, col3 = st.columns(3)
        #
        with col1:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.markdown("""
                <style>
                    /* Style du conteneur */
                    .container {
                        background-color: #f5f5f5;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    }

                    /* Style du texte */
                    .text {
                        font-size: 18px;
                        color: #333;
                        line-height: 1.5;
                    }
                </style>
            """, unsafe_allow_html=True)
            st.write("""
                <div class="container">
                    <p class="text">
                        L'application web de marketing prédictif basée sur Streamlit est un outil puissant conçu pour aider les professionnels du marketing à améliorer leur stratégie et leurs résultats. Elle intègre des modèles de machine learning avancés pour analyser les données marketing et générer des prévisions de vente pertinentes sur les ressources de l'entreprise.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.image("images/marketing4.jpg", width=380)
        with col3:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.markdown("""
                <style>
                    /* Style du conteneur */
                    .container {
                        background-color: #f5f5f5;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    }

                    /* Style du texte */
                    .text {
                        font-size: 18px;
                        color: #333;
                        line-height: 1.5;
                    }
                </style>
            """, unsafe_allow_html=True)
            st.write("""
                <div class="container">
                    <p class="text">
                        Le marketing prédictif fonctionne principalement par le biais du Big Data. Il est le carburant des systèmes prédictifs : sans lui, les analyses ne seraient pas concluantes ni pertinentes, voire impossibles. \n L’utilisation du marketing prédictif repose donc sur des outils de Machine learning ou encore des outils de scoring.</p>
                </div>
            """, unsafe_allow_html=True)


    #########################            SEGMENTATION            #########################

    if add_selectbox == 'SEGMENTATION':
        # ajout du sidebar (une sidebar est une colonne ( : bar) placée sur la droite ou la gauche de la page principale)

        st.sidebar.title("BARRE DE NAVIGATION")
        add_selectbox = st.sidebar.selectbox("Choisir la méthode", ("-----", "Segmentation RFM", "Rubrique anniversaire"), key=2)

        if add_selectbox == '-----':
            st.markdown("""
                <style>
                    /* Style du cadre */
                    .custom-frame {
                        border: 2px solid #0078d4;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    }
    
                    /* Style du titre */
                    .custom-title {
                        font-size: 36px;
                        font-weight: bold;
                        color: #0078d4;
                        text-align: center;
                        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
                    }
                </style>
            """, unsafe_allow_html=True)

            st.write("""
                <div class="custom-frame">
                    <div class="custom-title">
                        SEGMENTATION CLIENTELE
                    </div>
                </div>
            """, unsafe_allow_html=True)
            #####

            st.write("")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("")
                st.write("")
                st.image("images/segmentation1.jpg", width=400)
            with col2:
                st.write("")
                st.markdown("""
                            <style>
                                /* Style du conteneur */
                                .container {
                                    background-color: #f5f5f5;
                                    padding: 20px;
                                    border-radius: 10px;
                                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                                }
    
                                /* Style du texte */
                                .text {
                                    font-size: 18px;
                                    color: #333;
                                    line-height: 1.5;
                                }
                            </style>
                        """, unsafe_allow_html=True)
                st.write("""
                            <div class="container">
                                <p class="text">
                                    La segmentation est le processus de division du marché cible d’une entreprise en groupes de clientes et clients potentiels aux besoins et aux comportements similaires. Cela permet à l’entreprise de vendre à chaque groupe de clientes et clients en utilisant des stratégies distinctes adaptées à leurs besoins. La segmentation permet à l’entreprise de cerner et de sélectionner les groupes de clientes et clients le plus fort potentiel de rentabilité. Cela dépend des besoins, du comportement et de la probabilité de payer de la cliente ou du client.
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

                with col3:
                    st.write("")
                    st.write("")
                    st.image("images/segmentation2.jpg", width=400)

####################################################################

        if add_selectbox == "Segmentation RFM":
            st.markdown(
                '''
                <style>
                    @keyframes blink {
                        0% { opacity: 1; color: red; }
                        50% { opacity: 1; color: blue; }
                        100% { opacity: 1; color: green; }
                    }

                    .blinking-text {
                        animation: blink 5s infinite;
                    }
                </style>
                ''', unsafe_allow_html=True
            )

            st.markdown(
                '''
                <h1 class="blinking-text">SEGMENTATION RFM</h1>
                ''', unsafe_allow_html=True
            )

            st.subheader('1. CHARGEMENT DU FICHIER EXCEL/CSV🔧')
            col1, col2 = st.columns(2)
            with col1:
                file_upload = st.file_uploader("Charger le fichier Excel/csv", type=["xlsx", "csv"], key=3)
                # chargement du fichier csv pour le traitement
                if file_upload is not None:
                    data = pd.read_excel(file_upload)
                if 'data' in locals():
                    # Si les données existent, informer l'utilisateur avec un message
                    st.success("Base de données chargées avec succès!")
                    st.write("")
                else:
                    # Si les données n'existent pas, affichez un avertissement
                    st.warning("Chargez d'abord le fichier Excel/csv.")

            analysis_date = data["Dernière_visite"].max()+ pd.Timedelta(days=1)
            rfm = data.groupby('ID').agg(
                {'Dernière_visite': lambda Dernière_visite: (analysis_date - Dernière_visite.max()).days,
                 'Nombre_de_visites': lambda Nombre_de_visites: Nombre_de_visites,
                 'MontantTotal': lambda MontantTotal: MontantTotal
                 })

            # renomination des colonnes
            rfm.columns = ['recency', 'frequency', 'monetary']
            rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
            rfm['frequency'] = pd.to_numeric(rfm['frequency'], errors='coerce')                                     # 'coerce' traite les valeurs non numériques comme NaN
            rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
            rfm['monetary'] = pd.to_numeric(rfm['monetary'], errors='coerce')                                       # 'coerce' traite les valeurs non numériques comme NaN
            rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
            rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

            # mesures de récence divisées en 5 parties égales en fonction de leurs tailles et étiquetées; la plus grande récence a obtenu 1 , la plus petite récence a obtenu 5 . , 4 , 5 ])
            # par définition RFM, la chaîne est créée avec le score de récence et de fréquence et le score RFM final formé
            # le score monétaire est nécessaire pour l'observation, mais il n'est pas utilisé dans la formation du score RFM

            # regex
            # RFM Naming (Pattern Matching)
            seg_map = {
                r'[1-2][1-2]': 'hibernating',
                r'[1-2][3-4]': 'at_Risk',
                r'[1-2]5': 'cant_loose',
                r'3[1-2]': 'about_to_sleep',
                r'33': 'need_attention',
                r'[3-4][4-5]': 'clients_loyaux',
                r'41': 'prometteur',
                r'51': 'nouveaux_clients',
                r'[4-5][2-3]': 'potential_loyalists',
                r'5[4-5]': 'champions'
            }
            rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)


            # Fondamentalement, la table RF est codée à l'aide de regex. Le paramètre 'regex=True' dans la méthode replace() indique que les clés du dictionnaire doivent être interprétées
            # comme des expressions régulières. Cela permet une correspondance plus flexible des valeurs dans la colonne 'RFM_SCORE'.

            # Par exemple, si la colonne 'RFM_SCORE' contient une valeur de '111' et que le dictionnaire 'seg_map' fait correspondre
            # la valeur '111' à l'étiquette de segment 'Meilleurs clients', alors la valeur correspondante dans la colonne 'segment' sera 'Meilleurs clients'

            if st.button("Segmenter"):
                rfm = pd.merge(rfm, data, on="ID")
                st.subheader("Tableau complet segmenté")
                st.write(rfm)

            st.write("--------------------------------------------------------------")

            st.subheader("CATEGORISATION DES CLIENTS")
            # st.write("\n")
            col1, col2, col3, col4, col5, col6,col7, col8, col9, col10 = st.columns(10)

            with col1:
                a = st.button("Clients Champions")
                st.write(a)
            with col2:
                b = st.button("Clients fidèles")
                st.write(b)
            with col3:
                c = st.button("Clients potentiellement fidèles")
                st.write(c)
            with col4:
                d = st.button("Nouveaux clients ")
                st.write(d)
            with col5:
                e = st.button("Clients prometteur")
                st.write(e)
            with col6:
                f = st.button("Clients méritant une attention")
                st.write(f)
            with col7:
                g = st.button("Clients sur le point de s’endormir")
                st.write(g)
            with col8:
                h = st.button("Clients à risque")
                st.write(h)
            with col9:
                i = st.button("Clients à ne pas perdre")
                st.write(i)
            with col10:
                j = st.button("Clients en hibernation")
                st.write(j)

            if a == True :
                st.markdown("""<style>.marquee {width: 100%; overflow: hidden; white-space: nowrap;}.marquee p {display: inline-block;animation: marquee 10s linear infinite;font-size: 20px;color: red; align=center}@keyframes marquee {0% { transform: translateX(100%); }100% { transform: translateX(-100%); }}      
                    </style><div class="marquee"><p>LES CLIENTS CHAMPIONS</p></div>""",unsafe_allow_html=True)

                st.write("Les Champions sont nos meilleurs clients. Cette catégorie regroupe ceux qui ont acheté récemment (R élevé), qui achètent très régulièrement (F élevée) et qui dépensent le plus (M élevé)."
                         " Ce sont des clients que nous devons chouchouter et qui méritent toute notre attention :")
                st.write("\n")
                st.write("\n")
                rfm1 = pd.merge(rfm, data, on="ID")
                champions = rfm1[rfm1["segment"] == "champions"]
                st.write(champions)

                # Convertissez le DataFrame en un fichier Excel
                a = champions
                excel_buffer = io.BytesIO()
                excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
                a.to_excel(excel_writer, index=False)
                excel_writer.save()
                excel_buffer.seek(0)

                # Téléchargez le fichier Excel
                st.download_button(
                    label="Download",
                    data=excel_buffer,
                    file_name="champions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.write("---------")
                st.write("")


            if b == True :
                st.markdown("""<style>.marquee {width: 100%; overflow: hidden; white-space: nowrap;}.marquee p {display: inline-block;animation: marquee 10s linear infinite;font-size: 20px;color: red; align=center}@keyframes marquee {0% { transform: translateX(100%); }100% { transform: translateX(-100%); }}      
                                    </style><div class="marquee"><p>LES CLIENTS FIDELES</p></div>""",
                            unsafe_allow_html=True)

                st.write("Le client fidèle se situe une catégorie en dessous : il dépense en général beaucoup (M élevé) et régulièrement (F élevé). Il doit tout de même faire l’objet de toute votre attention.")
                st.write("\n")
                st.write("\n")
                rfm2 = pd.merge(rfm, data, on="ID")
                fideles = rfm2[rfm2["segment"] == "clients_loyaux"]
                st.write(fideles)

                # Convertissez le DataFrame en un fichier Excel
                a = fideles
                excel_buffer = io.BytesIO()
                excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
                a.to_excel(excel_writer, index=False)
                excel_writer.save()
                excel_buffer.seek(0)

                # Téléchargez le fichier Excel
                st.download_button(
                    label="Download",
                    data=excel_buffer,
                    file_name="clients_loyaux.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            if c == True :
                st.markdown("""<style>.marquee {width: 100%; overflow: hidden; white-space: nowrap;}.marquee p {display: inline-block;animation: marquee 10s linear infinite;font-size: 20px;color: red; align=center}@keyframes marquee {0% { transform: translateX(100%); }100% { transform: translateX(-100%); }}      
                                    </style><div class="marquee"><p>LES CLIENTS POTENTIELLEMENT FIDELES</p></div>""",
                            unsafe_allow_html=True)

                st.write("Le client potentiellement fidèle est un client récemment acquis (R faible), mais qui a dépensé une belle somme d’argent (M élevé) et a acheté plusieurs fois (F élevé). Il vous reste encore un peu de chemin à parcourir pour faire naitre chez votre client le sentiment d’appartenance à la marque par rapport à la concurrence.")
                st.write("\n")
                st.write("\n")
                rfm3 = pd.merge(rfm, data, on="ID")
                potential_loyalists = rfm3[rfm3["segment"] == "potential_loyalists"]
                st.write(potential_loyalists)

                # Convertissez le DataFrame en un fichier Excel
                a = potential_loyalists
                excel_buffer = io.BytesIO()
                excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
                a.to_excel(excel_writer, index=False)
                excel_writer.save()
                excel_buffer.seek(0)

                # Téléchargez le fichier Excel
                st.download_button(
                    label="Download",
                    data=excel_buffer,
                    file_name="potential_loyalists.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            if d == True :
                st.markdown("""<style>.marquee {width: 100%; overflow: hidden; white-space: nowrap;}.marquee p {display: inline-block;animation: marquee 10s linear infinite;font-size: 20px;color: red; align=center}@keyframes marquee {0% { transform: translateX(100%); }100% { transform: translateX(-100%); }}      
                                    </style><div class="marquee"><p>LES NOUVEAUX CLIENTS                    </p></div>""",
                            unsafe_allow_html=True)

                st.write("Comme son nom l’indique, le client récent a fait récemment appel à vos services pour la première fois (R élevé). Parce qu’il est encore loin d’être acquis, le client récent demande une attention toute particulière")
                st.write("\n")
                st.write("\n")
                rfm4 = pd.merge(rfm, data, on="ID")
                nouveaux_clients = rfm4[rfm4["segment"] == "nouveaux_clients"]
                st.write(nouveaux_clients)

                # Convertissez le DataFrame en un fichier Excel
                a = nouveaux_clients
                excel_buffer = io.BytesIO()
                excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
                a.to_excel(excel_writer, index=False)
                excel_writer.save()
                excel_buffer.seek(0)

                # Téléchargez le fichier Excel
                st.download_button(
                    label="Download",
                    data=excel_buffer,
                    file_name="nouveaux_clients.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            if e == True :
                st.markdown("""<style>.marquee {width: 100%; overflow: hidden; white-space: nowrap;}.marquee p {display: inline-block;animation: marquee 10s linear infinite;font-size: 20px;color: red; align=center}@keyframes marquee {0% { transform: translateX(100%); }100% { transform: translateX(-100%); }}      
                                    </style><div class="marquee"><p>LES CLIENTS PROMETTEURS                 </p></div>""",
                            unsafe_allow_html=True)

                st.write("Ils sont prometteurs dans le sens où ils ont pris connaissance de votre marque mais ne se sentent pas assez engagés, ne vous montrent pas assez de confiance pour dépenser plus. Ils sont encore frileux.")
                st.write("\n")
                st.write("\n")
                rfm5 = pd.merge(rfm, data, on="ID")
                prometteur = rfm5[rfm5["segment"] == "prometteur"]
                st.write(prometteur)

                # Convertissez le DataFrame en un fichier Excel
                a = prometteur
                excel_buffer = io.BytesIO()
                excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
                a.to_excel(excel_writer, index=False)
                excel_writer.save()
                excel_buffer.seek(0)

                # Téléchargez le fichier Excel
                st.download_button(
                    label="Download",
                    data=excel_buffer,
                    file_name="prometteur.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            if f == True :
                st.markdown("""<style>.marquee {width: 100%; overflow: hidden; white-space: nowrap;}.marquee p {display: inline-block;animation: marquee 10s linear infinite;font-size: 20px;color: red; align=center}@keyframes marquee {0% { transform: translateX(100%); }100% { transform: translateX(-100%); }}      
                                    </style><div class="marquee"><p>LES CLIENTS NECESSITANT UNE ATTENTION</p></div>""",
                            unsafe_allow_html=True)

                st.write("Le client qui mérite votre attention se situe au-dessus de la moyenne concernant à la fois la récence, la fréquence et le montant. Cependant, il n’a pas forcément acheté récemment (R faible).")
                st.write("\n")
                st.write("\n")
                rfm6 = pd.merge(rfm, data, on="ID")
                need_attention = rfm6[rfm6["segment"] == "need_attention"]
                st.write(need_attention)

                # Convertissez le DataFrame en un fichier Excel
                a = need_attention
                excel_buffer = io.BytesIO()
                excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
                a.to_excel(excel_writer, index=False)
                excel_writer.save()
                excel_buffer.seek(0)

                # Téléchargez le fichier Excel
                st.download_button(
                    label="Download",
                    data=excel_buffer,
                    file_name="need_attention.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            if g == True:
                st.markdown("""<style>.marquee {width: 100%; overflow: hidden; white-space: nowrap;}.marquee p {display: inline-block;animation: marquee 10s linear infinite;font-size: 20px;color: red; align=center}@keyframes marquee {0% { transform: translateX(100%); }100% { transform: translateX(-100%); }}      
                                    </style><div class="marquee"><p>LES CLIENTS SUR LE POINT DE S'ENDORMIR</p></div>""",
                            unsafe_allow_html=True)

                st.write("le client sur le point de s’endormir se situe en dessous de la moyenne, aussi bien en termes de récence, de fréquence que de montant. Si vous ne faites rien pour le réactiver, vous risquez grandement de le perdre définitivement.")
                st.write("\n")
                st.write("\n")
                rfm7 = pd.merge(rfm, data, on="ID")
                about_to_sleep = rfm7[rfm7["segment"] == "about_to_sleep"]
                st.write(about_to_sleep)

                # Convertissez le DataFrame en un fichier Excel
                a = about_to_sleep
                excel_buffer = io.BytesIO()
                excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
                a.to_excel(excel_writer, index=False)
                excel_writer.save()
                excel_buffer.seek(0)

                # Téléchargez le fichier Excel
                st.download_button(
                    label="Download",
                    data=excel_buffer,
                    file_name="about_to_sleep.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            if h == True:
                st.markdown("""<style>.marquee {width: 100%; overflow: hidden; white-space: nowrap;}.marquee p {display: inline-block;animation: marquee 10s linear infinite;font-size: 20px;color: red; align=center}@keyframes marquee {0% { transform: translateX(100%); }100% { transform: translateX(-100%); }}      
                                    </style><div class="marquee"><p>LES CLIENTS A RISQUES</p></div>""",
                            unsafe_allow_html=True)

                st.write("Le client à risque est un ancien très bon client, c’est-à-dire qu’il a dépensé beaucoup d’argent dans votre marque (M élevé) et régulièrement (F élevé). Mais, tout cela était il y a bien longtemps (R faible). Ce genre de clients a beaucoup de potentiel, c’est pourquoi il faut vous évertuer à les faire revenir")
                st.write("\n")
                st.write("\n")
                rfm8 = pd.merge(rfm, data, on="ID")
                at_Risk = rfm8[rfm8["segment"] == "at_Risk"]
                st.write(at_Risk)

                # Convertissez le DataFrame en un fichier Excel
                a = at_Risk
                excel_buffer = io.BytesIO()
                excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
                a.to_excel(excel_writer, index=False)
                excel_writer.save()
                excel_buffer.seek(0)

                # Téléchargez le fichier Excel
                st.download_button(
                    label="Download",
                    data=excel_buffer,
                    file_name="at_Risk.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            if i == True:
                st.markdown("""<style>.marquee {width: 100%; overflow: hidden; white-space: nowrap;}.marquee p {display: inline-block;animation: marquee 10s linear infinite;font-size: 20px;color: red; align=center}@keyframes marquee {0% { transform: translateX(100%); }100% { transform: translateX(-100%); }}      
                                    </style><div class="marquee"><p>LES CLIENTS A NE PAS PERDRE</p></div>""",
                            unsafe_allow_html=True)

                st.write("Il y a des clients que vous pouvez vous permettre de perdre, comme ceux qui n’ont jamais montré de signe de fidélité, ou ceux qui sont très volatils dans leurs choix et montrent peu de cohérence dans leurs achats. Les clients que vous ne pouvez pas vous permettre de perdre sont ceux ayant acheté souvent (F élevé) et ayant effectué les plus gros achats (M élevé). Mais cela fait un moment qu’ils n’ont plus racheté (R faible).")
                st.write("\n")
                st.write("\n")
                rfm9 = pd.merge(rfm, data, on="ID")
                cant_loose = rfm9[rfm9["segment"] == "cant_loose"]
                st.write(cant_loose)

                # Convertissez le DataFrame en un fichier Excel
                a = cant_loose
                excel_buffer = io.BytesIO()
                excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
                a.to_excel(excel_writer, index=False)
                excel_writer.save()
                excel_buffer.seek(0)

                # Téléchargez le fichier Excel
                st.download_button(
                    label="Download",
                    data=excel_buffer,
                    file_name="cant_loose.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            if j == True:
                st.markdown("""<style>.marquee {width: 100%; overflow: hidden; white-space: nowrap;}.marquee p {display: inline-block;animation: marquee 10s linear infinite;font-size: 20px;color: red; align=center}@keyframes marquee {0% { transform: translateX(100%); }100% { transform: translateX(-100%); }}      
                                    </style><div class="marquee"><p>LES CLIENTS EN HIBERNATION</p></div>""",
                            unsafe_allow_html=True)

                st.write("Le client en hibernation est, comme son nom l’indique, passif : son dernier achat remonte à un certain temps (F faible), il ne dépense plus beaucoup (M faible) et de moins en moins (F faible). Bref, il est quasiment perdu !")
                st.write("\n")
                st.write("\n")
                rfm10 = pd.merge(rfm, data, on="ID")
                hibernating = rfm10[rfm10["segment"] == "hibernating"]
                st.write(hibernating)

                # Convertissez le DataFrame en un fichier Excel
                a = hibernating
                excel_buffer = io.BytesIO()
                excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
                a.to_excel(excel_writer, index=False)
                excel_writer.save()
                excel_buffer.seek(0)

                # Téléchargez le fichier Excel
                st.download_button(
                    label="Download",
                    data=excel_buffer,
                    file_name="hibernating.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
########################################################################
        if add_selectbox == 'Segmentation KMEANS':
            st.subheader('CHARGEMENT DU FICHIER EXCEL/CSV🔧')
            file_upload = st.file_uploader("Charger le fichier Excel/csv", type=["xlsx", "csv"], key=3)
            # chargement du fichier csv pour le traitement
            if file_upload is not None:
                data = pd.read_excel(file_upload)
            if 'data' in locals():
                # Si les données existent, informer l'utilisateur avec un message
                st.success("Base de données chargées avec succès!")
                st.write("")
            else:
                # Si les données n'existent pas, affichez un avertissement
                st.warning("Chargez d'abord le fichier Excel/csv.")

            st.subheader("SEGMENTATION DEMOGRAPHIQUE")
            data['Date de naissance'] = pd.to_datetime(data['Date de naissance'], format='%Y-%m-%d')
            now = datetime.now()
            data['Age'] = (now - data['Date de naissance']).astype('<m8[Y]')
            data['Adresse '] = data['Adresse '].astype(str)
            data['Ville '] = data['Ville '].astype(str)
            data['Notes '] = data['Notes '].astype(str)
            data[["Adresse ", "Notes ", "Ville ", "Age"]].fillna(0)
            data["Age"] = data["Age"].replace(np.nan, data["Age"].mean())
            data['Adresse '] = data['Adresse '].fillna(data['Adresse '].mode()[0])
            data['Ville '] = data['Ville '].fillna(data['Ville '].mode()[0])
            label_encoder = preprocessing.LabelEncoder()
            data['Adresse '] = label_encoder.fit_transform(data['Adresse '])
            data['Ville '] = label_encoder.fit_transform(data['Ville '])
            data["Notes "] = label_encoder.fit_transform(data["Notes "])
            data_demography = data[["ID","MontantTotal", "Age"]]
            data_geography = data[["ID","MontantTotal", "Adresse "]]
            st.write("")
        #     parameter = st.selectbox("Choisir le segment RFM à analyser",
        #                          ["Clients champions", "Clients fidèles", "Clients potentiellement fidèles", "Nouveaux clients", "Clients prometteurs", "Clients méritant une attention", "Clients sur le point de s’endormir", "Clients à risque", "Clients à ne pas perdre", "Clients en hibernation"])
        #     st.write("")
        #     parameter = st.radio("Choisir le paramètre de segmentation Kmeans",
        #                          ["Démographie", "Géographie"])
        #
        #     if parameter == "Démographie":
        #     st.markdown("SEGMENTATION DEMOGRAPHIQUE")
            scaler = StandardScaler()
            data_demography_standard = scaler.fit_transform(data_demography)
            data_demography_standard = pd.DataFrame(data=data_demography_standard,
                                                         columns=data_demography.columns)
            # st.write(champions_demography_standard)
            st.markdown("CHOIX DU NOMBRE DE CLUSTER")
            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans.fit(data_demography_standard)
                wcss.append(kmeans.inertia_)
            # Créer un objet figure explicitement
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(range(1, 11), wcss, marker='o', linestyle='-.', color='blue')
                ax.set_xlabel('Number of Clusters')
                ax.set_ylabel('WCSS')
                ax.set_title('K-means Clustering')
                # Afficher le graphique dans Streamlit en passant l'objet figure
                st.pyplot(fig)
            st.write("")
            st.write("-------")
            st.markdown("ENTRAINEMENT DU MODELE")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                v = st.button("Segmenter (1 cluster)", key=310)
            with col2:
                w = st.button("Segmenter (2 clusters)", key=311)
            with col3:
                x = st.button("Segmenter (3 clusters)", key=312)
            with col4:
                y = st.button("Segmenter (4 clusters)", key=313)
            with col5:
                z = st.button("Segmenter (5 clusters)", key=314)

            if v:
                kmeans = KMeans(n_clusters=1, init='k-means++', random_state=42)
                kmeans.fit(data_demography_standard)
                data_demography = pd.DataFrame(data_demography, index=data_demography.index,
                                                    columns=data_demography.columns)
                data_demography["clusters"] = kmeans.labels_
                data_demography = data_demography.merge(data[['ID', 'NomPrénom', 'Age', 'MontantTotal']], on='ID', how='left')
                st.write(data_demography)
                # Convertissez le DataFrame en un fichier Excel
                a = data_demography
                excel_buffer = io.BytesIO()
                excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
                a.to_excel(excel_writer, index=False)
                excel_writer.save()
                excel_buffer.seek(0)

                # Téléchargez le fichier Excel
                st.download_button(
                    label="Download",
                    data=excel_buffer,
                    file_name="segmentation_démographique.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                # Créez une figure Matplotlib
                fig = plt.figure(figsize=(8, 5))
                sns.barplot(x='clusters', y='Age_x', data=data_demography, ci=None)
                plt.xlabel("Cluster")
                plt.ylabel("Age")

                # Affichez la figure dans Streamlit en utilisant st.pyplot
                st.pyplot(fig)
            if w:
                kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
                kmeans.fit(data_demography_standard)
                champions_demography = pd.DataFrame(data_demography, index=data_demography.index,
                                                    columns=data_demography.columns)
                data_demography["clusters"] = kmeans.labels_
                data_demography = data_demography.merge(data[['ID', 'NomPrénom', 'Age', 'MontantTotal']], on='ID',how='left')
                st.write(data_demography)
                # Convertissez le DataFrame en un fichier Excel
                a = data_demography
                excel_buffer = io.BytesIO()
                excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
                a.to_excel(excel_writer, index=False)
                excel_writer.save()
                excel_buffer.seek(0)

                # Téléchargez le fichier Excel
                st.download_button(
                    label="Download",
                    data=excel_buffer,
                    file_name="segmentation_démographique.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                # Créez une figure Matplotlib
                fig = plt.figure(figsize=(8, 5))
                sns.barplot(x='clusters', y='Age_x', data=data_demography, ci=None)
                plt.xlabel("Cluster")
                plt.ylabel("Age")

                # Affichez la figure dans Streamlit en utilisant st.pyplot
                st.pyplot(fig)
            if x:
                kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
                kmeans.fit(data_demography_standard)
                data_demography = pd.DataFrame(data_demography, index=data_demography.index,
                                                    columns=data_demography.columns)
                data_demography["clusters"] = kmeans.labels_
                data_demography = data_demography.merge(data[['ID', 'NomPrénom', 'Age', 'MontantTotal']], on='ID',how='left')
                st.write(data_demography)
                # Convertissez le DataFrame en un fichier Excel
                a = data_demography
                excel_buffer = io.BytesIO()
                excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
                a.to_excel(excel_writer, index=False)
                excel_writer.save()
                excel_buffer.seek(0)

                # Téléchargez le fichier Excel
                st.download_button(
                    label="Download",
                    data=excel_buffer,
                    file_name="segmentation_démographique.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                # Créez une figure Matplotlib
                fig = plt.figure(figsize=(8, 5))
                sns.barplot(x='clusters', y='Age_x', data=data_demography, ci=None)
                plt.xlabel("Cluster")
                plt.ylabel("Age")

                # Affichez la figure dans Streamlit en utilisant st.pyplot
                st.pyplot(fig)
            if y:
                kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
                kmeans.fit(data_demography_standard)
                data_demography = pd.DataFrame(data_demography, index=data_demography.index,
                                                    columns=data_demography.columns)
                data_demography["clusters"] = kmeans.labels_
                data_demography = data_demography.merge(data[['ID', 'NomPrénom', 'Age', 'MontantTotal']], on='ID', how='left')
                st.write(data_demography)
                # Convertissez le DataFrame en un fichier Excel
                a = data_demography
                excel_buffer = io.BytesIO()
                excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
                a.to_excel(excel_writer, index=False)
                excel_writer.save()
                excel_buffer.seek(0)

                # Téléchargez le fichier Excel
                st.download_button(
                    label="Download",
                    data=excel_buffer,
                    file_name="segmentation_démographique.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                # Créez une figure Matplotlib
                fig = plt.figure(figsize=(8, 5))
                sns.barplot(x='clusters', y='Age_x', data=data_demography, ci=None)
                plt.xlabel("Cluster")
                plt.ylabel("Age")

                # Affichez la figure dans Streamlit en utilisant st.pyplot
                st.pyplot(fig)
            if z :
                kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
                kmeans.fit(data_demography_standard)
                data_demography = pd.DataFrame(data_demography, index=data_demography.index,
                                                    columns=data_demography.columns)
                data_demography["clusters"] = kmeans.labels_
                data_demography = data_demography.merge(data[['ID', 'NomPrénom', 'Age', 'MontantTotal']], on='ID', how='left')
                st.write(data_demography)
                # Convertissez le DataFrame en un fichier Excel
                a = data_demography
                excel_buffer = io.BytesIO()
                excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
                a.to_excel(excel_writer, index=False)
                excel_writer.save()
                excel_buffer.seek(0)

                # Téléchargez le fichier Excel
                st.download_button(
                    label="Download",
                    data=excel_buffer,
                    file_name="segmentation_démographique.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                # Créez une figure Matplotlib
                fig = plt.figure(figsize=(8, 5))
                sns.barplot(x='clusters', y='Age_x', data=data_demography, ci=None)
                plt.xlabel("Cluster")
                plt.ylabel("Age")
                # Affichez la figure dans Streamlit en utilisant st.pyplot
                st.pyplot(fig)
        #             ################################
        #
        #     if parameter == "Géographie":
        #         st.markdown("SEGMENTATION GEOGRAPHIQUE")
        #         scaler = StandardScaler()
        #         data_geography_standard = scaler.fit_transform(data_geography)
        #         data_geography_standard = pd.DataFrame(data=data_geography_standard,
        #                                                 columns=data_geography.columns)
        #         # st.write(champions_demography_standard)
        #         st.markdown("CHOIX DU NOMBRE DE CLUSTER")
        #         wcss = []
        #         for i in range(1, 11):
        #             kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        #             kmeans.fit(data_geography_standard)
        #             wcss.append(kmeans.inertia_)
        #         # Créer un objet figure explicitement
        #         col1, col2 = st.columns(2)
        #         with col1:
        #             fig, ax = plt.subplots(figsize=(5, 4))
        #             ax.plot(range(1, 11), wcss, marker='o', linestyle='-.', color='blue')
        #             ax.set_xlabel('Number of Clusters')
        #             ax.set_ylabel('WCSS')
        #             ax.set_title('K-means Clustering')
        #             # Afficher le graphique dans Streamlit en passant l'objet figure
        #             st.pyplot(fig)
        #         st.write("")
        #         st.write("-------")
        #         st.markdown("ENTRAINEMENT DU MODELE")
        #
        #         col1, col2, col3, col4, col5 = st.columns(5)
        #         with col1:
        #             v = st.button("Segmenter (1 cluster)", key=310)
        #         with col2:
        #             w = st.button("Segmenter (2 clusters)", key=311)
        #         with col3:
        #             x = st.button("Segmenter (3 clusters)", key=312)
        #         with col4:
        #             y = st.button("Segmenter (4 clusters)", key=313)
        #         with col5:
        #             z = st.button("Segmenter (5 clusters)", key=314)
        #
        #         if v:
        #             kmeans = KMeans(n_clusters=1, init='k-means++', random_state=42)
        #             kmeans.fit(data_geography_standard)
        #             data_geography = pd.DataFrame(data_geography, index=data_geography.index,
        #                                            columns=data_geography.columns)
        #             data_geography["clusters"] = kmeans.labels_
        #             data_geography = data_geography.merge(data[['ID', 'NomPrénom', 'Age', 'MontantTotal']],
        #                                                     on='ID', how='left')
        #             st.write(data_geography)
        #             # Convertissez le DataFrame en un fichier Excel
        #             a = data_geography
        #             excel_buffer = io.BytesIO()
        #             excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
        #             a.to_excel(excel_writer, index=False)
        #             excel_writer.save()
        #             excel_buffer.seek(0)
        #
        #             # Téléchargez le fichier Excel
        #             st.download_button(
        #                 label="Download",
        #                 data=excel_buffer,
        #                 file_name="segmentation_géographique.xlsx",
        #                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        #             )
        #             # Créez une figure Matplotlib
        #             fig = plt.figure(figsize=(8, 5))
        #             sns.barplot(x='clusters', y='Adresse ', data=data_geography, ci=None)
        #             plt.xlabel("Cluster")
        #             plt.ylabel("Adresse")
        #
        #             # Affichez la figure dans Streamlit en utilisant st.pyplot
        #             st.pyplot(fig)
        #         if w:
        #             kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
        #             kmeans.fit(data_geography_standard)
        #             champions_demography = pd.DataFrame(data_geography, index=data_geography.index,
        #                                                 columns=data_geography.columns)
        #             data_geography["clusters"] = kmeans.labels_
        #             data_geography = data_geography.merge(data[['ID', 'NomPrénom', 'Age', 'MontantTotal']],
        #                                                     on='ID', how='left')
        #             st.write(data_geography)
        #             # Convertissez le DataFrame en un fichier Excel
        #             a = data_geography
        #             excel_buffer = io.BytesIO()
        #             excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
        #             a.to_excel(excel_writer, index=False)
        #             excel_writer.save()
        #             excel_buffer.seek(0)
        #
        #             # Téléchargez le fichier Excel
        #             st.download_button(
        #                 label="Download",
        #                 data=excel_buffer,
        #                 file_name="segmentation_géographique.xlsx",
        #                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        #             )
        #             # Créez une figure Matplotlib
        #             fig = plt.figure(figsize=(8, 5))
        #             sns.barplot(x='clusters', y='Adresse ', data=data_geography, ci=None)
        #             plt.xlabel("Cluster")
        #             plt.ylabel("Adresse")
        #
        #             # Affichez la figure dans Streamlit en utilisant st.pyplot
        #             st.pyplot(fig)
        #         if x:
        #             kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
        #             kmeans.fit(data_geography_standard)
        #             data_geography = pd.DataFrame(data_geography, index=data_geography.index,
        #                                            columns=data_geography.columns)
        #             data_geography["clusters"] = kmeans.labels_
        #             data_geography = data_geography.merge(data[['ID', 'NomPrénom', 'Age', 'MontantTotal']],
        #                                                     on='ID', how='left')
        #             st.write(data_geography)
        #             # Convertissez le DataFrame en un fichier Excel
        #             a = data_geography
        #             excel_buffer = io.BytesIO()
        #             excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
        #             a.to_excel(excel_writer, index=False)
        #             excel_writer.save()
        #             excel_buffer.seek(0)
        #
        #             # Téléchargez le fichier Excel
        #             st.download_button(
        #                 label="Download",
        #                 data=excel_buffer,
        #                 file_name="segmentation_géographique.xlsx",
        #                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        #             )
        #             # Créez une figure Matplotlib
        #             fig = plt.figure(figsize=(8, 5))
        #             sns.barplot(x='clusters', y='Adresse ', data=data_geography, ci=None)
        #             plt.xlabel("Cluster")
        #             plt.ylabel("Adresse")
        #
        #             # Affichez la figure dans Streamlit en utilisant st.pyplot
        #             st.pyplot(fig)
        #         if y:
        #             kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
        #             kmeans.fit(data_geography_standard)
        #             data_geography = pd.DataFrame(data_geography, index=data_geography.index,
        #                                            columns=data_geography.columns)
        #             data_geography["clusters"] = kmeans.labels_
        #             data_geography = data_geography.merge(data[['ID', 'NomPrénom', 'Age', 'MontantTotal']],
        #                                                     on='ID', how='left')
        #             st.write(data_geography)
        #             # Convertissez le DataFrame en un fichier Excel
        #             a = data_geography
        #             excel_buffer = io.BytesIO()
        #             excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
        #             a.to_excel(excel_writer, index=False)
        #             excel_writer.save()
        #             excel_buffer.seek(0)
        #
        #             # Téléchargez le fichier Excel
        #             st.download_button(
        #                 label="Download",
        #                 data=excel_buffer,
        #                 file_name="segmentation_géographique.xlsx",
        #                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        #             )
        #             # Créez une figure Matplotlib
        #             fig = plt.figure(figsize=(8, 5))
        #             sns.barplot(x='clusters', y='Adresse ', data=data_geography, ci=None)
        #             plt.xlabel("Cluster")
        #             plt.ylabel("Adresse")
        #
        #             # Affichez la figure dans Streamlit en utilisant st.pyplot
        #             st.pyplot(fig)
        #         if z:
        #             kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
        #             kmeans.fit(data_geography_standard)
        #             data_geography = pd.DataFrame(data_geography, index=data_geography.index,
        #                                                 columns=data_geography.columns)
        #             data_geography["clusters"] = kmeans.labels_
        #             data_geography = data_geography.merge(data[['ID', 'NomPrénom', 'Age', 'MontantTotal']],
        #                                                     on='ID', how='left')
        #             st.write(data_geography)
        #             # Convertissez le DataFrame en un fichier Excel
        #             a = data_geography
        #             excel_buffer = io.BytesIO()
        #             excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
        #             a.to_excel(excel_writer, index=False)
        #             excel_writer.save()
        #             excel_buffer.seek(0)
        #
        #             # Téléchargez le fichier Excel
        #             st.download_button(
        #                 label="Download",
        #                 data=excel_buffer,
        #                 file_name="segmentation_géographique.xlsx",
        #                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        #             )
        #             # Créez une figure Matplotlib
        #             fig = plt.figure(figsize=(8, 5))
        #             sns.barplot(x='clusters', y='Adresse ', data=data_geography, ci=None)
        #             plt.xlabel("Cluster")
        #             plt.ylabel("Adresse")
        #             # Affichez la figure dans Streamlit en utilisant st.pyplot
        #             st.pyplot(fig)





        ################################################################################
        if add_selectbox == "Rubrique anniversaire":
            st.markdown("""<style>.marquee {width: 100%; overflow: hidden; white-space: nowrap;}.marquee p {display: inline-block;animation: marquee 10s linear infinite;font-size: 20px;color: red; align=center}@keyframes marquee {0% { transform: translateX(100%); }100% { transform: translateX(-100%); }}      
                                </style><div class="marquee"><p>PROGRAMME DE FIDELISATION</p></div>""",unsafe_allow_html=True)
            st.subheader('1. CHARGEMENT DU FICHIER EXCEL/CSV🔧')
            # Charger le fichier Excel si nécessaire
            uploaded_file = st.file_uploader("Uploader un fichier Excel", type=["xlsx"])
            if uploaded_file is not None:
                data = pd.read_excel(uploaded_file)

            # Vérifier si les données existent
            if 'data' in locals():
                # Si les données existent, effectuez la transformation
                data['Date de naissance'] = pd.to_datetime(data['Date de naissance'], format='%Y-%m-%d')
                st.success("Base de données chargées avec succès!")
            else:
                # Si les données n'existent pas, affichez un avertissement
                st.warning("Chargez d'abord le fichier Excel/csv.")

            # Fonction pour calculer l'âge à partir de la date de naissance
            data['Date de naissance'] = pd.to_datetime(data['Date de naissance'], format='%Y-%m-%d')
            def calculer_age(date_naissance):
                aujourdhui = datetime.today()
                age = aujourdhui.year - date_naissance.year - (
                        (aujourdhui.month, aujourdhui.day) < (date_naissance.month, date_naissance.day))
                return age

            # Titre de l'application
            st.title("Vérification des anniversaires")
            # Création d'une liste pour stocker les informations des clients dont c'est l'anniversaire
            anniversaire_clients = []
            # Boucle à travers les clients du DataFrame pour vérifier les anniversaires
            for index, row in data.iterrows():
                nom = row["NomPrénom"]
                date_naissance = row["Date de naissance"]
                Numéro = row["Portable"]

                # Utilisation correcte de datetime
                aujourdhui = datetime.today()

                # Calcul de l'âge
                age = aujourdhui.year - date_naissance.year - (
                        (aujourdhui.month, aujourdhui.day) < (date_naissance.month, date_naissance.day))

                # Vérification si c'est l'anniversaire aujourd'hui
                if aujourdhui.month == date_naissance.month and aujourdhui.day == date_naissance.day:
                    anniversaire_clients.append({
                        "NomPrénom": nom,
                        "Date de naissance": date_naissance,
                        "Âge": age,
                        "Numéro de téléphone": Numéro

                    })

            # Création d'un DataFrame à partir de la liste des clients
            df_anniversaire = pd.DataFrame(anniversaire_clients)

            # Affichage du DataFrame contenant les informations des clients dont c'est l'anniversaire
            st.write("Clients fêtant leur anniversaire aujourd'hui :")
            st.dataframe(df_anniversaire)



    if add_selectbox == 'PREVISIONS DE VENTE' :
        add_selectbox = st.sidebar.selectbox("Choisir la stratégie", ("-----", "PREVISIONS EN LIGNE",  "PREVISIONS PAR LOTS"), key=10)

########
        if add_selectbox == "-----":
            st.markdown("""
                        <style>
                            /* Style du cadre */
                            .custom-frame {
                                border: 2px solid #0078d4;
                                padding: 20px;
                                border-radius: 10px;
                                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                            }
    
                            /* Style du titre */
                            .custom-title {
                                font-size: 36px;
                                font-weight: bold;
                                color: #0078d4;
                                text-align: center;
                                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
                            }
                        </style>
                    """, unsafe_allow_html=True)

            st.write("""
                        <div class="custom-frame">
                            <div class="custom-title">
                                PREVISIONS DE VENTE
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            #####

            st.write("")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("")
                st.write("")
                st.image("images/previsions1.jpg", width=405)
            with col2:
                st.write("")
                st.markdown("""
                                    <style>
                                        /* Style du conteneur */
                                        .container {
                                            background-color: #f5f5f5;
                                            padding: 20px;
                                            border-radius: 10px;
                                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                                        }
    
                                        /* Style du texte */
                                        .text {
                                            font-size: 18px;
                                            color: #333;
                                            line-height: 1.5;
                                        }
                                    </style>
                                """, unsafe_allow_html=True)
                st.write("""
                                    <div class="container">
                                        <p class="text">
                                            La prévision des ventes est une méthode qui consiste à estimer les ventes à venir en fonction des données passées et des études comparatives correspondant à notre secteur d’activité. La prévision des ventes est un processus qui permet à l'entreprise de générer les données nécessaires à la préparation des plans et des stratégies futures. \n La prévision des ventes permet donc à l'entreprise de fixer ses objectifs avec précision. Grâce à des objectifs précis et réalistes, aux équipes de ventes travaillent de manière plus efficace et contribuent au succès de l'entreprise.
                                        </p>
                                    </div>
                                """, unsafe_allow_html=True)

                with col3:
                    st.write("")
                    st.write("")
                    st.image("images/previsions2.jpg", width=400)

        #######

        if add_selectbox == "PREVISIONS EN LIGNE":
            st.header('CHARGEMENT DU MODELE🔧')
            file_upload2 = st.file_uploader("Charger le fichier PKL pour les prévisions", type=None)
            if file_upload2 is not None:
                modele = joblib.load(file_upload2)

            st.write("------")
            # st.subheader("PREVISIONS EN LIGNE")
            st.markdown(
                '''
                <style>
                    @keyframes blink {
                        0% { opacity: 1; color: red; }
                        50% { opacity: 1; color: blue; }
                        100% { opacity: 1; color: green; }
                    }

                    .blinking-text {
                        animation: blink 5s infinite;
                    }
                </style>
                ''', unsafe_allow_html=True
            )

            st.markdown(
                '''
                <h1 class="blinking-text">PREVISIONS EN LIGNE</h1>
                ''', unsafe_allow_html=True
            )
            st.success("Les scénarios de prédiction en ligne sont pour les cas où l'on veut générer des prévisions sur une base individuelle, en l'occurrence, une seule date")
            st.subheader("insertion de toutes les entrées requises pour la prévision")
            col1, col2, col3 = st.columns(3)
            with col1 :
                Date = st.date_input("Date")
                Date = pd.to_datetime(Date)
            with col2 :
                statut_ouverture_fermeture = st.selectbox("Choisir le statut de la journée", ("Jour ouvert","Jour fermé"))
                if statut_ouverture_fermeture == "Jour ouvert":
                    statut_ouverture_fermeture=1
                if statut_ouverture_fermeture == "Jour fermé":
                    statut_ouverture_fermeture=0
            with col3 :
                Evènement_périmètre = st.selectbox("Préciser la particularité du jour", ("ascension", "assomption", "fête du travail", "huit mars", "korite", "korite j-1", "korite j-2",
                            "korite j-3", "lundi de pentecote", "magal", "maouloud", "mariage", "new an", "noel", "noel j-1", "noel j-2", "noel j-3", "normal", "pâques",
                            "pentecote", "saint valentin", "tabaski", "tabaski j-1", "tabaski j-2", "tabaski j-3", "tamkharit", "toussaint"))

            # date = dt.datetime(date)
            weekday = calendar.day_name[Date.weekday()]
            # weekday = date.strftime("%A")
            Year = Date.year
            Month = Date.month
            Day = Date.day
            WeekOfYear = Date.strftime("%U")
            # WeekOfYear = date.isocalendar()[1]
            DayOfWeek = Date.weekday()
            WeekOfMonth = Date.day // 7 + 1
            if DayOfWeek >= 5:
                weekend = 1
            else :
                weekend = 0

            output = ""
            input_dict = {'Date': Date	, 'statut_ouverture_fermeture': statut_ouverture_fermeture,
                          'Evènement_périmètre': Evènement_périmètre,'weekday': weekday, 'Year': Year,
                          'Month': Month, 'Day': Day,'WeekOfYear': WeekOfYear, 'DayOfWeek': DayOfWeek,
                          'WeekOfMonth': WeekOfMonth, 'weekend': weekend
                          }
            st.write("--------")
            st.write("\n")
            st.write("LES DONNEES D'ENTREE")
            input_df = pd.DataFrame([input_dict])
            # input_df['Date'] = pd.to_datetime(input_df['Date'])
            st.write(input_df)
            # appel de la fonction predict quand le bouton est cliqué
            st.write("--------")
            if st.button("Prévoir"):
                output = modele.predict(input_df)
                output = str(output)
                st.success("Les prévisions sont évaluées à {}".format(output))


        if add_selectbox == "PREVISIONS PAR LOTS":
            st.markdown(
                '''
                <style>
                    @keyframes blink {
                        0% { opacity: 1; color: red; }
                        50% { opacity: 1; color: blue; }
                        100% { opacity: 1; color: green; }
                    }

                    .blinking-text {
                        animation: blink 5s infinite;
                    }
                </style>
                ''', unsafe_allow_html=True
            )

            st.markdown(
                '''
                <h1 class="blinking-text">PREVISIONS PAR LOTS</h1>
                ''', unsafe_allow_html=True
            )

            col1,col2 = st.columns(2)
            with col1 :
                st.success("La prédiction par lots est utile lorsque l'on souhaite générer des prédictions pour un ensemble d’observations en même temps, puis agir sur un certain pourcentage ou nombre d’observations.")
            st.subheader('CHARGEMENT DU MODELE🔧')
            col1, col2, col3 = st.columns(3)
            with col1:
                file_upload2 = st.file_uploader("Chargement du modèle", type=None)
                if file_upload2 is not None:
                    from pycaret.regression import *
                    modele = joblib.load(file_upload2)
            st.write("-----------")

            st.subheader("Previsions")
            # choice = st.radio("", ['Charger la période de prévisions', 'Définir la période de prévisions'])

            # if choice == 'Charger la période de prévisions':
            col1, col2, col3 = st.columns(3)
            with col1:
                file_upload = st.file_uploader("Chargement du fichier ", type=["xlsx", "csv"], key=35)
            if file_upload is not None:
                data_previsions = pd.read_excel(file_upload)
                if st.button("Afficher"):
                    st.write(data_previsions)
            st.write("---")

            if st.button("PREVOIR"):
                predictions = predict_model(estimator=modele, data=data_previsions)
                st.success("éffectué !")
                st.write(predictions)
                # Convertissez le DataFrame en un fichier Excel
                excel_buffer = io.BytesIO()
                excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
                predictions.to_excel(excel_writer, index=False)
                excel_writer.save()
                excel_buffer.seek(0)

                # Téléchargez le fichier Excel
                st.download_button(
                    label="Download",
                    data=excel_buffer,
                    file_name="Prévisions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # Créer un titre pour l'application
                st.title("Tableau de Bord")

                # Sélectionner l'année
                selected_year = st.selectbox("Sélectionner une Année", predictions['Date'].dt.year.unique())

                # Filtrer les données par année
                filtered_data = predictions[predictions['Date'].dt.year == selected_year]

                # Créer un graphique à barres par mois
                fig = px.bar(filtered_data, x=filtered_data['Date'].dt.month, y=filtered_data["prediction_label"],
                             labels={'x': 'Mois'}, title=f"Chiffre d'Affaires pour l'année {selected_year}")

                # Personnalisation du graphique
                fig.update_xaxes(type='category')
                fig.update_yaxes(title_text='Chiffre d\'Affaires')
                fig.update_traces(marker_color='blue')

                # Afficher le graphique dans l'application Streamlit
                st.plotly_chart(fig)

                # Créer un graphique à barres par année en utilisant Plotly Express
                fig = px.bar(predictions, x=predictions['Date'].dt.year, y=predictions["prediction_label"],
                             title='Évolution du CA par Année')
                fig.update_xaxes(title_text='Année')

                # Personnalisation du graphique
                fig.update_yaxes(title_text='Chiffre d\'Affaires')
                fig.update_traces(marker_color='blue')

                # Afficher le graphique dans l'application Streamlit
                st.plotly_chart(fig)












