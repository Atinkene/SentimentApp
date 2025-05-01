# -*- coding: utf-8 -*-
"""
sentiment_app.ipynb
Application Streamlit pour entraîner des modèles de détection de sentiment sur un CSV ou prédire le sentiment d'un texte.
"""

# Importation des bibliothèques
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import optuna
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# Vérification des ressources NLTK pré-téléchargées
def verifier_nltk_ressources():
    try:
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        return True
    except LookupError:
        return False

# Arrêter si les ressources sont absentes
if not verifier_nltk_ressources():
    st.error("Ressources NLTK manquantes. Exécution du script suivant pour les télécharger :")
    st.code("""
    import nltk
    nltk.download('tokenizers/punkt_tab')
    nltk.download('corpora/stopwords')
    nltk.download('corpora/wordnet')
    """)
    nltk.download('tokenizers/punkt_tab')
    nltk.download('corpora/stopwords')
    nltk.download('corpora/wordnet')
    # st.stop()

# Fonction de prétraitement du texte 
def preprocesser_texte(texte):
    stop_words = set(stopwords.words('english'))
    lemmatiseur = WordNetLemmatizer()
    tokens = word_tokenize(texte.lower())
    tokens = [lemmatiseur.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(tokens)

# Titre de l'application
st.title("Détection de Sentiment")

# Widget pour télécharger un fichier ou entrer du texte
option = st.radio("Choisissez une option :", ("Entraîner un modèle (CSV)", "Prédire un sentiment (Texte)"))

if option == "Entraîner un modèle (CSV)":
    uploaded_file = st.file_uploader("Téléchargez un fichier CSV (colonnes : texte, sentiment)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            donnees = pd.read_csv(uploaded_file)
            st.write(f"Dataset chargé ! Dimensions : {donnees.shape}")

            if "texte" not in donnees.columns or "sentiment" not in donnees.columns:
                st.error("Le CSV doit contenir les colonnes 'texte' et 'sentiment'.")
            else:
                # Prétraitement
                donnees['texte_pretraite'] = donnees['texte'].apply(preprocesser_texte)
                X = donnees['texte_pretraite']
                y = donnees['sentiment']

                # Vectorisation
                vectoriseur = TfidfVectorizer(max_features=5000)
                X_vectorise = vectoriseur.fit_transform(X)
                joblib.dump(vectoriseur, 'vectoriseur.pkl')

                # Séparation des données
                X_train, X_test, y_train, y_test = train_test_split(X_vectorise, y, test_size=0.3, random_state=42)
                st.write(f"Dimensions entraînement : {X_train.shape}")
                st.write(f"Dimensions test : {X_test.shape}")

                # Modèle Logistic Regression
                st.subheader("Régression Logistique")
                modele_lr = LogisticRegression(max_iter=1000)
                modele_lr.fit(X_train, y_train)
                predictions_lr = modele_lr.predict(X_test)
                precision_lr = accuracy_score(y_test, predictions_lr)
                st.write(f"Précision : {precision_lr:.2f}")
                joblib.dump(modele_lr, 'modele_lr.pkl')

                # Modèle Naive Bayes
                st.subheader("Naive Bayes")
                modele_nb = MultinomialNB()
                modele_nb.fit(X_train, y_train)
                predictions_nb = modele_nb.predict(X_test)
                precision_nb = accuracy_score(y_test, predictions_nb)
                st.write(f"Précision : {precision_nb:.2f}")
                joblib.dump(modele_nb, 'modele_nb.pkl')

                # Modèle Random Forest
                st.subheader("Random Forest")
                modele_rf = RandomForestClassifier(random_state=42)
                modele_rf.fit(X_train, y_train)
                predictions_rf = modele_rf.predict(X_test)
                precision_rf = accuracy_score(y_test, predictions_rf)
                st.write(f"Précision : {precision_rf:.2f}")
                joblib.dump(modele_rf, 'modele_rf.pkl')

                # Optimisation Optuna (Logistic Regression)
                st.subheader("Optimisation Régression Logistique")
                def objectif(trial):
                    C = trial.suggest_float('C', 0.01, 10.0, log=True)
                    modele = LogisticRegression(C=C, max_iter=1000)
                    return cross_val_score(modele, X_train, y_train, cv=3, scoring='accuracy').mean()

                etude = optuna.create_study(direction='maximize')
                etude.optimize(objectif, n_trials=20)
                st.write(f"Meilleurs hyperparamètres : {etude.best_params}")
                st.write(f"Meilleur score : {etude.best_value:.2f}")

        except Exception as e:
            st.error(f"Erreur lors du traitement du CSV : {str(e)}")

else:
    texte_input = st.text_area("Entrez un texte pour prédire le sentiment :")
    
    if texte_input:
        try:
            # Charger les modèles et le vectoriseur
            modele_lr = joblib.load('modele_lr.pkl')
            modele_nb = joblib.load('modele_nb.pkl')
            modele_rf = joblib.load('modele_rf.pkl')
            vectoriseur = joblib.load('vectoriseur.pkl')

            # Prétraitement et vectorisation
            texte_pretraite = preprocesser_texte(texte_input)
            texte_vectorise = vectoriseur.transform([texte_pretraite])

            # Prédictions
            pred_lr = modele_lr.predict(texte_vectorise)[0]
            pred_nb = modele_nb.predict(texte_vectorise)[0]
            pred_rf = modele_rf.predict(texte_vectorise)[0]

            st.write("Résultats des Prédictions :")
            st.write(f"Régression Logistique : {pred_lr}")
            st.write(f"Naive Bayes : {pred_nb}")
            st.write(f"Random Forest : {pred_rf}")

        except FileNotFoundError:
            st.error("Modèles ou vectoriseur introuvables. Veuillez d'abord entraîner les modèles avec un CSV.")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {str(e)}")

st.info("Téléchargez un CSV pour entraîner ou entrez un texte pour prédire le sentiment.")