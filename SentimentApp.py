# -*- coding: utf-8 -*-
"""
sentiment_app.py
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
import unicodedata

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from deep_translator import GoogleTranslator

# Télécharger les ressources NLTK nécessaires
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialisation des outils NLP
stop_words = set(stopwords.words('english'))
lemmatiseur = WordNetLemmatizer()
stemmer = PorterStemmer()

# Fonction de prétraitement du texte
def preprocesser_texte(texte):
    # Traduction vers l'anglais si nécessaire
    try:
        texte = GoogleTranslator(source='auto', target='en').translate(texte)
    except Exception as e:
        st.error(f"Erreur de traduction automatique : {e}")
        return ""

    # Normalisation : suppression des accents et mise en minuscule
    texte = unicodedata.normalize('NFKD', texte).encode('ASCII', 'ignore').decode('utf-8').lower()

    if not isinstance(texte, str):
        texte = ""
        tokens = word_tokenize(texte.lower())


    # Suppression des stopwords, lemmatisation et stemming
    tokens = [
        stemmer.stem(lemmatiseur.lemmatize(token))
        for token in tokens
        if token.isalnum() and token not in stop_words
    ]

    return ' '.join(tokens)

# Titre de l'application
st.title("🧠 Détection de Sentiment Multilingue")

# Choix de l'utilisateur
option = st.radio("Choisissez une option :", ("Entraîner un modèle (CSV)", "Prédire un sentiment (Texte)"))

if option == "Entraîner un modèle (CSV)":
    uploaded_file = st.file_uploader("Téléchargez un fichier CSV (colonnes : texte, sentiment)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            donnees = pd.read_csv(uploaded_file)
            st.write(f"✅ Dataset chargé ! Dimensions : {donnees.shape}")

            if "texte" not in donnees.columns or "sentiment" not in donnees.columns:
                st.error("❌ Le CSV doit contenir les colonnes 'texte' et 'sentiment'.")
            else:
                # Nettoyage et prétraitement
                donnees = donnees.dropna(subset=["texte", "sentiment"])
                donnees["texte"] = donnees["texte"].astype(str)
                st.info("Prétraitement en cours...")
                donnees["texte_pretraite"] = donnees["texte"].apply(preprocesser_texte)
                X = donnees["texte_pretraite"]
                y = donnees["sentiment"]

                # Vectorisation
                vectoriseur = TfidfVectorizer(max_features=5000)
                X_vectorise = vectoriseur.fit_transform(X)
                joblib.dump(vectoriseur, 'vectoriseur.pkl')

                # Séparation des données
                X_train, X_test, y_train, y_test = train_test_split(X_vectorise, y, test_size=0.3, random_state=42)
                st.write(f"📊 Dimensions entraînement : {X_train.shape}")
                st.write(f"📊 Dimensions test : {X_test.shape}")

                # Régression Logistique
                st.subheader("🔍 Régression Logistique")
                modele_lr = LogisticRegression(max_iter=1000)
                modele_lr.fit(X_train, y_train)
                y_pred_lr = modele_lr.predict(X_test)
                st.write(f"- Accuracy : {accuracy_score(y_test, y_pred_lr):.2f}")
                st.write(f"- F1-score : {f1_score(y_test, y_pred_lr, average='weighted'):.2f}")
                st.text(classification_report(y_test, y_pred_lr))
                st.write("Matrice de confusion :")
                st.write(confusion_matrix(y_test, y_pred_lr))
                joblib.dump(modele_lr, 'modele_lr.pkl')

                # Naive Bayes
                st.subheader("🧪 Naive Bayes")
                modele_nb = MultinomialNB()
                modele_nb.fit(X_train, y_train)
                y_pred_nb = modele_nb.predict(X_test)
                st.write(f"- Accuracy : {accuracy_score(y_test, y_pred_nb):.2f}")
                st.write(f"- F1-score : {f1_score(y_test, y_pred_nb, average='weighted'):.2f}")
                st.text(classification_report(y_test, y_pred_nb))
                st.write("Matrice de confusion :")
                st.write(confusion_matrix(y_test, y_pred_nb))
                joblib.dump(modele_nb, 'modele_nb.pkl')

                # Random Forest
                st.subheader("🌲 Random Forest")
                modele_rf = RandomForestClassifier(random_state=42)
                modele_rf.fit(X_train, y_train)
                y_pred_rf = modele_rf.predict(X_test)
                st.write(f"- Accuracy : {accuracy_score(y_test, y_pred_rf):.2f}")
                st.write(f"- F1-score : {f1_score(y_test, y_pred_rf, average='weighted'):.2f}")
                st.text(classification_report(y_test, y_pred_rf))
                st.write("Matrice de confusion :")
                st.write(confusion_matrix(y_test, y_pred_rf))
                joblib.dump(modele_rf, 'modele_rf.pkl')

                # Optimisation Optuna
                st.subheader("⚙️ Optimisation Régression Logistique")
                def objectif(trial):
                    C = trial.suggest_float("C", 0.01, 10.0, log=True)
                    modele = LogisticRegression(C=C, max_iter=1000)
                    return cross_val_score(modele, X_train, y_train, cv=3, scoring='accuracy').mean()

                etude = optuna.create_study(direction="maximize")
                etude.optimize(objectif, n_trials=20)
                st.write(f"Meilleurs hyperparamètres : {etude.best_params}")
                st.write(f"Meilleur score (validation croisée) : {etude.best_value:.2f}")

        except Exception as e:
            st.error(f"Erreur lors du traitement du CSV : {str(e)}")

else:
    texte_input = st.text_area("🌍 Entrez un texte (toute langue prise en charge) :")

    if texte_input:
        try:
            modele_lr = joblib.load("modele_lr.pkl")
            modele_nb = joblib.load("modele_nb.pkl")
            modele_rf = joblib.load("modele_rf.pkl")
            vectoriseur = joblib.load("vectoriseur.pkl")

            texte_pretraite = preprocesser_texte(texte_input)
            texte_vectorise = vectoriseur.transform([texte_pretraite])

            pred_lr = modele_lr.predict(texte_vectorise)[0]
            pred_nb = modele_nb.predict(texte_vectorise)[0]
            pred_rf = modele_rf.predict(texte_vectorise)[0]

            st.success("✅ Résultats des Prédictions :")
            st.write(f"🔍 Régression Logistique : {pred_lr}")
            st.write(f"🧪 Naive Bayes : {pred_nb}")
            st.write(f"🌲 Random Forest : {pred_rf}")

        except FileNotFoundError:
            st.error("⚠️ Modèles ou vectoriseur introuvables. Veuillez d'abord entraîner les modèles avec un CSV.")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {str(e)}")

st.info("ℹ️ Téléchargez un CSV pour entraîner ou entrez un texte dans n'importe quelle langue pour prédire le sentiment.")
