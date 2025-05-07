# -*- coding: utf-8 -*-
"""
sentiment_app.py
Application Streamlit pour entraîner des modèles de détection de sentiment sur un CSV ou prédire le sentiment d'un texte.
Inclut un modèle de deep learning (LSTM) avec visualisation graphique.
"""

# Importation des bibliothèques
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
import os
import unicodedata
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ressources NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialisation des outils
stop_words = set(stopwords.words('english'))
lemmatiseur = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocesser_texte(texte):
    texte = unicodedata.normalize('NFKD', texte).encode('ASCII', 'ignore').decode('utf-8').lower()
    tokens = word_tokenize(texte)
    tokens = [
        stemmer.stem(lemmatiseur.lemmatize(token))
        for token in tokens
        if token.isalnum() and token not in stop_words
    ]
    return ' '.join(tokens)

st.title("Détection de Sentiment avec Deep Learning (LSTM)")

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
                donnees = donnees.dropna(subset=["texte", "sentiment"])
                donnees["texte"] = donnees["texte"].astype(str)
                donnees["texte_pretraite"] = donnees["texte"].apply(preprocesser_texte)
                X = donnees["texte_pretraite"]
                y = donnees["sentiment"].astype('category')
                y_cat = y.cat.codes

                # Tokenizer
                tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
                tokenizer.fit_on_texts(X)
                X_seq = tokenizer.texts_to_sequences(X)
                X_pad = pad_sequences(X_seq, padding='post', maxlen=100)

                joblib.dump(tokenizer, "tokenizer.pkl")

                X_train, X_test, y_train, y_test = train_test_split(X_pad, y_cat, test_size=0.3, random_state=42)

                st.subheader("Modèle LSTM")
                model_lstm = Sequential([
                    Embedding(input_dim=5000, output_dim=64, input_length=100),
                    LSTM(64),
                    Dropout(0.5),
                    Dense(64, activation='relu'),
                    Dense(len(np.unique(y_cat)), activation='softmax')
                ])

                model_lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                history = model_lstm.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

                model_lstm.save("modele_lstm.h5")
                st.success("Modèle LSTM entraîné et sauvegardé !")

                # Visualisation
                st.subheader("Courbes d'entraînement")
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                ax[0].plot(history.history['loss'], label='Train')
                ax[0].plot(history.history['val_loss'], label='Validation')
                ax[0].set_title("Loss")
                ax[0].legend()

                ax[1].plot(history.history['accuracy'], label='Train')
                ax[1].plot(history.history['val_accuracy'], label='Validation')
                ax[1].set_title("Accuracy")
                ax[1].legend()

                st.pyplot(fig)

        except Exception as e:
            st.error(f"Erreur lors du traitement du CSV : {str(e)}")

else:
    texte_input = st.text_area("Entrez un texte pour prédire le sentiment :")

    if texte_input:
        try:
            model_lstm = load_model("modele_lstm.h5")
            tokenizer = joblib.load("tokenizer.pkl")

            texte_pretraite = preprocesser_texte(texte_input)
            seq = tokenizer.texts_to_sequences([texte_pretraite])
            pad = pad_sequences(seq, padding='post', maxlen=100)

            prediction = model_lstm.predict(pad)
            classe_predite = np.argmax(prediction)

            st.write(f"Prédiction LSTM : Classe {classe_predite}")

        except FileNotFoundError:
            st.error("Modèle ou tokenizer introuvable. Veuillez entraîner le modèle d'abord.")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {str(e)}")

st.info("Téléchargez un CSV pour entraîner un modèle ou entrez un texte pour obtenir une prédiction.")
