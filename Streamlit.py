import streamlit as st
import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Charger les modèles et outils ---
@st.cache_resource
def load_resources():
    with open("model_ml.pkl", "rb") as f:
        model_ml = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    model_dl = load_model("dl_lstm_model.h5")

    return model_ml, tfidf, tokenizer, label_encoder, model_dl

model_ml, tfidf, tokenizer, label_encoder, model_dl = load_resources()

# --- Paramètres DL ---
max_len = 50

# --- Fonction de prédiction ML ---
def predict_with_ml(text):
    vec = tfidf.transform([text])
    pred = model_ml.predict(vec)
    label = label_encoder.inverse_transform(pred)[0]
    proba = model_ml.predict_proba(vec).max()
    return label, proba

# --- Fonction de prédiction DL ---
def predict_with_dl(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model_dl.predict(pad)
    label_index = np.argmax(pred)
    label = label_encoder.inverse_transform([label_index])[0]
    proba = np.max(pred)
    return label, proba

# --- Interface Streamlit ---
st.set_page_config(page_title="Chatbot Émotionnel", page_icon="🤖")
st.title("🤖 Chatbot Intelligent avec ML et DL")

model_choice = st.radio("Choisissez le modèle à utiliser :", ["Machine Learning", "Deep Learning"])

# Historique
if 'history' not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Votre message :")

if st.button("Envoyer") and user_input.strip() != "":
    if model_choice == "Machine Learning":
        prediction, confidence = predict_with_ml(user_input)
    else:
        prediction, confidence = predict_with_dl(user_input)

    response = f"🔍 Émotion détectée : **{prediction}** (Confiance : {confidence:.2f})"
    st.success(response)

    # Ajouter à l’historique
    st.session_state.history.append((user_input, prediction, confidence))

# Affichage de l’historique
if st.session_state.history:
    st.subheader("📜 Historique de la conversation")
    for i, (msg, intent, conf) in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"**Vous :** {msg}")
        st.markdown(f"**Bot :** {intent} (Confiance : {conf:.2f})")
        st.markdown("---")
