import streamlit as st
import pandas as pd
import joblib
from feature_importance import show_feature_importance_page
from eda import show_eda
from model_classification import show_model_classification
from song_recommendation import show_song_recommendation

@st.cache_data
def load_data():
    df = pd.read_csv("D:/SEMESTER 6/PENGGALIAN DATA/DASHBOARD/DATASET/dataset.csv")
    return df

df = load_data()

model = joblib.load("D:/SEMESTER 6/PENGGALIAN DATA/DASHBOARD/FILE_PKL/model_spotify.pkl")
feature_names = [
    "explicit", "danceability", "energy", "mode", "speechiness",
    "instrumentalness", "liveness", "valence", "tempo", "time_signature"
]

st.sidebar.title("Spotify Dashboard")
page = st.sidebar.radio("Pilih Halaman", ["EDA", "Model Klasifikasi", "Rekomendasi Lagu", "Feature Importance"])

if page == "EDA":
    show_eda(df)
elif page == "Model Klasifikasi":
    show_model_classification(df)
elif page == "Feature Importance":
    show_feature_importance_page(df, model, feature_names)
elif page == "Rekomendasi Lagu":
    show_song_recommendation(df)
else:
    st.title(f"Halaman {page} sedang dalam pengembangan")
