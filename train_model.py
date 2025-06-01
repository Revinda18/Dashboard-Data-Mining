import streamlit as st
import pandas as pd
import joblib
from eda import show_eda
from model_classification import show_model_classification
from feature_importance import show_feature_importance_page

@st.cache_data
def load_data():
    df = pd.read_pickle(r"D:\SEMESTER 6\PENGGALIAN DATA\DASHBOARD\FILE_PKL\dataset_final_normalisasi.pkl")
    return df

df = load_data()

# Load model, scaler, dan fitur yang sudah disimpan
model = joblib.load(r"D:\SEMESTER 6\PENGGALIAN DATA\DASHBOARD\FILE_PKL\model_spotify.pkl")
scaler = joblib.load(r"D:\SEMESTER 6\PENGGALIAN DATA\DASHBOARD\FILE_PKL\scaler_spotify.pkl")
feature_names = joblib.load(r"D:\SEMESTER 6\PENGGALIAN DATA\DASHBOARD\FILE_PKL\fitur_spotify.pkl")

st.sidebar.title("Spotify Dashboard")
page = st.sidebar.radio("Pilih Halaman", ["EDA", "Model Klasifikasi", "Rekomendasi Lagu", "Feature Importance"])

if page == "EDA":
    show_eda(df)
elif page == "Model Klasifikasi":
    show_model_classification(df)
elif page == "Feature Importance":
    show_feature_importance_page(df, model, feature_names)
else:
    st.title(f"Halaman {page} sedang dalam pengembangan")
