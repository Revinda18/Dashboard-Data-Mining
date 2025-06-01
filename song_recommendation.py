import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_model_scaler():
    model = joblib.load("D:/SEMESTER 6/PENGGALIAN DATA/DASHBOARD/FILE_PKL/model_spotify.pkl")
    scaler = joblib.load("D:/SEMESTER 6/PENGGALIAN DATA/DASHBOARD/FILE_PKL/scaler_spotify.pkl")
    feature_list = joblib.load("D:/SEMESTER 6/PENGGALIAN DATA/DASHBOARD/FILE_PKL/fitur_spotify.pkl")
    return model, scaler, feature_list

def align_features(df_input, feature_list):
    df_new = df_input.copy()
    for col in feature_list:
        if col not in df_new.columns:
            df_new[col] = 0
    df_new = df_new[feature_list]
    return df_new

def create_popularity_label(df, threshold=50):
    df['popularity_class'] = df['popularity'].apply(lambda x: 1 if x >= threshold else 0)
    return df

def show_song_recommendation(df):
    st.title("🎵 Rekomendasi Lagu Untuk Kamu")

    model, scaler, feature_list = load_model_scaler()

    if 'popularity_class' not in df.columns:
        df = create_popularity_label(df)

    song_options = df[['track_name', 'artists']].drop_duplicates()
    song_options['display'] = song_options['track_name'] + " - " + song_options['artists']
    
    selected_song = st.selectbox("Select a song from the list:", song_options['display'].tolist())

    if selected_song:
        track_name = song_options[song_options['display'] == selected_song]['track_name'].values[0]
        artist_name = song_options[song_options['display'] == selected_song]['artists'].values[0]

        selected_song_data = df[(df['track_name'] == track_name) & (df['artists'] == artist_name)]

        st.write(f"Selected Song:")
        st.dataframe(selected_song_data[['track_name', 'artists', 'popularity']].drop_duplicates())

        try:
            df_features = align_features(df, feature_list).astype(float)
            df_scaled = scaler.transform(df_features)

            predictions = model.predict(df_scaled)
            df['Prediction'] = predictions
            df['Prediction_Label'] = df['Prediction'].map({1: "Like", 0: "Dislike"})

            recommended_songs = df[df['Prediction_Label'] == "Like"].copy()

        except Exception as e:
            # Jika terjadi error, tampilkan pesan error (opsional, bisa juga dihilangkan)
            st.error(f"Gagal membuat rekomendasi: {e}")

    else:
        st.info("Silakan pilih lagu di atas untuk mendapatkan rekomendasi.")
