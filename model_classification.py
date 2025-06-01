import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

@st.cache_resource
def load_model_scaler():
    model = joblib.load("D:/SEMESTER 6/PENGGALIAN DATA/DASHBOARD/FILE_PKL/model_spotify.pkl")
    scaler = joblib.load("D:/SEMESTER 6/PENGGALIAN DATA/DASHBOARD/FILE_PKL/scaler_spotify.pkl")
    fitur_model = joblib.load("D:/SEMESTER 6/PENGGALIAN DATA/DASHBOARD/FILE_PKL/fitur_spotify.pkl")
    return model, scaler, fitur_model

# Fungsi untuk menyelaraskan input fitur dengan fitur saat training
def sesuaikan_fitur(df_input, fitur_model):
    df_baru = df_input.copy()
    for kolom in fitur_model:
        if kolom not in df_baru.columns:
            df_baru[kolom] = 0
    df_baru = df_baru[fitur_model]
    return df_baru

# Fungsi untuk membuat label klasifikasi dari popularity
def buat_label_popularity(df, threshold=50):
    df['popularity_class'] = df['popularity'].apply(lambda x: 1 if x >= threshold else 0)
    return df

def show_model_classification(df):
    st.title("🎧 Model Klasifikasi Preferensi Lagu")

    model, scaler, fitur_model = load_model_scaler()

    # Pastikan kolom popularity_class ada untuk evaluasi
    if 'popularity_class' not in df.columns:
        df = buat_label_popularity(df)

    st.subheader("Prediksi Preferensi Lagu")
    lagu_options = df[['track_name', 'artists']].drop_duplicates()
    lagu_options['display'] = lagu_options['track_name'] + " - " + lagu_options['artists']
    selected_lagu = st.selectbox("Pilih lagu:", lagu_options['display'])

    # Ambil data fitur lagu yang dipilih
    track_name = lagu_options[lagu_options['display'] == selected_lagu]['track_name'].values[0]
    track_data = df[df['track_name'] == track_name].iloc[0]

    try:
        # Siapkan data input sesuai fitur model
        X_input = pd.DataFrame([track_data])
        X_input = sesuaikan_fitur(X_input, fitur_model).astype(float)

        # Transformasi dan prediksi
        X_input_scaled = scaler.transform(X_input)
        prediksi = model.predict(X_input_scaled)[0]
        label = "Suka" if prediksi == 1 or prediksi == 'Suka' else "Tidak Suka"

        st.markdown(f"Lagu **{track_name}** diprediksi sebagai: **{label}**")

    except ValueError as e:
        st.error(f"Terjadi error saat transformasi fitur: {e}")
        st.stop()

    # Evaluasi Model
    st.subheader("Evaluasi Model")
    try:
        # Siapkan seluruh data
        X_all = sesuaikan_fitur(df, fitur_model).astype(float)
        X_all_scaled = scaler.transform(X_all)
        y_all = df['popularity_class']

        y_pred = model.predict(X_all_scaled)

        # Perbaikan penting: konversi hasil prediksi ke angka jika masih berupa string
        if isinstance(y_pred[0], str):
            y_pred = pd.Series(y_pred).map({'Suka': 1, 'Tidak Suka': 0}).values

        # Confusion Matrix dengan label
        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_all, y_pred, labels=[1, 0])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Suka", "Tidak Suka"])
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        st.pyplot(fig)

        # Classification Report dengan label string
        report = classification_report(
            y_all,
            y_pred,
            target_names=["Tidak Suka", "Suka"]
        )
        st.text("Classification Report:")
        st.text(report)

    except Exception as e:
        st.error(f"Gagal evaluasi model: {e}")

    # Tabel Prediksi Contoh
    st.subheader("Contoh Prediksi Lagu")
    try:
        preview = df[['track_name', 'artists'] + fitur_model].copy()
        preview_fitur = sesuaikan_fitur(preview, fitur_model).astype(float)
        preview['Prediksi'] = model.predict(scaler.transform(preview_fitur))

        # Mapping ke string hanya untuk tampilan UI
        preview['Prediksi'] = preview['Prediksi'].map({
            1: "Suka", 0: "Tidak Suka", 'Suka': "Suka", 'Tidak Suka': "Tidak Suka"
        })

        st.dataframe(preview[['track_name', 'artists', 'Prediksi']].head(10))

    except Exception as e:
        st.error(f"Gagal menampilkan prediksi contoh: {e}")
