import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st

def plot_feature_importance(model, feature_names, top_n=20):
    # Cek apakah model punya atribut feature_importances_ (RandomForest, GradientBoosting, dll)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    # Atau punya coef_ (LogisticRegression, LinearModel)
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        raise AttributeError("Model tidak memiliki atribut feature_importances_ atau coef_")

    # Buat dataframe fitur dan nilai pentingnya
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    # Sort dan ambil top_n fitur teratas
    feat_imp = feat_imp.sort_values(by='importance', ascending=False).head(top_n)

    # Plot horizontal bar chart dengan fitur teratas (urut dari paling kecil di bawah)
    plt.figure(figsize=(10,6))
    plt.barh(feat_imp['feature'][::-1], feat_imp['importance'][::-1], color='skyblue')
    plt.xlabel('🔍 Feature Importance')
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    
    # Tampilkan plot di Streamlit
    st.pyplot(plt)
    plt.clf()  # Clear figure agar tidak menumpuk pada plot berikutnya

    return feat_imp

def show_feature_importance_page(df, model, feature_names):
    st.title("Feature Importance")
    st.write("Menampilkan Feature Importance dari model klasifikasi yang sudah dilatih")

    # Slider untuk memilih jumlah fitur yang akan ditampilkan
    top_n = st.slider("Pilih jumlah fitur teratas yang ditampilkan:", min_value=5, max_value=30, value=15)

    # Tampilkan plot dan dataframe feature importance
    feat_imp_df = plot_feature_importance(model, feature_names, top_n=top_n)
    st.dataframe(feat_imp_df)
