# EDA.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def show_eda(df):
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.subheader("Statistik Umum Dataset")
    st.write(df.describe())

    st.subheader("Distribusi Popularitas Lagu")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['popularity'], bins=30, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Jumlah Lagu per Genre")
    fig2, ax2 = plt.subplots(figsize=(10,5))
    genre_count = df['track_genre'].value_counts().head(20)
    sns.barplot(x=genre_count.index, y=genre_count.values, ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig2)

    st.subheader("Korelasi antar Fitur")
    numeric_df = df.select_dtypes(include=[np.number])
    fig3, ax3 = plt.subplots(figsize=(12, 10), dpi=150)
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax3, linewidths=0.5,
                cbar_kws={"shrink": 0.5}, annot_kws={"size": 8})
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax3.set_yticklabels(ax3.get_yticklabels(), fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)
