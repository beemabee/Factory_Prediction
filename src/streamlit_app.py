import streamlit as st
import joblib
import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
from datetime import time, datetime

# Dapatkan path absolut ke direktori saat ini
current_dir = os.path.dirname(os.path.abspath(__file__))

# Muat model dan scaler
@st.cache_resource
def load_model_and_scaler():
    model_path = os.path.join(current_dir, '..', 'model', 'best_model.joblib')
    scaler_path = os.path.join(current_dir, '..', 'model', 'scaler.joblib')
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model_and_scaler()

# Fitur yang digunakan oleh model
model_features = ['SamplingNC', 'SamplingChek', 'QTY', 'TimeProduce', 'Years']

# Fungsi untuk melakukan prediksi dan SHAP analysis
def predict_and_explain(features):
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_scaled)
    
    return prediction, shap_values, explainer

# UI Streamlit
st.title('Prediksi NC dengan Analisis SHAP')

# Input fields untuk fitur yang digunakan model
input_data = {}
for col in model_features:
    if col == 'TimeProduce':
        time_input = st.time_input(f"Pilih {col}", value=time(0, 0))
        input_data[col] = time_input.hour + time_input.minute / 60.0
    elif col == 'Years':
        input_data[col] = st.number_input(col, min_value=2000, max_value=2100, value=2000)
    else:
        input_data[col] = st.number_input(col, min_value=0, value=0)

if st.button('Prediksi dan Analisis'):
    features = pd.DataFrame([input_data])
    prediction, shap_values, explainer = predict_and_explain(features)
    
    st.write(f'Prediksi NC %: {prediction[0]:.2f}%')

    # Menampilkan data input yang digunakan
    st.write("Data Input:")
    display_features = features.copy()
    display_features['TimeProduce'] = time(int(features['TimeProduce']), int((features['TimeProduce'] % 1) * 60)).strftime("%H:%M")
    st.write(display_features)

    # Visualisasi SHAP (Beeswarm plot)
    st.write("Analisis SHAP (Pengaruh Fitur):")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, features, plot_type="bar", show=False)
    plt.title("Pengaruh Fitur terhadap Prediksi")
    plt.xlabel("Rata-rata dampak pada prediksi")
    plt.tight_layout()
    st.pyplot(fig)
    st.write("Interpretasi: Panjang bar menunjukkan seberapa besar pengaruh fitur terhadap prediksi. "
             "Warna merah menunjukkan pengaruh positif (meningkatkan NC %), "
             "sedangkan warna biru menunjukkan pengaruh negatif (menurunkan NC %).")

    # Waterfall plot untuk feature importance
    st.write("Kontribusi Fitur untuk Prediksi Ini:")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], features.iloc[0], max_display=10, show=False)
    plt.title("Kontribusi Setiap Fitur terhadap Prediksi")
    plt.tight_layout()
    st.pyplot(fig)
    st.write("Interpretasi: Plot ini menunjukkan bagaimana setiap fitur berkontribusi terhadap prediksi akhir. "
             "Batang merah menunjukkan peningkatan NC %, sedangkan batang biru menunjukkan penurunan NC %. "
             "Nilai awal adalah rata-rata prediksi, dan nilai akhir adalah prediksi untuk input ini.")