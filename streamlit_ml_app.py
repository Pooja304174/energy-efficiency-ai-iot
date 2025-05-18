import streamlit as st
import pandas as pd
import random
import joblib

# ---------- Simulated IoT Data ----------
def simulate_iot_data():
    heart_rate = random.randint(60, 110)
    spo2 = random.randint(90, 100)
    temperature = round(random.uniform(36.0, 39.0), 1)
    return heart_rate, spo2, temperature

# ---------- Load Trained Model ----------
model = joblib.load("diagnosis_model.pkl")
symptom_columns = ['fever', 'cough', 'headache', 'fatigue', 'chest_pain']

# ---------- Convert Symptoms to Features ----------
def process_symptoms(symptom_text):
    symptom_text = symptom_text.lower()
    features = []
    for symptom in symptom_columns:
        features.append(1 if symptom.replace("_", " ") in symptom_text or symptom in symptom_text else 0)
    return features

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Healthcare Assistant", layout="centered")

st.title("🧠 Healthcare Diagnostics and Treatment (AI)")
st.subheader("ML-powered symptom checker with simulated vitals.")

st.markdown("### 👉 Enter your symptoms below:")
user_input = st.text_area("Example: fever, cough, headache")

if st.button("Run Diagnosis"):
    if user_input.strip() == "":
        st.warning("Please enter some symptoms.")
    else:
        # Simulated Vitals
        heart_rate, spo2, temperature = simulate_iot_data()
        st.markdown("### 📊 Simulated Vital Signs:")
        st.write(f"**Heart Rate:** {heart_rate} bpm")
        st.write(f"**SpO2 Level:** {spo2}%")
        st.write(f"**Body Temperature:** {temperature} °C")

        # Make Prediction
        input_vector = process_symptoms(user_input)
        df_input = pd.DataFrame([input_vector], columns=symptom_columns)
        prediction = model.predict(df_input)[0]

        st.success(f"🩺 **Predicted Disease:** {prediction}")
        st.info("💊 **Note:** Treatment advice should always be confirmed by a doctor.")

st.markdown("---")
st.caption("Prototype using ML model + simulated vitals. Not for real medical use.")