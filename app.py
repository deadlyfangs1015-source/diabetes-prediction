import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Page config
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Prediction")
st.write("Enter patient details below and click **Predict** to check diabetes risk using Machine Learning.")

# Train model (cached - runs only once)
@st.cache_resource
def train_model():
    df = pd.read_csv("training_data.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    return model, scaler, accuracy

model, scaler, accuracy = train_model()

# Sidebar - Patient inputs
st.sidebar.header("📋 Patient Details")
pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 1)
glucose = st.sidebar.slider("Glucose Level", 0, 200, 120)
bp = st.sidebar.slider("Blood Pressure (mm Hg)", 0, 122, 70)
skin = st.sidebar.slider("Skin Thickness (mm)", 0, 99, 20)
insulin = st.sidebar.slider("Insulin (mu U/ml)", 0, 846, 80)
bmi = st.sidebar.slider("BMI", 0.0, 67.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.47)
age = st.sidebar.slider("Age", 21, 81, 30)

# Predict button
if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.markdown("---")
    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ **Prediction: Diabetic**")
        st.metric("Confidence", f"{probability[1]*100:.1f}%")
    else:
        st.success(f"✅ **Prediction: Not Diabetic**")
        st.metric("Confidence", f"{probability[0]*100:.1f}%")

    # Show input summary
    st.markdown("---")
    st.subheader("Patient Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Glucose", glucose)
    col2.metric("BMI", bmi)
    col3.metric("Age", age)
    col4.metric("BP", bp)

# Footer
st.markdown("---")
st.caption(f"Model: Logistic Regression | Accuracy: {accuracy:.2%} | Trained on Pima Indians Diabetes Dataset (768 samples)")
