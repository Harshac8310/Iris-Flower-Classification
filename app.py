import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("iris_model.joblib")

# Species names
species_names = ["Setosa", "Versicolor", "Virginica"]

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)

# App title
st.title("🌸 Iris Flower Classification")
st.write("Predict the species of an Iris flower using its measurements.")

st.divider()

# User inputs
st.subheader("Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)", min_value=0.1,
        max_value=10.0, value=5.1, step=0.1
    )
    petal_length = st.number_input(
        "Petal Length (cm)", min_value=0.1,
        max_value=10.0, value=1.4, step=0.1
    )

with col2:
    sepal_width = st.number_input(
        "Sepal Width (cm)", min_value=0.1,
        max_value=10.0, value=3.5, step=0.1
    )
    petal_width = st.number_input(
        "Petal Width (cm)", min_value=0.1,
        max_value=10.0, value=0.2, step=0.1
    )

# Prediction
if st.button("Predict Species", type="primary", use_container_width=True):
    features = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(features)[0]
    predicted_species = species_names[prediction]

    st.success(f"Predicted Species: {predicted_species}")

    # Display prediction probabilities
    probabilities = model.predict_proba(features)[0]

    st.subheader("Prediction Probabilities")

    for name, probability in zip(species_names, probabilities):
        st.write(f"{name}: {probability * 100:.2f}%")
        st.progress(float(probability))

st.divider()
st.caption("Machine Learning Project | Logistic Regression | Scikit-learn")