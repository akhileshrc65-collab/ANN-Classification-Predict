import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model and encoders
model = tf.keras.models.load_model('model.h5', compile=False)
# Recompile with compatible loss function
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the Streamlit app
st.title("Customer Churn Prediction")

# UserInput fields
geography = st.selectbox("Geography", onehot_encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder.classes_)
age = st.slider("Age", min_value=18, max_value=92)
balance = st.number_input("Balance", min_value=0.0)
credit_score=st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)
tenure = st.slider("Tenure", min_value=0, max_value=10)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

# One-hot encode the 'Geography' feature
geography_encoded = onehot_encoder.transform([[geography]]).toarray()
geography_encoded_df = pd.DataFrame(geography_encoded, columns = onehot_encoder.get_feature_names_out(['Geography']))


input_data = pd.concat([input_data.reset_index(drop=True), geography_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    churn_probability = prediction[0][0]
    st.write(f"Churn Probability: {churn_probability:.2f}")
    if churn_probability > 0.5:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is unlikely to churn.")