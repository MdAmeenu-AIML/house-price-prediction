import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("house_data5.csv")

# Features and target
X = df[["Area", "Bedrooms", "Bathrooms", "Age"]]
y = df["Price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("🏠 House Price Prediction App")

area = st.number_input("Area (sq ft)", min_value=500)
bedrooms = st.number_input("Bedrooms", min_value=1)
bathrooms = st.number_input("Bathrooms", min_value=1)
age = st.number_input("House Age", min_value=0)

if st.button("Predict Price"):
    prediction = model.predict([[area, bedrooms, bathrooms, age]])

    st.success(f"Predicted House Price: ₹{prediction[0]:,.2f}")
