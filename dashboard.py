import streamlit as st
import sys
import requests
import pandas as pd
import matplotlib.pyplot as plt

# st.write(f"Python version: {sys.version}")

# Streamlit App Title
st.title("Mental Health Monitoring Dashboard")

# Input for Sentiment Analysis
st.header("Analyze Mental Health Sentiment")
text_input = st.text_area("Enter text to analyze:")
model_choice = st.selectbox("Select Model:", ["BERT", "Random Forest"])

if st.button("Analyze"):
    # Send request to Flask API
    api_url = "http://127.0.0.1:5000/predict"
    response = requests.post(api_url, json={"text": text_input, "model": model_choice.lower()})
    
    if response.status_code == 200:
        prediction = response.json()
        st.success(f"Prediction using {prediction['model']}: {prediction['prediction']}")
    else:
        st.error("Error: Could not connect to the API.")

# Example Visualization
st.header("Sentiment Trends Over Time")
# Mock Data for Trends
mock_data = pd.DataFrame({
    'Date': pd.date_range(start="2024-01-01", periods=10),
    'Sentiment': [1, -1, 0, 1, -2, 2, 0, -1, 1, -2]
})

plt.plot(mock_data['Date'], mock_data['Sentiment'], marker='o')
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.xticks(rotation=45)  # Rotate date labels
plt.tight_layout()  # Adjust layout to prevent label cutoff
st.pyplot(plt)
