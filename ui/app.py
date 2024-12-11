# ui/app.py


import os
import streamlit as st
import requests

st.title("IMDB Sentiment Classifier")
st.write("Enter a movie review, and the model will predict its sentiment.")

# API_URL from environment variables
API_URL = os.getenv("API_URL", "http://service:8000/predict")

# user input
review = st.text_area("Your Review:", height=200)

# predict button
if st.button("Predict"):
    if not review.strip():
        st.warning("Please enter a movie review.")
    else:
        try:
            payload = {"review": review}
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                sentiment = response.json().get("sentiment", "unknown")
                st.success(f"Sentiment: **{sentiment.capitalize()}**")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Please ensure the service is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
