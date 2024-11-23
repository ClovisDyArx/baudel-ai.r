import streamlit as st
import requests
from dotenv import load_dotenv
import os

# Loading env variables.
load_dotenv(dotenv_path='../env/network.env')
HOST = os.getenv('HOST')
PORT = os.getenv('PORT')
api_url = f"http://{HOST}:{PORT}/generate/"

load_dotenv(dotenv_path='../env/model.env')
api_token = os.getenv('EXAMPLE_TOKEN')

# Streamlit interface.
st.title("Love Letter Generator - GPT2-Medium")
prompt = st.text_input("Enter a prompt:")
max_length = st.slider("Max Length", 100, 200, 150)
min_length = st.slider("Min Length", 50, 100, 50)
temperature = st.slider("Temperature", 0.1, 1.5, 0.7)
top_k = st.slider("Top-K", 10, 100, 50)
top_p = st.slider("Top-P", 0.1, 1.0, 0.95)

if st.button("Generate Text"):
    payload = {
        "prompt": prompt,
        "min_length": min_length,
        "max_length": max_length,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }
    headers = {"token": api_token}
    response = requests.post(api_url, json=payload, headers=headers)

    if response.status_code == 200:
        st.success("Generated Text:")
        st.write(response.json()["generated_text"])

    else:
        st.error(f"Error: {response.json()['detail']}")
