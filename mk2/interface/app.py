import streamlit as st
import requests
import base64

st.title("💌 Baudel-AI.r 💌")
st.write("Generate a personalized love letter or poem using LLaVA 13B.")

uploaded_image = st.file_uploader("Upload an image (optional):", type=["jpg", "png"])

recipient_name = st.text_input("Recipient's Name:")
custom_message = st.text_input("Custom Sentence to include in the poem (optional):")
additional_context = st.text_area("Additional Context (optional):")

if st.button("Generate Love Letter"):
    if not recipient_name:
        st.error("Please enter the recipient's name.")
    else:
        prompt = f"Write a love letter to {recipient_name}."
        if custom_message:
            prompt += f" Include the following: {custom_message}."

        if additional_context:
            prompt += f" Additional context: {additional_context}"
        prompt += "."

        image_data = []
        if uploaded_image:
            image_data.append(base64.b64encode(uploaded_image.read()).decode('utf-8'))
            prompt += " Use the provided image to inspire your letter. If it's a person, describe it accurately in the poem while complimenting them."

        payload = {
            "model": "llava:13b",
            "prompt": prompt,
            "images": image_data
        }

        try:
            response = requests.post("http://fastapi_backend:8000/generate", json=payload)
            response.raise_for_status()

            response_data = response.json()

            if "text" in response_data:
                st.subheader("Your Love Letter 💖")
                st.write(response_data["text"])
            else:
                st.error("Error: No text received from the model.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with the backend: {e}")
