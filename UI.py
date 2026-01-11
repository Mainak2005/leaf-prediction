import streamlit as st
import requests
from PIL import Image


st.set_page_config(page_title="Potato Leaf Disease Classifier", layout="centered")
st.title("🍀 Potato Leaf Disease Classifier")
st.write("Upload an image of a potato leaf to predict its disease class.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            try:
                
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    files={"file": uploaded_file.getvalue()}
                )

                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Prediction: **{data['class']}**")
                    st.info(f"Confidence: {data['confidence']}")
                else:
                    st.error("Prediction failed. Check FastAPI server.")
            except Exception as e:
                st.error(f"Error: {e}")
