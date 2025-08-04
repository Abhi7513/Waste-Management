import streamlit as st
from predict import predict_image
from PIL import Image
import os


st.title("♻ AI-Based Waste Classifier")


uploaded = st.file_uploader("Upload an image of waste", type=["jpg", "jpeg", "png"])

if uploaded:
    with open('temp.jpg', 'wb') as f:
        f.write(uploaded.read())

    st.image('temp.jpg', caption='Uploaded Image', use_container_width=True)
    label, confidence = predict_image('temp.jpg')
    st.success(f"Predicted Category: {label} ** ({confidence:.2f},confidence)")