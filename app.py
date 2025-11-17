import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pickle

model = tf.keras.models.load_model("Face_mask_detection.h5")

class_names = ['No Mask', 'Mask']


def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((128, 128))
    img = img.convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 128, 128, 3))
    return img_array

st.title("Face Mask Detection System 😷")

uploaded_image = st.file_uploader(
    "Upload an image of a person...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:

    image = Image.open(uploaded_image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", width=200)

    with col2:
        if st.button("Classify"):
            img_array = preprocess_image(uploaded_image)

            result = model.predict(img_array)
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]

            if prediction == "Mask":
                st.success("Prediction: Person is **Wearing a Mask** 😷")
            else:
                st.error("Prediction: Person is **NOT Wearing a Mask** ❌")