import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load Model
model = tf.keras.models.load_model("Face_mask_detection.h5")

class_names = ['No Mask', 'Mask']

def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((128, 128))
    img = img.convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 128, 128, 3))
    return img_array

# ------------------- CUSTOM STYLING ---------------------
st.markdown("""
<style>

body {
    background-color: #f0f2f6;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #2E86C1;
    padding: 10px;
}

.upload-box {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
}

.pred-result {
    font-size: 26px;
    font-weight: bold;
    text-align: center;
    padding: 12px;
    border-radius: 10px;
    margin-top: 20px;
}

.mask {
    background: #C8F7C5;
    color: #1E8449;
    border: 2px solid #1E8449;
}

.nomask {
    background: #F5B7B1;
    color: #922B21;
    border: 2px solid #922B21;
}

.footer {
    margin-top: 40px;
    text-align: center;
    font-size: 14px;
    color: grey;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------

st.markdown("<div class='title'>Face Mask Detection System 😷</div>", unsafe_allow_html=True)

st.markdown("<div class='upload-box'>", unsafe_allow_html=True)

uploaded_image = st.file_uploader("Upload an image of a person", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:

    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", width=250)

    with col2:
        if st.button("🔍 Classify", use_container_width=True):
            img_array = preprocess_image(uploaded_image)

            result = model.predict(img_array)
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]

            if prediction == "Mask":
                st.markdown("<div class='pred-result mask'>😷 Wearing a Mask</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='pred-result nomask'>❌ Not Wearing a Mask</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>Developed using TensorFlow & Streamlit</div>", unsafe_allow_html=True)