import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Page title
st.title("🐶 Dog vs 🐱 Cat Classifier")

# Load model
@st.cache_resource
def load_classification_model():
    return load_model('dog_cat_final_model_2.keras')

try:
    model = load_classification_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Upload file
uploaded_file = st.file_uploader("Upload an image (Dog/Cat)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # Preprocess
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):
        prediction = model.predict(img_array)[0][0]

        # Result
        if prediction > 0.5:
            st.success(f"Prediction: 🐶 Dog (Confidence: {prediction:.2f})")
            conf = prediction
        else:
            st.success(f"Prediction: 🐱 Cat (Confidence: {1 - prediction:.2f})")
            conf = 1 - prediction

        # Progress bar
        st.progress(float(conf))

        # Bar chart (fixed)
        df = pd.DataFrame({
            "Label": ["Cat", "Dog"],
            "Confidence": [1 - prediction, prediction]
        }).set_index("Label")

        st.bar_chart(df)

# Sidebar
st.sidebar.header("Instructions")
st.sidebar.info(
    "1. Upload an image of a dog or cat.\n"
    "2. Click **Predict**.\n"
    "3. See the classification result!"
)

st.sidebar.header("About")
st.sidebar.info(
    "This app uses a CNN model trained with TensorFlow/Keras "
    "to classify dog and cat images."
)
