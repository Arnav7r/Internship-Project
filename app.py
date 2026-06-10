import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("leaf_model.keras")

# Class names
class_names = [
    'Tomato___Bacterial_spot',
    'Tomato___Septoria_leaf_spot',
    'Tomato___healthy'
]

st.title("🍅 Tomato Leaf Disease Detection")

uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.resize((224,224))
    img_array = np.array(img)

    # Remove alpha channel
    if len(img_array.shape) == 3 and img_array.shape[-1] == 4:
        img_array = img_array[:,:,:3]

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")