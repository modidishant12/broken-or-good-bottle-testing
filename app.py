import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D
from PIL import Image, ImageOps
import os

# Set page config
st.set_page_config(page_title="InspectorsAlly", page_icon=":camera:")

# Custom layer to fix the Teachable Machine export issue
class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Remove the problematic 'groups' parameter
        super().__init__(*args, **kwargs)

# Title and intro
st.title("InspectorsAlly")
st.caption("Boost Your Quality Control with InspectorsAlly - The Ultimate AI-Powered Inspection App")
st.write("Try clicking a product image and watch how an AI Model will classify it between Good / Anomaly.")

# Sidebar
with st.sidebar:
    overview_path = r"C:\ai for mfg\inspectorally\overview_dataset.jpg"
    if os.path.exists(overview_path):
        st.image(overview_path, caption="Dataset Overview")
    st.subheader("About InspectorsAlly")
    st.write("InspectorsAlly is a powerful AI-powered application...")

# Load Keras model with the fix
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            "keras_model.h5",
            custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D},
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Image Input UI
input_method = st.radio("Choose input type:", ["File Upload", "Camera Input"], label_visibility="collapsed")

image = None
if input_method == "File Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
elif input_method == "Camera Input":
    camera_file = st.camera_input("Take a picture")
    if camera_file:
        image = Image.open(camera_file)
        st.image(image, caption="Camera Input Image", width=300)

# Prediction function with proper preprocessing
def predict(image):
    try:
        # Preprocess image exactly like Teachable Machine
        img = ImageOps.fit(image, (224, 224), method=Image.Resampling.LANCZOS)
        img_array = np.asarray(img).astype("float32")
        normalized_img = (img_array / 127.5) - 1  # Normalization to [-1, 1]
        
        # Add batch dimension
        input_array = np.expand_dims(normalized_img, axis=0)
        
        # Predict
        prediction = model.predict(input_array)
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[0][predicted_class])
        
        return {
    "class": (
        "LARGELY BROKEN BOTTLE" if predicted_class == 0 
        else "SMALL BROKEN BOTTLE" if predicted_class == 1 
        else "GOOD BOTTLE"
    ),
    "confidence": confidence,
    "all_predictions": prediction[0].tolist()
}

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# Run prediction
if st.button("Submit for Inspection") and image is not None:
    if model is None:
        st.error("Model failed to load. Please check the error message above.")
    else:
        with st.spinner("Analyzing image..."):
            result = predict(image)
            if result:
                if result["class"] == "Good":
                    st.success(f"✅ {result['class']} (Confidence: {result['confidence']:.1%})")
                else:
                    st.error(f"⚠️ {result['class']} (Confidence: {result['confidence']:.1%})")
                
                # Show detailed probabilities
                with st.expander("See detailed analysis"):
                    st.write("Prediction probabilities:")
                    for i, prob in enumerate(result["all_predictions"]):
                        st.write(f"Class {i}: {prob:.1%}")

# Installation instructions in sidebar
#with st.sidebar:
    #st.subheader("Installation Requirements")
    #st.code("pip install streamlit tensorflow pillow numpy")\\
