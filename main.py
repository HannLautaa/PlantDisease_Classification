import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

st.title("Upload and Predict")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Decode image from bytes (OpenCV reads in BGR)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Resize to match model input
    image_resized = cv2.resize(image, (128, 128))
    
    # Convert BGR to RGB (if needed for model)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_ril_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize
    image_array = image_rgb / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Load model and predict
    model = tf.keras.models.load_model("daun_model_final.h5")
    pred = model.predict(image_array)
    predict_name = [
        'Pepper_bell / Bacterial Spot',
        'Pepper_bell / Healthy',
        'Potato / Early Blight',
        'Potato / Late Blight',
        'Potato / Healthy',
        'Tomato-Bacterial Spot',
        'Tomato-Early Blight',
        'Tomato-Late Blight',
        'Tomato-Leaf Mold',
        'Tomato-Septoria Leaf Spot',
        'Tomato-2 Spotted Spider Mites',
        'Tomato_Target Spot',
        'Tomato_YellowLeaf_Curl Virus',
        'Tomato_Mosaic Virus'
    ]
    class_name = predict_name[np.argmax(pred)]
    # st.success(f"Prediction: {class_name}")
    # st.success(f"Confidence {np.max(pred):.2f}")
    # st.write(f"Prediction: {class_name}")
    #
    # # Show image
    # st.image(image_ril_rgb, caption="Uploaded Image", use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image_ril_rgb, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.success(f"Prediction: {class_name}")
        st.success(f"Confidence {np.max(pred):.2f}")
