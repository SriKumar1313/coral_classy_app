import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image, ImageOps
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Function to preprocess the uploaded image
def preprocess_image(image, img_size=(64, 64)):
    img = ImageOps.fit(image, img_size, Image.LANCZOS)  # Resize using LANCZOS method
    img = ImageOps.grayscale(img)  # Convert to grayscale
    img_array = np.array(img).flatten()
    return img_array

# Function to load the SVM model and preprocessing components
def load_model(model_path):
    with open(model_path, 'rb') as file:
        svm_model, scaler, pca = pickle.load(file)
    return svm_model, scaler, pca

# Function to make predictions
def predict(image, model, scaler, pca):
    img_array = preprocess_image(image)
    img_scaled = scaler.transform(img_array.reshape(1, -1))
    img_pca = pca.transform(img_scaled)
    prediction = model.predict(img_pca)
    return prediction

# Main function to run the Streamlit web app
def main():
    st.set_page_config(
        page_title="Coral Reef Image Classifier App",
        page_icon=":shark:",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Adding background image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('assets/background.png');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Coral Reef Image Classifier")
    st.sidebar.title("Upload Image")
    
    # Upload image
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.sidebar.image(image, caption='Uploaded Image.', use_column_width=True)
        st.sidebar.write("")
        st.sidebar.write("Classifying...")

        # Load SVM model and preprocessing components
        model_path = 'svm_model_pca.pkl'
        if not os.path.exists(model_path):
            st.write("Error: SVM model file not found.")
            return
        
        svm_model, scaler, pca = load_model(model_path)

        # Make prediction
        prediction = predict(image, svm_model, scaler, pca)
        categories = ['healthy_corals', 'bleached_corals']
        st.write(f"Prediction: {categories[prediction[0]]}")

if __name__ == '__main__':
    main()
