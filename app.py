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

    # Adding background image and custom CSS
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('https://github.com/SriKumar1313/coral_classy_app/blob/main/assets/background.jpg?raw=true');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: #FFFFFF;
        }}
        .sidebar .sidebar-content {{
            background-color: rgba(255, 255, 255, 0.9);
        }}
        .block-container {{
            padding: 2rem;
            border-radius: 10px;
            background-color: rgba(0, 0, 0, 0.7);
            color: #FFFFFF;
            box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.5);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🌊 Coral Reef Image Classifier 🐠")
    st.write("""
        Welcome to the **Coral Reef Image Classifier**!
        
        This app helps in identifying whether the coral reefs in the uploaded image are **healthy** or **bleached**.
        
        🐟 **Steps to use the app**:
        1. Upload an image of a coral reef using the sidebar.
        2. Wait for the app to process and classify the image.
        3. See the results and enjoy the interactive feedback!
    """)

    st.sidebar.title("📤 Upload Your Image Here")
    st.sidebar.write("Please upload an image of coral reefs to start the classification.")

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
            st.error("Error: SVM model file not found.")
            return
        
        svm_model, scaler, pca = load_model(model_path)

        # Add progress bar and status text
        with st.spinner("Processing..."):
            prediction = predict(image, svm_model, scaler, pca)

        categories = ['Healthy Corals', 'Bleached Corals']
        result_text = f"Prediction: **{categories[prediction[0]]}**"
        st.success(result_text)

        if prediction[0] == 0:
            st.balloons()  # Add balloons animation for healthy corals
            st.markdown("<div style='text-align: center;'><img src='https://media.giphy.com/media/l4FGEyRzaeg5p0DXm/giphy.gif' width='300'></div>", unsafe_allow_html=True)
        else:
            st.snow()  # Add snow animation for bleached corals
            st.markdown("<div style='text-align: center;'><img src='https://media.giphy.com/media/l4FGF94cInGdfk9MI/giphy.gif' width='300'></div>", unsafe_allow_html=True)

    else:
        st.sidebar.info("Please upload an image to start the classification.")
        st.image('https://media.giphy.com/media/l4FGGafcOHmrlQxGk/giphy.gif', width=400)

if __name__ == '__main__':
    main()
