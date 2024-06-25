import streamlit as st
import numpy as np
import pandas as pd  # Import pandas
import pickle
import os
from PIL import Image, ImageOps
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_image(image, img_size=(64, 64)):
    img = ImageOps.fit(image, img_size, Image.LANCZOS)  # Resize using LANCZOS method
    img = ImageOps.grayscale(img)  # Convert to grayscale
    img_array = np.array(img).flatten()
    return img_array

def load_model(model_path):
    with open(model_path, 'rb') as file:
        svm_model, scaler, pca = pickle.load(file)
    return svm_model, scaler, pca

def predict(image, model, scaler, pca):
    img_array = preprocess_image(image)
    img_scaled = scaler.transform(img_array.reshape(1, -1))
    img_pca = pca.transform(img_scaled)
    prediction = model.predict(img_pca)
    return prediction

def main():
    st.set_page_config(
        page_title="Coral Reef Image Classifier App",
        page_icon=":shark:",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('https://github.com/SriKumar1313/coral_classy_app/blob/main/assets/background.png?raw=true');
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
        h1 {{
            color: #FFD700; /* Gold color for the title */
            text-align: center;
        }}
        .prediction {{
            font-weight: bold;
            text-align: center;
        }}
        .prediction-healthy {{
            color: #00BFFF; /* DeepSkyBlue color for healthy prediction text */
        }}
        .prediction-bleached {{
            color: #8B0000; /* DarkRed color for bleached prediction text */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1>🌊 Coral Reef Image Classifier 🐠</h1>", unsafe_allow_html=True)
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


    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.sidebar.image(image, caption='Uploaded Image.', use_column_width=True)
        st.sidebar.write("")
        st.sidebar.write("Classifying...")

        model_path = 'svm_model_pca.pkl'
        if not os.path.exists(model_path):
            st.error("Error: SVM model file not found.")
            return
        
        svm_model, scaler, pca = load_model(model_path)

        with st.spinner("Processing..."):
            prediction = predict(image, svm_model, scaler, pca)

        categories = ['Healthy Corals', 'Bleached Corals']
        if prediction[0] == 0:
            result_text = f"<div class='prediction prediction-healthy'>Prediction: **{categories[prediction[0]]}**</div>"
            st.markdown(result_text, unsafe_allow_html=True)
            st.markdown("<div style='text-align: center;'><img src='https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExdndxZDV2NWxwemYxeDVtbXZ3c2Y2d3ZocnExYmtycXlhZmJ6YnowaiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/oywDPeCxPK0HeKz7pz/giphy.webp' width='300'></div>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>🎉 Healthy Corals! 🎉</h2>", unsafe_allow_html=True)
            st.pydeck_chart(create_confetti())
        else:
            result_text = f"<div class='prediction prediction-bleached'>Prediction: **{categories[prediction[0]]}**</div>"
            st.markdown(result_text, unsafe_allow_html=True)
            st.markdown("<div style='text-align: center;'><img src='https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExdndxZDV2NWxwemYxeDVtbXZ3c2Y2d3ZocnExYmtycXlhZmJ6YnowaiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/oywDPeCxPK0HeKz7pz/giphy.webp' width='300'></div>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>❄️ Bleached Corals ❄️</h2>", unsafe_allow_html=True)
            st.snow()  # Add snow animation for bleached corals

    else:
        st.sidebar.info("Please upload an image to start the classification.")
        st.image('https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExdndxZDV2NWxwemYxeDVtbXZ3c2Y2d3ZocnExYmtycXlhZmJ6YnowaiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/oywDPeCxPK0HeKz7pz/giphy.webp', width=400)

def create_confetti():
    import pydeck as pdk

    data = pd.DataFrame({
        'lat': np.random.uniform(-90, 90, 500),
        'lon': np.random.uniform(-180, 180, 500),
        'size': np.random.uniform(10, 100, 500),
        'color': [[46, 139, 87] for _ in range(500)]  # DarkGreen color for confetti
    })

    layer = pdk.Layer(
        'ScatterplotLayer',
        data,
        get_position='[lon, lat]',
        get_radius='size',
        get_fill_color='color',
        pickable=True,
        opacity=0.6
    )

    view_state = pdk.ViewState(
        latitude=0,
        longitude=0,
        zoom=0.5,
        pitch=0
    )

    return pdk.Deck(layers=[layer], initial_view_state=view_state)

if __name__ == '__main__':
    main()
