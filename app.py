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
            color: #00FF00; /* Green color for the prediction text */
            font-weight: bold;
            text-align: center;
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
        result_text = f"<div class='prediction'>Prediction: **{categories[prediction[0]]}**</div>"
        st.markdown(result_text, unsafe_allow_html=True)

        if prediction[0] == 0:
            st.markdown("<div style='text-align: center;'><img src='https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExdndxZDV2NWxwemYxeDVtbXZ3c2Y2d3ZocnExYmtycXlhZmJ6YnowaiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/oywDPeCxPK0HeKz7pz/giphy.webp' width='300'></div>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>🎉 Healthy Corals! 🎉</h2>", unsafe_allow_html=True)
            st.bokeh_chart(create_confetti())
        else:
            st.markdown("<div style='text-align: center;'><img src='https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExdndxZDV2NWxwemYxeDVtbXZ3c2Y2d3ZocnExYmtycXlhZmJ6YnowaiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/oywDPeCxPK0HeKz7pz/giphy.webp' width='300'></div>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>❄️ Bleached Corals ❄️</h2>", unsafe_allow_html=True)
            st.snow()  # Add snow animation for bleached corals

    else:
        st.sidebar.info("Please upload an image to start the classification.")
        st.image('https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExdndxZDV2NWxwemYxeDVtbXZ3c2Y2d3ZocnExYmtycXlhZmJ6YnowaiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/oywDPeCxPK0HeKz7pz/giphy.webp', width=400)

def create_confetti():
    from bokeh.plotting import figure, output_file, show
    from bokeh.models import ColumnDataSource
    from bokeh.io import output_notebook
    import pandas as pd

    output_notebook()

    n = 500
    x = np.random.random(size=n) * 100
    y = np.random.random(size=n) * 100
    radii = np.random.random(size=n) * 1.5
    colors = [
        "#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50 + 2*x, 30 + 2*y)
    ]

    p = figure(title="Confetti!", tools="hover", toolbar_location=None,
               plot_height=400, plot_width=400, x_range=[0, 100], y_range=[0, 100],
               tooltips="@desc", background_fill_color=None, border_fill_color=None)

    source = ColumnDataSource(data=dict(
        x=x,
        y=y,
        radius=radii,
        colors=colors,
        desc=["Confetti" for _ in range(n)],
    ))

    p.circle('x', 'y', radius='radius', fill_color='colors', fill_alpha=0.6, line_color=None, source=source)

    return p

if __name__ == '__main__':
    main()
