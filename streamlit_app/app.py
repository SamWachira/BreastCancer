import streamlit as st
import requests

st.title("Breast Cancer Image Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

    # Change this URL to your FastAPI URL if needed
    response = requests.post("http://fastapi:8000/predict/", files=files)

    if response.status_code == 200:
        prediction = response.json()
        st.write(f"Predicted Class Index: {prediction['predicted_class_index']}")
        st.write(f"Predicted Class Label: {prediction['predicted_class_label']}")
    else:
        st.write("Error in prediction.")
