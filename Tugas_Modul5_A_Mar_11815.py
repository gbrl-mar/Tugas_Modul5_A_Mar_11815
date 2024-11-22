import streamlit as st
from PIL import Image
import numpy as np
import pickle

# Fungsi untuk memuat model
def load_model():
    model_path = os.path.join(os.getcwd(), 'best_model.pkl')  # Path relatif untuk GitHub
    with open(model_path, 'rb') as file:  # Gunakan path relatif
        return pickle.load(file)


# Fungsi untuk melakukan prediksi
def predict_image(image, model):
    labels = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    # Preprocessing gambar
    image = image.resize((28, 28)).convert("L")  # Resize dan grayscale
    img_array = np.array(image) / 255.0  # Normalisasi
    img_array = img_array.flatten().reshape(1, -1)  # Ubah menjadi array 1D untuk model Scikit-learn
    # Prediksi
    predictions = model.predict_proba(img_array)  # Gunakan predict_proba untuk mendapatkan probabilitas
    confidence = np.max(predictions) * 100  # Ambil nilai confidence tertinggi
    predicted_class = np.argmax(predictions)  # Ambil kelas dengan confidence tertinggi
    return labels[predicted_class], confidence



# Judul aplikasi
st.title("Fashion MNIST Image Classifier")
st.markdown("Unggah satu atau lebih gambar item fashion (misalnya sepatu, tas, baju), dan model akan memprediksi kelasnya.")

# Unggah gambar
uploaded_files = st.file_uploader(
    "Pilih gambar...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Model Scikit-learn
model = load_model()

# Prediksi dan hasil
def display_results():
    if uploaded_files:
        st.write("### Hasil Prediksi:")
        for uploaded_file in uploaded_files:
            # Baca file gambar
            image = Image.open(uploaded_file)
            # Prediksi kelas dan confidence
            predicted_class, confidence = predict_image(image, model)

            # Tampilkan hasil
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption=f"{uploaded_file.name}", use_column_width=True)
            with col2:
                st.write(f"**Nama File:** {uploaded_file.name}")
                st.write(f"**Kelas Prediksi:** {predicted_class}")
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.write("---")
    else:
        st.warning("Harap unggah setidaknya satu gambar.")


# Sidebar navigasi
st.sidebar.title("Navigator")
if st.sidebar.button("Predict", key="predict"):
    display_results()
