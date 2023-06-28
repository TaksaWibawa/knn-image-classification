import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from modules.knn import *

# Get the images from folder
parent_folder = "FacialExpression/"
subfolder_names = ["happy", "neutral", "sad"]

if __name__ == '__main__':
    st.title("Image Classification with K-Nearest Neighbor and GLCM")
    st.markdown(f'''
                    Oleh Kelompok 3 Kelas D : **(image, KNN)**
                    \n 1. Yehezkiel Batara Lumbung (2108561048)
                    \n 2. Ni Wayan Sani Utari Dewi (2108561089)
                    \n 3. I Made Sudarsana Taksa Wibawa (2108561109)
                    \nLink Github : 
                    <a href="https://github.com/TaksaWibawa/knn-image-classification.git">https://github.com/TaksaWibawa/knn-image-classification.git</a>
                    \n-----
                ''', unsafe_allow_html=True)

    #1 Upload Gambar yang Ingin Diprediksi (bisa lebih dari 1 gambar)
    st.header("Upload Gambar")
    uploaded_file = st.file_uploader(label="Dapat memilih lebih dari satu gambar", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    submit = st.button('Submit')

    df_img = pd.DataFrame(columns=['Image Name', 'Prediction'])
    df_list = []

    if submit:
        if uploaded_file is not None:
            for img in uploaded_file:
                try:
                    gray = rgb_to_gray(img)
                    features = feature_extraction(gray)
                    prediction = knn_model(features)
                    pred_df = pd.DataFrame(
                    {"Image Name": [img.name],
                    "Prediction": [prediction[0]]})
                    df_list.append(pred_df)
                    df_img = pd.concat(df_list, ignore_index=True)
                except:
                    st.write("Image not found or corrupted!")
                
            # Tampilkan list nama gambar yang diupload
            st.write("---")
            st.header("List Gambar")                
            st.dataframe(df_img['Image Name'], width=800)

            # Tampilkan hasil ekstraksi fitur glcm
            st.write("---")
            st.header("Hasil Ekstraksi Fitur GLCM")
            st.info(f"### Work in Progress")
            st.info(f"### Ekstraksi Fitur Kurang Optimal!")

            # Tampilkan hasil prediksi
            st.write("---")
            st.header("Hasil Prediksi")
            st.dataframe(df_img, width=800)

            # Tampilkan gambar dan prediksinya
            st.write("---")
            st.header("Gambar dengan Prediksi")
            num_images = len(uploaded_file)
            num_cols = 6  # Maximum of 6 images per row
            num_rows = (num_images - 1) // num_cols + 1
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))

            for i, uploaded_file in enumerate(uploaded_file):
                image = Image.open(uploaded_file)

                row_idx = i // num_cols
                col_idx = i % num_cols

                if num_rows > 1:  # Handle multiple rows
                    ax = axes[row_idx, col_idx]
                else:  # Handle single row
                    ax = axes[col_idx]

                ax.set_title(df_img['Prediction'][i], fontsize=32)
                ax.imshow(image, cmap='gray')
                
                ax.axis('off')

            # Remove empty subplots if the number of images is not a multiple of num_cols
            if num_images % num_cols != 0:
                if num_rows > 1:  # Handle multiple rows
                    for j in range(num_images % num_cols, num_cols):
                        axes[-1, j].axis('off')
                else:  # Handle single row
                    for j in range(num_images, num_cols):
                        axes[j].axis('off')

            st.pyplot(fig)

        else:
            st.write("No Image Uploaded")