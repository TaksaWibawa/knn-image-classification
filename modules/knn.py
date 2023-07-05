import cv2
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import pandas as pd
import pickle

angles = [0, 45, 90, 135, 180]
n_neighbors = [3, 5, 7, 9, 11]
features = []
labels = []

model = pickle.load(open('model/knn_model.sav', 'rb'))

def read_image(img):
    img = cv2.imdecode(np.fromstring(img.read(), np.uint8), 1)
    resize_image(img)
    return img

def rgb_to_gray(img):
    gray = cv2.cvtColor(read_image(img), cv2.COLOR_BGR2GRAY)
    return gray

def resize_image(image, target_size=128):
    # Resize image while maintaining aspect ratio
    height, width = image.shape[:2]
    if height > width:
        new_height = target_size
        new_width = int(width * (target_size / height))
    else:
        new_width = target_size
        new_height = int(height * (target_size / width))
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image

def feature_extraction(img):
    features = []
    for angle in angles:
        glcm = graycomatrix(img, [1], [angle], 256, symmetric=True, normed=True)
        dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
        correlation = graycoprops(glcm, 'correlation').ravel()
        homogeneity = graycoprops(glcm, 'homogeneity').ravel()
        contrast = graycoprops(glcm, 'contrast').ravel()
        asm = graycoprops(glcm, 'ASM').ravel()
        energy = graycoprops(glcm, 'energy').ravel()
        angle_features = np.concatenate((dissimilarity, correlation, homogeneity, contrast, asm, energy))
        features.extend(angle_features)
    
    # Reshape features to a 2-dimensional array with 1 row
    features_column = np.array(features).reshape(1, -1)
    
    # Convert features to a 2-dimensional array
    features_array = np.array(features)

    # # Create a dataframe for the image features
    properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
    columns = []

    columns.extend([f'{property}_{angle}' for angle in angles for property in properties])
    
    df_features = pd.DataFrame(features_column, columns=columns)

    return features_array, df_features

def knn_model(features):
    prediction = model.predict([features])
    return prediction
