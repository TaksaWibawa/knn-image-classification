import cv2
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import pickle

angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
n_neighbors = [3, 5, 7, 9, 11]
features = []
labels = []

model = pickle.load(open('model/knn_model.sav', 'rb'))

def read_image(img):
    img = cv2.imdecode(np.fromstring(img.read(), np.uint8), 1)
    return img

def rgb_to_gray(img):
    gray = cv2.cvtColor(read_image(img), cv2.COLOR_BGR2GRAY)
    return gray

def feature_extraction(img):
    features = []
    for angle in angles:
        glcm = graycomatrix(img, [1], [angle], levels=256, symmetric=True, normed=True)
        dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
        correlation = graycoprops(glcm, 'correlation').ravel()
        homogeneity = graycoprops(glcm, 'homogeneity').ravel()
        contrast = graycoprops(glcm, 'contrast').ravel()
        asm = graycoprops(glcm, 'ASM').ravel()
        energy = graycoprops(glcm, 'energy').ravel()
        angle_features = np.concatenate((dissimilarity, correlation, homogeneity, contrast, asm, energy))
        features.extend(angle_features)

    return np.array(features)

def knn_model(features):
    prediction = model.predict([features])
    return prediction