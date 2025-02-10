import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from PIL import Image
import io

# Streamlit App Title
st.title("Signature Verification App ✍️")

# File Upload
st.sidebar.header("Upload Signature Images")
uploaded_file1 = st.sidebar.file_uploader("Upload Original Signature", type=["png", "jpg", "jpeg"])
uploaded_file2 = st.sidebar.file_uploader("Upload Test Signature", type=["png", "jpg", "jpeg"])

if uploaded_file1 and uploaded_file2:
    # Convert uploaded files to OpenCV format
    def load_image(file):
        image = Image.open(file).convert("L")  # Convert to grayscale
        return np.array(image)

    img1 = load_image(uploaded_file1)
    img2 = load_image(uploaded_file2)

    # Resize images to same dimensions
    fixed_size = (500, 200)
    img1 = cv2.resize(img1, fixed_size)
    img2 = cv2.resize(img2, fixed_size)

    # Apply Otsu's thresholding
    _, thresh1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Display Images
    st.image([img1, img2], caption=["Original Signature", "Test Signature"], width=300)

    # Compute SSIM
    similarity_index, diff = ssim(thresh1, thresh2, full=True)
    ssim_percentage = similarity_index * 100
    st.write(f"🔹 **SSIM Similarity:** {ssim_percentage:.2f}%")

    # ORB Feature Matching
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(thresh1, None)
    kp2, des2 = orb.detectAndCompute(thresh2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    if des1 is not None and des2 is not None:
        matches = bf.match(des1, des2)
        good_matches = [m for m in matches if m.distance < 70]
        max_keypoints = max(len(kp1), len(kp2))
        orb_percentage = (len(good_matches) / max_keypoints) * 100 if max_keypoints > 0 else 0
    else:
        good_matches = []
        orb_percentage = 0.0

    st.write(f"🔹 **ORB Feature Match:** {orb_percentage:.2f}%")

    # Draw ORB Matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Deep Learning Model: Siamese Network
    def build_siamese_model(input_shape):
        input_layer = Input(shape=input_shape)
        x = Conv2D(64, (3,3), activation="relu")(input_layer)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Conv2D(128, (3,3), activation="relu")(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        return Model(input_layer, x)

    # Preprocess images for deep learning model
    def preprocess_for_model(image, target_size=(128, 128)):
        img = cv2.resize(image, target_size) / 255.0
        img = np.expand_dims(img, axis=-1)
        return np.expand_dims(img, axis=0)

    # Create Model
    input_shape = (128, 128, 1)
    siamese_model = build_siamese_model(input_shape)

    # Compute Similarity using Siamese Network
    img1_dl = preprocess_for_model(img1)
    img2_dl = preprocess_for_model(img2)

    feature1 = siamese_model.predict(img1_dl)
    feature2 = siamese_model.predict(img2_dl)

    # Compute L2 Distance (Euclidean)
    distance = np.linalg.norm(feature1 - feature2)

    # Convert to Similarity Score (Lenient function)
    similarity_score = 100 / (1 + distance)

    # Adjusted Threshold
    THRESHOLD = 50
    match_result = "✅ Signatures Match!" if similarity_score >= THRESHOLD else "❌ Signatures Do NOT Match!"
    st.write(f"🔹 **Siamese Network Similarity:** {similarity_score:.2f}%")
    st.write(f"📝 **Result:** {match_result}")

    # Generate Difference Image for Deep Learning
    diff_image = cv2.absdiff(img1, img2)
    _, diff_image = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)
    diff_colored = cv2.merge([diff_image, np.zeros_like(diff_image), np.zeros_like(diff_image)])  # Red channel

    overlay_img1 = cv2.addWeighted(cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR), 1, diff_colored, 0.5, 0)
    overlay_img2 = cv2.addWeighted(cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), 1, diff_colored, 0.5, 0)

    # Display Comparison Visuals
    st.subheader("🔍 Signature Comparison")
    st.image(img_matches, caption="ORB Feature Matches", use_column_width=True)
    st.image([overlay_img1, overlay_img2], caption=["Deep Learning Mismatch - Signature 1", "Deep Learning Mismatch - Signature 2"], width=300)

    # Final Decision
    st.success(match_result) if "✅" in match_result else st.error(match_result)
