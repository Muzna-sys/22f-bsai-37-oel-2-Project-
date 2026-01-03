import streamlit as st
import numpy as np
from PIL import Image
import time

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Image Retrieval + Clustering",
    layout="wide"
)

st.title("Image Retrieval Dashboard")
st.write("CNN-based Image Retrieval using ResNet50 + Cosine Similarity")

# =========================
# LOAD MODEL & DATA WITH PROGRESS
# =========================
load_bar = st.progress(0)
status = st.empty()

status.text("Loading ResNet50 model...")
load_bar.progress(20)

@st.cache_resource
def load_model():
    return ResNet50(weights="imagenet", include_top=False, pooling="avg")

model = load_model()
time.sleep(0.5)
load_bar.progress(50)

status.text("Loading dataset arrays...")
images = np.load("images_array.npy")
features = np.load("resnet50_features.npy")
time.sleep(0.5)
load_bar.progress(80)

status.text("System ready ✅")
load_bar.progress(100)
time.sleep(0.5)

load_bar.empty()
status.empty()

# =========================
# USER CONTROLS
# =========================
st.sidebar.header("Controls")
top_k = st.sidebar.slider("Top-K Similar Images", 1, 10, 5)

uploaded = st.file_uploader(
    "📤 Upload Query Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# IMAGE RETRIEVAL
# =========================
if uploaded is not None:
    st.subheader("Query Image")
    query_img = Image.open(uploaded).convert("RGB")
    st.image(query_img, width=250)

    # Progress bar for retrieval
    retrieve_bar = st.progress(0)
    retrieve_text = st.empty()

    retrieve_text.text("Preprocessing query image...")
    retrieve_bar.progress(25)
    time.sleep(0.3)

    img_resized = query_img.resize((224, 224))
    img_array = preprocess_input(
        np.expand_dims(np.array(img_resized), axis=0)
    )

    retrieve_text.text("Extracting deep features...")
    retrieve_bar.progress(50)
    query_feature = model.predict(img_array, verbose=0).flatten()
    time.sleep(0.3)

    retrieve_text.text("Computing similarity scores...")
    retrieve_bar.progress(75)
    similarities = cosine_similarity(
        [query_feature], features
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]
    time.sleep(0.3)

    retrieve_bar.progress(100)
    retrieve_text.text("Done ✅")
    time.sleep(0.5)

    retrieve_bar.empty()
    retrieve_text.empty()

    # =========================
    # DISPLAY RESULTS
    # =========================
    st.subheader("Retrieved Similar Images")

    cols = st.columns(top_k)
    for i, idx in enumerate(top_indices):
        cols[i].image(images[idx])
        cols[i].caption(f"Similarity: {similarities[idx]:.2f}")

else:
    st.info("⬆Please upload an image to start retrieval.")
