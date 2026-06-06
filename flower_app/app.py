import os
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Flower Classification",
    page_icon="flower",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Flower Classification App")
st.caption("Classify flower images using the SVM model.")

app_dir = Path(os.environ.get("APP_DIR", os.path.dirname(__file__))).resolve()
repo_dir = app_dir.parent


def get_candidate_model_paths(filename):
    """Return candidate model locations for local and Streamlit Cloud runs."""
    return [
        app_dir / filename,
        repo_dir / filename,
        Path.cwd() / filename,
        Path.cwd() / "flower_app" / filename,
    ]


def resolve_model_path(filename):
    """Resolve the first existing model path from known candidate locations."""
    for path in get_candidate_model_paths(filename):
        if path.exists():
            return path
    return None


FLOWER_CLASSES = [
    "Daisy",
    "Dandelion",
    "Rose",
    "Sunflower",
    "Tulip",
]


@st.cache_resource
def load_svm_model():
    """Load the SVM model."""
    model_path = resolve_model_path("svm_mkl_bundle.pkl")

    if model_path is None:
        st.error(
            "Error loading SVM model: file `svm_mkl_bundle.pkl` was not found "
            f"in expected locations: {', '.join(str(path) for path in get_candidate_model_paths('svm_mkl_bundle.pkl'))}"
        )
        return None

    try:
        with open(model_path, "rb") as model_file:
            header = model_file.read(16)
            model_file.seek(0)

            if not header.startswith(b"\x80"):
                preview = header.decode("utf-8", errors="replace")
                raise ValueError(
                    "model file is not a valid pickle binary. "
                    f"Resolved path: {model_path}. "
                    f"File header preview: {preview!r}"
                )

            return pickle.load(model_file)
    except Exception as exc:
        st.error(f"Error loading SVM model from `{model_path}`: {exc}")
        return None


def preprocess_image_svm(image, target_size=(224, 224)):
    """Preprocess image for SVM model."""
    prepared_image = image.convert("RGB").resize(target_size)
    image_array = np.array(prepared_image, dtype=np.float32).flatten() / 255.0
    return image_array.reshape(1, -1)


def predict_svm(image):
    """Make prediction using SVM model."""
    model = load_svm_model()
    if model is None:
        return None, None

    image_array = preprocess_image_svm(image)
    prediction = model.predict(image_array)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(image_array)[0]))

    class_index = int(prediction)
    if class_index < 0 or class_index >= len(FLOWER_CLASSES):
        return "Unknown", confidence

    return FLOWER_CLASSES[class_index], confidence


st.sidebar.header("Configuration")
st.sidebar.success("Active model: SVM")

resolved_svm_path = resolve_model_path("svm_mkl_bundle.pkl")
if resolved_svm_path is not None:
    st.sidebar.caption(f"SVM model file: {resolved_svm_path}")
else:
    st.sidebar.warning("SVM model file not found.")

left_column, right_column = st.columns([1, 1])

with left_column:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a flower image",
        type=["jpg", "jpeg", "png"],
        help="Upload an image of a flower for classification.",
    )

with right_column:
    st.subheader("Information")
    st.info(
        """
        Supported flower classes:
        - Daisy
        - Dandelion
        - Rose
        - Sunflower
        - Tulip

        Model:
        - SVM: Support Vector Machine
        """
    )

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
    except Exception as exc:
        image = None
        st.error(f"Failed to open image: {exc}")

    if image is not None:
        st.subheader("Uploaded Image")
        preview_column, detail_column = st.columns([1, 2])

        with preview_column:
            st.image(image, use_column_width=True)

        with detail_column:
            st.write("Image details:")
            st.write(f"- Width: {image.width} px")
            st.write(f"- Height: {image.height} px")
            st.write(f"- Format: {image.format}")

        st.subheader("Prediction Results")

        if st.button("Classify Image", use_container_width=True):
            with st.spinner("Processing image..."):
                flower_class, confidence = predict_svm(image)

            if flower_class is None:
                st.error("Failed to make prediction. Please check if the SVM model is available.")
            else:
                result_column, confidence_column = st.columns([1, 1])

                with result_column:
                    st.metric("Predicted Class", flower_class)

                with confidence_column:
                    if confidence is None:
                        st.metric("Confidence", "N/A")
                    else:
                        st.metric("Confidence", f"{confidence * 100:.2f}%")
                        st.progress(
                            float(confidence),
                            text=f"Confidence: {confidence * 100:.2f}%",
                        )

st.divider()
st.caption("Flower Classification App | Built with Streamlit")
