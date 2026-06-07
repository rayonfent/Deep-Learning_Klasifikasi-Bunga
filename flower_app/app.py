import os
import pickle
from pathlib import Path

import joblib
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
st.caption("Classify flower images using CNN and SVM models simultaneously.")

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
    """Load the SVM/MKL bundle.

    The bundled SVM file was saved with joblib, not plain pickle. Loading it
    with pickle can fail on Streamlit Cloud with errors such as
    "invalid load key". Use joblib first and keep pickle only as a fallback.
    """
    model_path = resolve_model_path("svm_mkl_bundle.pkl")

    if model_path is None:
        st.error(
            "Error loading SVM model: file `svm_mkl_bundle.pkl` was not found "
            f"in expected locations: {', '.join(str(path) for path in get_candidate_model_paths('svm_mkl_bundle.pkl'))}"
        )
        return None

    try:
        return joblib.load(model_path)
    except Exception as joblib_exc:
        try:
            with open(model_path, "rb") as model_file:
                return pickle.load(model_file)
        except Exception as pickle_exc:
            st.warning(
                "SVM model could not be loaded. "
                f"joblib error: {joblib_exc}; pickle fallback error: {pickle_exc}"
            )
            return None


@st.cache_resource
def load_cnn_model():
    """Load the CNN Keras model if TensorFlow is available."""
    model_path = resolve_model_path("flower_cnn_efficientnetb0.keras")

    if model_path is None:
        st.error(
            "Error loading CNN model: file `flower_cnn_efficientnetb0.keras` was not found "
            f"in expected locations: {', '.join(str(path) for path in get_candidate_model_paths('flower_cnn_efficientnetb0.keras'))}"
        )
        return None

    try:
        from tensorflow import keras
    except Exception as exc:
        st.warning(
            "CNN model is available, but TensorFlow is not installed in this deployment runtime. "
            "SVM prediction will still run. "
            f"TensorFlow import error: {exc}"
        )
        return None

    try:
        return keras.models.load_model(model_path)
    except Exception as exc:
        st.error(f"Error loading CNN model from `{model_path}`: {exc}")
        return None


def preprocess_image_svm(image, target_size=(224, 224)):
    """Preprocess image for SVM model."""
    prepared_image = image.convert("RGB").resize(target_size)
    image_array = np.array(prepared_image, dtype=np.float32).flatten() / 255.0
    return image_array.reshape(1, -1)


def preprocess_image_cnn(image, target_size=(224, 224)):
    """Preprocess image for CNN model."""
    prepared_image = image.convert("RGB").resize(target_size)
    image_array = np.array(prepared_image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def get_svm_estimator(model_bundle):
    """Return the actual SVM estimator from either a model object or a bundle dict."""
    if not isinstance(model_bundle, dict):
        return model_bundle

    for key in ("svm", "model", "classifier", "clf", "svc"):
        candidate = model_bundle.get(key)
        if candidate is not None and hasattr(candidate, "predict"):
            return candidate

    for candidate in model_bundle.values():
        if hasattr(candidate, "predict"):
            return candidate

    return None


def get_svm_class_names(model_bundle):
    """Return class names stored in the bundle, falling back to the default list."""
    if isinstance(model_bundle, dict):
        class_names = model_bundle.get("class_names")
        if class_names is not None:
            return list(class_names)
    return FLOWER_CLASSES


def build_svm_feature_candidates(image, model_bundle):
    """Build several safe feature candidates for different SVM bundle formats."""
    raw_features = preprocess_image_svm(image)
    candidates = [raw_features]

    if isinstance(model_bundle, dict):
        transformed_parts = []
        for key in ("kmeans_internal", "kmeans_boundary", "kmeans"):
            transformer = model_bundle.get(key)
            if transformer is not None and hasattr(transformer, "transform"):
                try:
                    transformed_parts.append(transformer.transform(raw_features))
                except Exception:
                    pass

        if transformed_parts:
            candidates.extend(transformed_parts)
            try:
                candidates.append(np.concatenate(transformed_parts, axis=1))
            except Exception:
                pass

    return candidates


def predict_with_svm_estimator(estimator, feature_candidates):
    """Try SVM prediction with available feature formats."""
    last_error = None

    for features in feature_candidates:
        try:
            prediction = estimator.predict(features)[0]
            confidence = None
            if hasattr(estimator, "predict_proba"):
                confidence = float(np.max(estimator.predict_proba(features)[0]))
            return prediction, confidence
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"No compatible SVM feature format found. Last error: {last_error}")


def predict_svm(image):
    """Make prediction using SVM model or SVM/MKL bundle."""
    model_bundle = load_svm_model()
    if model_bundle is None:
        return None, None

    estimator = get_svm_estimator(model_bundle)
    if estimator is None:
        st.warning("SVM bundle was loaded, but no estimator with a predict method was found.")
        return None, None

    try:
        feature_candidates = build_svm_feature_candidates(image, model_bundle)
        prediction, confidence = predict_with_svm_estimator(estimator, feature_candidates)
    except Exception as exc:
        st.warning(f"SVM prediction could not be completed with the deployed bundle: {exc}")
        return None, None

    class_names = get_svm_class_names(model_bundle)

    try:
        class_index = int(prediction)
        if class_index < 0 or class_index >= len(class_names):
            return str(prediction), confidence
        return class_names[class_index], confidence
    except Exception:
        return str(prediction), confidence


def predict_cnn(image):
    """Make prediction using CNN model."""
    model = load_cnn_model()
    if model is None:
        return None, None

    image_array = preprocess_image_cnn(image)
    predictions = model.predict(image_array, verbose=0)[0]

    class_index = int(np.argmax(predictions))
    confidence = float(predictions[class_index])

    if class_index < 0 or class_index >= len(FLOWER_CLASSES):
        return "Unknown", confidence

    return FLOWER_CLASSES[class_index], confidence


st.sidebar.header("Configuration")
st.sidebar.success("Active models: CNN + SVM")

resolved_cnn_path = resolve_model_path("flower_cnn_efficientnetb0.keras")
if resolved_cnn_path is not None:
    st.sidebar.caption(f"CNN model file: {resolved_cnn_path}")
else:
    st.sidebar.warning("CNN model file not found.")

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

        Models used simultaneously:
        - CNN: EfficientNetB0 (.keras)
        - SVM: Support Vector Machine (.pkl)
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
            with st.spinner("Processing image with CNN and SVM..."):
                cnn_class, cnn_confidence = predict_cnn(image)
                svm_class, svm_confidence = predict_svm(image)

            cnn_column, svm_column = st.columns(2)

            with cnn_column:
                st.markdown("### CNN Result")
                if cnn_class is None:
                    st.error("Failed to make CNN prediction. Please check if the CNN model is available.")
                else:
                    st.metric("Predicted Class", cnn_class)
                    if cnn_confidence is None:
                        st.metric("Confidence", "N/A")
                    else:
                        st.metric("Confidence", f"{cnn_confidence * 100:.2f}%")
                        st.progress(
                            float(cnn_confidence),
                            text=f"Confidence: {cnn_confidence * 100:.2f}%",
                        )

            with svm_column:
                st.markdown("### SVM Result")
                if svm_class is None:
                    st.warning("SVM prediction is currently unavailable for this deployed model bundle.")
                else:
                    st.metric("Predicted Class", svm_class)
                    if svm_confidence is None:
                        st.metric("Confidence", "N/A")
                    else:
                        st.metric("Confidence", f"{svm_confidence * 100:.2f}%")
                        st.progress(
                            float(svm_confidence),
                            text=f"Confidence: {svm_confidence * 100:.2f}%",
                        )

            if cnn_class is not None and svm_class is not None:
                st.subheader("Summary")
                if cnn_class == svm_class:
                    st.success(f"Both models agree on the prediction: {cnn_class}")
                else:
                    st.warning(
                        f"Models disagree. CNN predicts {cnn_class}, while SVM predicts {svm_class}."
                    )

st.divider()
st.caption("Flower Classification App | Built with Streamlit")