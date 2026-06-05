"""Utility functions for Flower Classification App"""

import os
import numpy as np
from PIL import Image
import streamlit as st


def get_model_path(model_type):
    """Get the full path to a model file"""
    from config import MODEL_CONFIG
    model_name = MODEL_CONFIG[model_type]["path"]
    return os.path.join(os.path.dirname(__file__), model_name)


def validate_image(image):
    """Validate uploaded image"""
    if image is None:
        return False, "No image uploaded"
    
    try:
        img = Image.open(image)
        # Check image format
        if img.format.lower() not in ['jpeg', 'jpg', 'png']:
            return False, "Invalid image format. Please upload JPG or PNG."
        return True, "Image is valid"
    except Exception as e:
        return False, f"Error reading image: {str(e)}"


def get_image_info(image):
    """Get information about the image"""
    info = {
        "width": image.width,
        "height": image.height,
        "format": image.format,
        "mode": image.mode,
        "size_mb": image.info.get('size', 0) / (1024 * 1024) if 'size' in image.info else 0
    }
    return info


def format_confidence(confidence):
    """Format confidence as percentage"""
    return f"{confidence * 100:.2f}%"


def get_color_for_confidence(confidence):
    """Get color based on confidence level"""
    if confidence >= 0.8:
        return "green"
    elif confidence >= 0.6:
        return "orange"
    else:
        return "red"


def create_probability_dict(predictions, classes):
    """Create a dictionary of class probabilities"""
    prob_dict = {
        classes[i]: predictions[i] * 100
        for i in range(len(classes))
    }
    return dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))


@st.cache_data
def get_flower_info(flower_name):
    """Get information about a flower"""
    flower_data = {
        "Daisy": {
            "description": "A small, white flower with a yellow center",
            "emoji": "🌼",
            "characteristics": ["White petals", "Yellow center", "Simple structure"]
        },
        "Dandelion": {
            "description": "A bright yellow flower with multiple petals",
            "emoji": "🌼",
            "characteristics": ["Yellow color", "Round shape", "Multiple thin petals"]
        },
        "Rose": {
            "description": "A classic flower with layered petals",
            "emoji": "🌹",
            "characteristics": ["Layered petals", "Often red or pink", "Thorny stem"]
        },
        "Sunflower": {
            "description": "A large flower with a distinctive dark center",
            "emoji": "🌻",
            "characteristics": ["Large size", "Yellow petals", "Dark center", "Follows sun"]
        },
        "Tulip": {
            "description": "An elegant flower with smooth petals",
            "emoji": "🌷",
            "characteristics": ["Cup-shaped", "Smooth petals", "Vibrant colors"]
        }
    }
    return flower_data.get(flower_name, {})