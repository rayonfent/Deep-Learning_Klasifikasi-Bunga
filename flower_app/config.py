"""Configuration file for Flower Classification App"""

# Model configuration
MODEL_CONFIG = {
    "cnn": {
        "name": "CNN (EfficientNetB0)",
        "path": "flower_cnn_efficientnetb0.keras",
        "input_size": (224, 224),
        "description": "EfficientNetB0 based CNN model for flower classification"
    },
    "svm": {
        "name": "SVM",
        "path": "svm_mkl_bundle.pkl",
        "input_size": (224, 224),
        "description": "Support Vector Machine model for flower classification"
    }
}

# Flower classes
FLOWER_CLASSES = [
    "Daisy",
    "Dandelion",
    "Rose",
    "Sunflower",
    "Tulip"
]

# App configuration
APP_CONFIG = {
    "title": "🌸 Flower Classification App",
    "description": "Classify flowers using CNN and SVM models",
    "page_icon": "🌸",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Image upload configuration
IMAGE_CONFIG = {
    "max_file_size_mb": 10,
    "allowed_formats": ["jpg", "jpeg", "png"],
    "target_size": (224, 224)
}

# Model cache settings
CACHE_CONFIG = {
    "ttl": 3600,  # Time to live in seconds
    "cache_resource": True
}