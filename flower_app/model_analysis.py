"""Model Analysis Script for Flower Classification Models"""

import os
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from config import MODEL_CONFIG, FLOWER_CLASSES


def analyze_cnn_model(model_path):
    """Analyze CNN model structure and properties"""
    print("=" * 60)
    print("CNN MODEL ANALYSIS")
    print("=" * 60)
    
    try:
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Model summary
        print("\n1. MODEL SUMMARY:")
        print("-" * 40)
        model.summary()
        
        # Model architecture
        print("\n2. MODEL ARCHITECTURE:")
        print("-" * 40)
        print(f"Total layers: {len(model.layers)}")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        
        # Model parameters
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        
        print(f"\n3. MODEL PARAMETERS:")
        print("-" * 40)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        
        # Model layers
        print(f"\n4. LAYER INFORMATION:")
        print("-" * 40)
        for i, layer in enumerate(model.layers):
            print(f"Layer {i}: {layer.name} ({layer.__class__.__name__})")
            print(f"  Input shape: {layer.input_shape}")
            print(f"  Output shape: {layer.output_shape}")
            print(f"  Parameters: {layer.count_params() if hasattr(layer, 'count_params') else 0}")
        
        # Model configuration
        print(f"\n5. MODEL CONFIGURATION:")
        print("-" * 40)
        config = model.get_config()
        print(f"Model name: {config.get('name', 'N/A')}")
        
        return model
        
    except Exception as e:
        print(f"Error analyzing CNN model: {e}")
        return None


def analyze_svm_model(model_path):
    """Analyze SVM model structure and properties"""
    print("=" * 60)
    print("SVM MODEL ANALYSIS")
    print("=" * 60)
    
    try:
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Model type
        print(f"\n1. MODEL TYPE:")
        print("-" * 40)
        print(f"Model class: {model.__class__.__name__}")
        
        # Model parameters
        print(f"\n2. MODEL PARAMETERS:")
        print("-" * 40)
        if hasattr(model, 'get_params'):
            params = model.get_params()
            for key, value in params.items():
                print(f"{key}: {value}")
        
        # Support vectors
        if hasattr(model, 'support_vectors_'):
            print(f"\n3. SUPPORT VECTORS:")
            print("-" * 40)
            print(f"Number of support vectors: {len(model.support_vectors_)}")
            print(f"Support vector shape: {model.support_vectors_.shape}")
        
        # Classes
        if hasattr(model, 'classes_'):
            print(f"\n4. CLASSES:")
            print("-" * 40)
            print(f"Classes: {model.classes_}")
        
        # Decision function
        if hasattr(model, 'decision_function_shape'):
            print(f"\n5. DECISION FUNCTION:")
            print("-" * 40)
            print(f"Decision function shape: {model.decision_function_shape}")
        
        # Kernel
        if hasattr(model, 'kernel'):
            print(f"\n6. KERNEL:")
            print("-" * 40)
            print(f"Kernel type: {model.kernel}")
            if hasattr(model, 'gamma'):
                print(f"Gamma: {model.gamma}")
            if hasattr(model, 'C'):
                print(f"C (regularization): {model.C}")
        
        return model
        
    except Exception as e:
        print(f"Error analyzing SVM model: {e}")
        return None


def compare_models(cnn_model, svm_model):
    """Compare CNN and SVM models"""
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    print("\n1. MODEL TYPE:")
    print("-" * 40)
    print(f"CNN: Deep Learning (Convolutional Neural Network)")
    print(f"SVM: Machine Learning (Support Vector Machine)")
    
    print("\n2. ARCHITECTURE:")
    print("-" * 40)
    if cnn_model:
        print(f"CNN layers: {len(cnn_model.layers)}")
        print(f"CNN parameters: {cnn_model.count_params():,}")
    
    if svm_model and hasattr(svm_model, 'support_vectors_'):
        print(f"SVM support vectors: {len(svm_model.support_vectors_)}")
    
    print("\n3. STRENGTHS:")
    print("-" * 40)
    print("CNN:")
    print("  - Excellent for image recognition")
    print("  - Automatic feature extraction")
    print("  - High accuracy with sufficient data")
    print("  - Can learn complex patterns")
    
    print("\nSVM:")
    print("  - Good for smaller datasets")
    print("  - Effective in high-dimensional spaces")
    print("  - Memory efficient")
    print("  - Versatile with different kernels")
    
    print("\n4. WEAKNESSES:")
    print("-" * 40)
    print("CNN:")
    print("  - Requires large amounts of data")
    print("  - Computationally expensive")
    print("  - Can be prone to overfitting")
    
    print("\nSVM:")
    print("  - Manual feature extraction needed")
    print("  - Less effective for image data")
    print("  - Can be slow with large datasets")


def test_models(cnn_model, svm_model, test_image_path=None):
    """Test models with a sample image"""
    print("=" * 60)
    print("MODEL TESTING")
    print("=" * 60)
    
    # Create a dummy test image if no image provided
    if test_image_path is None or not os.path.exists(test_image_path):
        print("\nCreating dummy test image...")
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_image)
    else:
        test_image = Image.open(test_image_path)
    
    print(f"\nTest image size: {test_image.size}")
    
    # Test CNN model
    if cnn_model:
        print("\n1. CNN PREDICTION:")
        print("-" * 40)
        try:
            img_array = np.array(test_image.resize((224, 224))) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = cnn_model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            print(f"Predicted class: {FLOWER_CLASSES[predicted_class]}")
            print(f"Confidence: {confidence:.4f}")
            print(f"All probabilities:")
            for i, prob in enumerate(predictions[0]):
                print(f"  {FLOWER_CLASSES[i]}: {prob:.4f}")
        except Exception as e:
            print(f"Error testing CNN model: {e}")
    
    # Test SVM model
    if svm_model:
        print("\n2. SVM PREDICTION:")
        print("-" * 40)
        try:
            img_array = np.array(test_image.resize((224, 224))).flatten() / 255.0
            img_array = img_array.reshape(1, -1)
            prediction = svm_model.predict(img_array)[0]
            
            print(f"Predicted class: {FLOWER_CLASSES[int(prediction)]}")
            
            if hasattr(svm_model, 'predict_proba'):
                probabilities = svm_model.predict_proba(img_array)[0]
                confidence = max(probabilities)
                print(f"Confidence: {confidence:.4f}")
                print(f"All probabilities:")
                for i, prob in enumerate(probabilities):
                    print(f"  {FLOWER_CLASSES[i]}: {prob:.4f}")
        except Exception as e:
            print(f"Error testing SVM model: {e}")


def main():
    """Main analysis function"""
    print("FLOWER CLASSIFICATION MODEL ANALYSIS")
    print("=" * 60)
    
    # Get model paths
    cnn_path = os.path.join(os.path.dirname(__file__), MODEL_CONFIG["cnn"]["path"])
    svm_path = os.path.join(os.path.dirname(__file__), MODEL_CONFIG["svm"]["path"])
    
    # Check if models exist
    if not os.path.exists(cnn_path):
        print(f"CNN model not found at: {cnn_path}")
        cnn_model = None
    else:
        cnn_model = analyze_cnn_model(cnn_path)
    
    print("\n")
    
    if not os.path.exists(svm_path):
        print(f"SVM model not found at: {svm_path}")
        svm_model = None
    else:
        svm_model = analyze_svm_model(svm_path)
    
    print("\n")
    
    # Compare models
    compare_models(cnn_model, svm_model)
    
    print("\n")
    
    # Test models
    test_models(cnn_model, svm_model)


if __name__ == "__main__":
    main()