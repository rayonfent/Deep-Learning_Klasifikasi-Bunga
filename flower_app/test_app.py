"""Test script for Flower Classification App"""

import os
import sys
import numpy as np
from PIL import Image
import streamlit as st


def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        import tensorflow
        print("✓ TensorFlow imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TensorFlow: {e}")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import scikit-learn: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Pillow: {e}")
        return False
    
    try:
        import streamlit
        print("✓ Streamlit imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Streamlit: {e}")
        return False
    
    return True


def test_model_files():
    """Test if model files exist"""
    print("\nTesting model files...")
    
    from config import MODEL_CONFIG
    
    cnn_path = MODEL_CONFIG["cnn"]["path"]
    svm_path = MODEL_CONFIG["svm"]["path"]
    
    cnn_exists = os.path.exists(cnn_path)
    svm_exists = os.path.exists(svm_path)
    
    if cnn_exists:
        print(f"✓ CNN model found: {cnn_path}")
    else:
        print(f"✗ CNN model not found: {cnn_path}")
    
    if svm_exists:
        print(f"✓ SVM model found: {svm_path}")
    else:
        print(f"✗ SVM model not found: {svm_path}")
    
    return cnn_exists and svm_exists


def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
    
    try:
        from config import MODEL_CONFIG, FLOWER_CLASSES, APP_CONFIG, IMAGE_CONFIG
        
        print(f"✓ Configuration loaded successfully")
        print(f"  - Flower classes: {len(FLOWER_CLASSES)} classes")
        print(f"  - Model types: {len(MODEL_CONFIG)} models")
        
        # Validate flower classes
        if len(FLOWER_CLASSES) == 5:
            print(f"✓ All 5 flower classes found")
        else:
            print(f"✗ Expected 5 flower classes, got {len(FLOWER_CLASSES)}")
            return False
        
        # Validate model config
        required_models = ["cnn", "svm"]
        for model in required_models:
            if model in MODEL_CONFIG:
                print(f"✓ Model '{model}' configured")
            else:
                print(f"✗ Model '{model}' not configured")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return False


def test_utils():
    """Test utility functions"""
    print("\nTesting utility functions...")
    
    try:
        from utils import (
            validate_image, get_image_info, format_confidence,
            create_probability_dict, get_flower_info
        )
        
        print("✓ All utility functions imported successfully")
        
        # Test format_confidence
        confidence = 0.85
        formatted = format_confidence(confidence)
        if formatted == "85.00%":
            print(f"✓ format_confidence works: {formatted}")
        else:
            print(f"✗ format_confidence incorrect: {formatted}")
            return False
        
        # Test create_probability_dict
        from config import FLOWER_CLASSES
        predictions = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
        prob_dict = create_probability_dict(predictions, FLOWER_CLASSES)
        if len(prob_dict) == 5:
            print(f"✓ create_probability_dict works: {len(prob_dict)} classes")
        else:
            print(f"✗ create_probability_dict incorrect")
            return False
        
        # Test get_flower_info
        flower_info = get_flower_info("Rose")
        if flower_info.get("emoji") == "🌹":
            print(f"✓ get_flower_info works")
        else:
            print(f"✗ get_flower_info incorrect")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error testing utility functions: {e}")
        return False


def test_models_loading():
    """Test if models can be loaded"""
    print("\nTesting model loading...")
    
    try:
        import tensorflow as tf
        from config import MODEL_CONFIG
        
        cnn_path = MODEL_CONFIG["cnn"]["path"]
        
        if os.path.exists(cnn_path):
            try:
                model = tf.keras.models.load_model(cnn_path)
                print(f"✓ CNN model loaded successfully")
                print(f"  - Input shape: {model.input_shape}")
                print(f"  - Output shape: {model.output_shape}")
                print(f"  - Total parameters: {model.count_params():,}")
            except Exception as e:
                print(f"✗ Failed to load CNN model: {e}")
                return False
        else:
            print(f"⚠ CNN model file not found, skipping load test")
        
        # Test SVM model loading
        import pickle
        svm_path = MODEL_CONFIG["svm"]["path"]
        
        if os.path.exists(svm_path):
            try:
                with open(svm_path, 'rb') as f:
                    svm_model = pickle.load(f)
                print(f"✓ SVM model loaded successfully")
                print(f"  - Model type: {svm_model.__class__.__name__}")
            except Exception as e:
                print(f"✗ Failed to load SVM model: {e}")
                return False
        else:
            print(f"⚠ SVM model file not found, skipping load test")
        
        return True
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("FLOWER CLASSIFICATION APP - TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Utility Functions", test_utils),
        ("Model Files", test_model_files),
        ("Model Loading", test_models_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! App is ready to run.")
        return True
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)