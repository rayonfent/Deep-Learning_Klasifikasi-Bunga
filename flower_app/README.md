# 🌸 Flower Classification Streamlit App

A web-based application for classifying flower images using deep learning models (CNN with EfficientNetB0 and SVM).

## Features

- **Multiple Models**: Compare predictions from CNN (EfficientNetB0) and SVM models
- **Image Upload**: Upload flower images in JPG, JPEG, or PNG format
- **Real-time Predictions**: Get instant classification results with confidence scores
- **Probability Visualization**: View prediction probabilities for all flower classes
- **User-friendly Interface**: Clean and intuitive web interface built with Streamlit

## Supported Flower Classes

1. Daisy
2. Dandelion
3. Rose
4. Sunflower
5. Tulip

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone the repository** (or navigate to the project folder):
   ```bash
   cd flower_app
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the application**:
   - The app will open automatically in your default browser
   - If not, navigate to `http://localhost:8501`

3. **Classify a flower**:
   - Upload a flower image using the file uploader
   - Select your preferred model (CNN or SVM) from the sidebar
   - Click "Classify Image" to get predictions
   - View the results with confidence scores and probability distributions

## File Structure

```
flower_app/
├── app.py                           # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── flower_cnn_efficientnetb0.keras # Pre-trained CNN model
├── svm_mkl_bundle.pkl              # Pre-trained SVM model
└── README.md                        # This file
```

## Model Information

### CNN Model (EfficientNetB0)
- **Architecture**: EfficientNetB0 (transfer learning)
- **Input Size**: 224×224 pixels
- **Output**: Classification probabilities for 5 flower classes
- **Performance**: Optimized for high accuracy on flower classification

### SVM Model
- **Algorithm**: Support Vector Machine
- **Kernel**: RBF (Radial Basis Function)
- **Feature Extraction**: Flattened image features
- **Performance**: Fast inference with good accuracy

## Deployment to GitHub

### Step 1: Initialize Git Repository

```bash
git init
git add .
git commit -m "Initial commit: Flower Classification App"
```

### Step 2: Create GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository named `flower-app`
2. Do not initialize with README (we already have one)

### Step 3: Push to GitHub

```bash
git remote add origin https://github.com/your-username/flower-app.git
git branch -M main
git push -u origin main
```

### Step 4: Deploy with Streamlit Cloud

1. Visit [Streamlit Cloud](https://streamlit.io/cloud)
2. Click "New app"
3. Select your GitHub repository: `your-username/flower-app`
4. Set the main file path to: `flower_app/app.py`
5. Click "Deploy"

Your app will now be live at `https://flower-app-username.streamlit.app`

## Troubleshooting

### Model Loading Errors
- Ensure model files (`flower_cnn_efficientnetb0.keras` and `svm_mkl_bundle.pkl`) are in the same directory as `app.py`
- Check that TensorFlow and scikit-learn are properly installed

### Image Upload Issues
- Ensure the image is in JPG, JPEG, or PNG format
- Try with a smaller image file if facing performance issues

### Dependency Issues
- Clear pip cache: `pip cache purge`
- Reinstall requirements: `pip install --upgrade -r requirements.txt`

## System Requirements

- **RAM**: Minimum 2GB (recommended 4GB+)
- **Storage**: Minimum 500MB for models and dependencies
- **Internet**: Required for initial setup and Streamlit Cloud deployment

## Performance Tips

1. **Model Caching**: Models are automatically cached after first load
2. **Image Size**: Smaller images process faster
3. **Batch Processing**: Consider processing multiple images for better throughput

## License

This project is open source and available under the MIT License.

## Support

For issues or questions, please:
1. Check the troubleshooting section
2. Review the Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)
3. Open an issue on GitHub

## Author

Flower Classification App | Built with Streamlit

---

**Happy Flower Classifying! 🌸**