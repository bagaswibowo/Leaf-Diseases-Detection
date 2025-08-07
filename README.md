# Leaf Disease Classification

A Flask-based web application for real-time detection and classification of plant leaf diseases using deep learning. This application leverages TensorFlow and TensorFlow Hub to provide accurate predictions through both image upload and live camera functionality.

## 🌟 Features

- **Image Upload Classification**: Upload leaf images (JPG, PNG, JPEG) to detect diseases with confidence scores
- **Live Camera Detection**: Real-time disease detection using device camera
- **89+ Disease Classes**: Supports detection of various leaf diseases across multiple plant species including:
  - Apple, Banana, Corn, Grape, Tomato, Potato, and many more
  - Healthy vs. diseased leaf classification
- **Modern Web Interface**: Responsive Bootstrap-based UI with intuitive design
- **High Accuracy**: Uses EfficientNet-B3 pre-trained model with transfer learning
- **Mobile-Friendly**: Optimized for both desktop and mobile devices

## 🛠️ Technology Stack

- **Backend**: Flask (Python 3.11+)
- **Machine Learning**: TensorFlow 2.15.0, TensorFlow Hub, Keras
- **Image Processing**: OpenCV, Pillow (PIL)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 4
- **Model Architecture**: EfficientNet-B3 with custom classification layers
- **Dependencies**: NumPy, Werkzeug

## 📋 Prerequisites

- Python 3.11 or higher (recommended for compatibility)
- Webcam/camera for live detection feature
- Modern web browser with camera support

## ⚙️ Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/bagaswibowo/Leaf-Diseases-Detection.git
cd Leaf-Diseases-Detection
```

### 2. Create Virtual Environment
Using Python 3.11 is recommended for optimal compatibility:
```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation
Test if all dependencies are installed correctly:
```bash
python -c "import tensorflow as tf; import tensorflow_hub as hub; print('✅ All dependencies installed successfully')"
```

## 🚀 Running the Application

### Local Development
```bash
python app.py
```

The application will start on `http://localhost:8080` by default.

### Access Points
- **Image Upload**: `http://localhost:8080/`
- **Live Camera**: Use the camera tab in the main interface

### Mobile Access (HTTPS Required)
For mobile camera access, you need HTTPS. Use ngrok for testing:

1. Install [ngrok](https://ngrok.com/download)
2. Run ngrok tunnel:
   ```bash
   ngrok http 8080
   ```
3. Access the HTTPS URL provided by ngrok on your mobile device

## 📁 Project Structure

```
Leaf-Diseases-Detection/
├── app.py                          # Main Flask application
├── leaf_disease_classifier.keras   # Pre-trained model file
├── class_names.json               # Disease class names (89 classes)
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── klasifikasi_penyakit_daun.ipynb # Jupyter notebook for training
├── templates/
│   └── index.html                 # Main web interface
└── static/
    └── uploads/                   # Uploaded images storage
```

## 🔧 Model Architecture

The application uses a transfer learning approach with:
- **Base Model**: EfficientNet-B3 from TensorFlow Hub
- **Custom Layers**: Dense layers with batch normalization and dropout
- **Output**: 89 disease classes with softmax activation
- **Input Size**: 300x300x3 RGB images
- **Preprocessing**: Images normalized to [0,1] range

## 📊 Supported Disease Classes

The model can detect 89+ different conditions including:
- **Fruits**: Apple (scab, black rot, cedar rust), Banana (sigatoka, mosaic virus)
- **Vegetables**: Tomato (blight, leaf mold, mosaic), Potato (early/late blight)
- **Crops**: Corn (rust, leaf blight), Grape (black rot, leaf blight)
- **Others**: Cherry, Peach, Pepper, Strawberry, and many more
- **Healthy**: Healthy leaf classifications for comparison

## � Troubleshooting

### Common Issues

1. **TensorFlow Hub Import Error**
   ```bash
   # Ensure you're using Python 3.11 and correct TensorFlow versions
   pip install tensorflow==2.15.0 tensorflow-hub==0.15.0
   ```

2. **Model Loading Error**
   - Ensure `leaf_disease_classifier.keras` exists in the project root
   - Check if `class_names.json` is present and readable

3. **Camera Access Denied**
   - Use HTTPS for mobile devices
   - Check browser permissions for camera access
   - Ensure you're accessing from localhost or a secure connection

4. **Memory Issues**
   - Model loading requires ~2GB RAM
   - Consider closing other applications if experiencing issues

## 🔄 Development

### Code Structure
- `app.py`: Main Flask application with routes and prediction logic
- `process_and_predict()`: Core function for image preprocessing and prediction
- `FeatureExtractorLayer`: Custom Keras layer for TensorFlow Hub integration

### Key Functions
- Image upload handling with secure filename validation
- Real-time prediction from base64 encoded camera frames
- Error handling for missing models or invalid inputs

## 📝 API Endpoints

- `GET /`: Main interface with upload and camera tabs
- `POST /`: Image upload and prediction
- `POST /predict_frame`: Real-time camera frame prediction
- `GET /display/<filename>`: Serve uploaded images

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow and TensorFlow Hub teams for pre-trained models
- EfficientNet architecture by Google Research
- Bootstrap team for responsive UI components
- Plant pathology datasets contributors

## 📧 Contact

For questions or support, please open an issue on GitHub or contact the repository maintainer.

---

**Note**: This application is for educational and research purposes. For critical agricultural decisions, consult with professional plant pathologists.
