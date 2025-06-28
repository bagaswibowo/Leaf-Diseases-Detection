# Leaf Disease Classification

A Flask-based web application for real-time detection and classification of plant leaf diseases. This application can make predictions based on user-uploaded images or through a live camera stream.

## ✨ Key Features

- **Classification via Image Upload**: Users can upload an image file (JPG, PNG, JPEG) to get a leaf disease prediction along with a confidence score.
- **Live Camera Classification**: Uses the device's camera (PC or mobile) to perform real-time disease detection without needing to capture a photo first.
- **Responsive Web Interface**: A simple and user-friendly interface built with Bootstrap.

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: TensorFlow, Keras
- **Image Processing**: OpenCV, Pillow
- **Others**: NumPy

## ⚙️ Installation and Setup

Follow these steps to run this project in your local environment.

**1. Clone the Repository**
```bash
git clone <YOUR_REPOSITORY_URL>
cd <PROJECT_DIRECTORY_NAME>
```

**2. Create and Activate a Virtual Environment**
Using a virtual environment is highly recommended.
```bash
# Create an environment (e.g., named 'venv')
python3 -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

**3. Install Dependencies**
Install all required Python libraries from the `requirements.txt` file and add OpenCV.
```bash
pip install -r requirements.txt
pip install opencv-python
```

## 🚀 How to Run the Application

**1. Run the Flask Server**
Once all dependencies are installed, run the application with the command:
```bash
python app.py
```

The server will start and be accessible at `http://127.0.0.1:5000` (or another port if you have changed it).

**2. Access the Application**
- **For Image Upload**: Open your browser and go to `http://127.0.0.1:5000/`.
- **For Live Camera**: Open your browser and go to `http://127.0.0.1:5000/live`.

**3. Accessing from a Mobile Phone (Requires `ngrok`)**
To use the live camera feature on a mobile phone, you need to expose your local server to the internet over a secure HTTPS connection. Mobile browsers will not grant camera permissions over an insecure HTTP connection.

- Download and install [ngrok](https://ngrok.com/download).
- Run the following command in a new terminal to create a tunnel to the port your Flask app is using (e.g., 5000):
  ```bash
  ./ngrok http 5000
  ```
- `ngrok` will provide you with a public URL in the format `https://<random-string>.ngrok.io`.
- Open the URL `https://<random-string>.ngrok.io/live` in your mobile browser to access the live camera feature.

## 📁 Project Structure

```
. 
├── app.py                     # Main Flask application logic (routes, prediction)
├── leaf_disease_classifier.keras  # Machine Learning model file
├── class_names.json           # List of class/disease names
├── requirements.txt           # Python dependency list
├── README.md                  # Project documentation
├── templates/                   # Folder for HTML files
│   ├── index.html             # Page for image upload
│   └── live.html              # Page for live camera prediction
└── static/                    # Folder for static files
    └── uploads/               # Directory to store uploaded images
```
