import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, jsonify, Response
from werkzeug.utils import secure_filename
import tensorflow as tf
import keras
import tensorflow_hub as hub
import json
import cv2
import base64
import io
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

try:
    @keras.utils.register_keras_serializable()
    class FeatureExtractorLayer(keras.layers.Layer):
        def __init__(self, model_handle, trainable=False, **kwargs):
            super().__init__(**kwargs)
            self.model_handle = model_handle
            self.trainable = trainable
            self._hub_layer = hub.KerasLayer(
                self.model_handle,
                trainable=self.trainable
            )

        def call(self, inputs):
            return self._hub_layer(inputs)

        def get_config(self):
            config = super().get_config()
            config.update({
                "model_handle": self.model_handle,
                "trainable": self.trainable
            })
            return config

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    model = keras.models.load_model('leaf_disease_classifier.keras')
    print("✅ Model berhasil dimuat.")
except Exception as e:
    print(f"❌ Gagal memuat model. Error: {e}")
    model = None

try:
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    print(f"✅ Nama kelas berhasil dimuat ({len(class_names)} kelas).")
except FileNotFoundError:
    print("❌ File 'class_names.json' tidak ditemukan. Harap buat file ini dari notebook training.")
    class_names = []

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_and_predict(image_input):
    if model is None:
        return "Model tidak tersedia", 0.0

    try:
        # Jika input adalah string (path file), buka dengan PIL
        if isinstance(image_input, str):
            img = Image.open(image_input)
        else:
            # Jika input adalah stream (BytesIO), buka dengan PIL
            img = Image.open(image_input)
        
        # Resize gambar ke 300x300
        img = img.resize((300, 300))
        img = img.convert('RGB')  # Pastikan format RGB
        
        # Konversi ke numpy array
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class_index]
        confidence = np.max(predictions[0]) * 100
        
        return predicted_class_name, confidence
    except Exception as e:
        return f"Error saat prediksi: {e}", 0.0
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = 0
    filename = None
    if request.method == 'POST':
        # Periksa apakah ada file yang diunggah
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # Jika pengguna tidak memilih file, browser mungkin
        # mengirimkan file kosong tanpa nama file.
        if file.filename == '':
            return redirect(request.url)
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Lakukan prediksi
            # Gunakan path file yang sudah disimpan untuk prediksi
            prediction, confidence = process_and_predict(file_path)

    return render_template('index.html', prediction=prediction, confidence=f"{confidence:.2f}%", filename=filename)

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    try:
        data = request.get_json()
        image_data = data['image']
        # Decode base64 string
        image_data = base64.b64decode(image_data.split(',')[1])

        # Ubah data byte menjadi stream di memori
        image_stream = io.BytesIO(image_data)

        # Lakukan prediksi
        prediction, confidence = process_and_predict(image_stream)

        return jsonify({'prediction': prediction, 'confidence': f"{confidence:.2f}%"})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
