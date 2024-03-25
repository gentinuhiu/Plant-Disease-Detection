import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

model = load_model('model.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

def get_result(image_path):
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0] # here
    return labels[np.argmax(predictions)] # -1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    f = request.files['file']
    if f.filename == '':
        return 'No selected file'
    if f:
        filename = secure_filename(f.filename)
        file_path = os.path.join('uploads', filename)
        try:
            # Ensure that the directory exists before saving the file
            os.makedirs('uploads', exist_ok=True)
            f.save(file_path)
            result = get_result(file_path)
            return result
        except Exception as e:
            return str(e)
    return 'Error'

if __name__ == '__main__':
    app.run(debug=True)