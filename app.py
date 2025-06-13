from flask import Flask, render_template, request, session
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import base64
import os
import logging

app = Flask(__name__)

# Configure a secret key for session (required for session to work)
app.secret_key = 'your-secret-key-here'  # Replace with a secure key in production

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load the model once at startup
model = tf.keras.models.load_model('pneumonia_model.h5')

def preprocess_image(img):
    img = img.convert('RGB')
    w, h = img.size
    min_dim = min(w, h)
    img = img.crop(((w - min_dim) // 2, (h - min_dim) // 2,
                    (w + min_dim) // 2, (h + min_dim) // 2))
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Convert image to base64 for displaying in HTML
def image_to_base64(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    error = None

    if request.method == 'POST':
        if 'file' not in request.files:
            error = "No file uploaded"
            logger.error(error)
            return render_template('index.html', error=error)

        file = request.files['file']
        if file.filename == '':
            error = "No file selected"
            logger.error(error)
            return render_template('index.html', error=error)

        try:
            # Load and preprocess the image
            logger.debug("Loading and preprocessing image")
            img = Image.open(file)
            img_array = preprocess_image(img)
            
            # Convert image to base64 and store in session
            logger.debug("Converting image to base64")
            img_data = image_to_base64(img)
            session['img_data'] = img_data  # Store in session
            logger.debug(f"Image data (first 50 chars): {img_data[:50]}")
            
            # Make prediction
            logger.debug("Making prediction")
            pred = model.predict(img_array)[0][0]
            if pred > 0.5:
                prediction = 'PNEUMONIA'
                confidence = pred * 100  # Convert to percentage
            else:
                prediction = 'NORMAL'
                confidence = (1 - pred) * 100  # Convert to percentage
            logger.info(f"Prediction: {prediction}, Confidence: {confidence}%")
        except Exception as e:
            error = f"Error processing image: {str(e)}"
            logger.error(error)

    # Retrieve img_data from session if available
    img_data = session.get('img_data', None)

    return render_template('index.html', prediction=prediction, confidence=confidence, img_data=img_data, error=error)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)