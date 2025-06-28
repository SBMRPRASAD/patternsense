import os
import numpy as np
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your model
try:
    model = load_model("fabric_model.h5")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

classes = ['floral', 'geometric', 'polka_dot', 'stripes', 'plain', 'tribes']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Please select an image file.', 'error')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Invalid file type. Upload PNG, JPG, JPEG, or WEBP.', 'error')
            return redirect(request.url)

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Check if image is valid
            try:
                with Image.open(filepath) as img:
                    img.verify()
            except Exception:
                flash('Uploaded file is not a valid image.', 'error')
                os.remove(filepath)
                return redirect(request.url)

            # Reload for processing
            img = load_img(filepath, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            if model is None:
                flash('Model is not loaded.', 'error')
                return redirect(request.url)

            prediction = model.predict(img_array)
            predicted_class = classes[np.argmax(prediction)]
            confidence = round(np.max(prediction) * 100, 2)

            # ✅ Make sure `img_path` is relative to `static/`
            return render_template('results.html',
                                   prediction=predicted_class,
                                   confidence=confidence,
                                   img_path=f"uploads/{filename}")
        except Exception as e:
            flash(f"Error processing the image: {str(e)}", 'error')
            return redirect(request.url)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
