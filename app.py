from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
model = load_model('best_model.h5')
print("Model loaded successfully!")

# Class labels
class_names = ['Violence', 'Stampade', 'Normal']

# Preprocessing function
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224)).astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return "No file uploaded"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        ext = file.filename.split('.')[-1].lower()

        if ext in ['jpg', 'jpeg', 'png']:
            frame = cv2.imread(filepath)
            if frame is None:
                return "Invalid image file"

            rgb = preprocess_frame(frame)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv = cv2.resize(hsv, (224, 224)).astype('float32') / 255.0
            hsv = np.expand_dims(hsv, axis=0)

            pred = model.predict([rgb, hsv])
            label = class_names[np.argmax(pred)]
            confidence = float(np.max(pred)) * 100

            return render_template('results.html', 
                                   file_url=filepath, 
                                   label=label, 
                                   confidence=confidence, 
                                   filetype='image')

        elif ext in ['mp4', 'avi', 'mov', 'mkv']:
            cap = cv2.VideoCapture(filepath)
            frame_count = 0
            results = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % 10 == 0:
                    rgb = preprocess_frame(frame)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    hsv = cv2.resize(hsv, (224, 224)).astype('float32') / 255.0
                    hsv = np.expand_dims(hsv, axis=0)
                    pred = model.predict([rgb, hsv])
                    results.append(np.argmax(pred))
                frame_count += 1

            cap.release()
            if not results:
                return "No frames processed from video."

            summary = {cls: results.count(i) for i, cls in enumerate(class_names)}
            total = len(results)
            summary_percent = {k: f"{(v/total)*100:.2f}%" for k, v in summary.items()}

            return render_template('results.html', 
                                   file_url=filepath, 
                                   summary=summary_percent, 
                                   filetype='video')
        else:
            return "Unsupported file format."
    return render_template('index.html')


import os

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))  # Use Render's PORT if available
    app.run(debug=False, host='0.0.0.0', port=port)

