from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import gc
import tensorflow as tf

# Limit TensorFlow memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Limit TensorFlow memory usage
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

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

def cleanup_file(filepath):
    """Clean up temporary files after processing"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Error removing file {filepath}: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded"
            
        file = request.files['file']
        if file.filename == '':
            return "No file selected"

        # Create a unique filename to prevent overwriting
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            ext = file.filename.split('.')[-1].lower()

            if ext in ['jpg', 'jpeg', 'png']:
                frame = cv2.imread(filepath)
                if frame is None:
                    cleanup_file(filepath)
                    return "Invalid image file"

                rgb = preprocess_frame(frame)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv = cv2.resize(hsv, (224, 224)).astype('float32') / 255.0
                hsv = np.expand_dims(hsv, axis=0)

                pred = model.predict([rgb, hsv], verbose=0)
                label = class_names[np.argmax(pred)]
                confidence = float(np.max(pred)) * 100
                
                # Force garbage collection to free memory
                del rgb, hsv, pred
                gc.collect()

                return render_template('results.html', 
                                    file_url=filepath, 
                                    label=label, 
                                    confidence=confidence, 
                                    filetype='image')

            elif ext in ['mp4', 'avi', 'mov', 'mkv']:
                cap = cv2.VideoCapture(filepath)
                if not cap.isOpened():
                    cleanup_file(filepath)
                    return "Could not open video file"
                
                frame_count = 0
                results = []
                sample_rate = 30  # Process every 30th frame to reduce memory usage

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % sample_rate == 0:
                        rgb = preprocess_frame(frame)
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        hsv = cv2.resize(hsv, (224, 224)).astype('float32') / 255.0
                        hsv = np.expand_dims(hsv, axis=0)
                        
                        pred = model.predict([rgb, hsv], verbose=0)
                        results.append(np.argmax(pred))
                        
                        # Free memory after each prediction
                        del rgb, hsv, pred
                        gc.collect()
                    
                    frame_count += 1
                    
                    # Process at most 100 frames to prevent timeout
                    if frame_count > sample_rate * 100:
                        break

                cap.release()
                
                if not results:
                    cleanup_file(filepath)
                    return "No frames processed from video."

                summary = {cls: results.count(i) for i, cls in enumerate(class_names)}
                total = len(results)
                summary_percent = {k: f"{(v/total)*100:.2f}%" for k, v in summary.items()}

                return render_template('results.html', 
                                    file_url=filepath, 
                                    summary=summary_percent, 
                                    filetype='video')
            else:
                cleanup_file(filepath)
                return "Unsupported file format."
                
        except Exception as e:
            cleanup_file(filepath)
            return f"Error processing file: {str(e)}"
            
    return render_template('index.html')

if __name__ == "__main__":
    # Use environment variable for port or default to 10000 (Render's preferred port)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, threaded=False)