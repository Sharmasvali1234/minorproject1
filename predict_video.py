import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load the model
model = load_model('best_model.h5')
print("Model loaded successfully!")

class_names = ['Violence', 'Stampade', 'Normal']

def preprocess_frame(frame, target_size=(224, 224)):
    frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    gaussian = cv2.GaussianBlur(frame_rgb, (0, 0), 3.0)
    frame_rgb = cv2.addWeighted(frame_rgb, 1.5, gaussian, -0.5, 0)

    frame_rgb = cv2.resize(frame_rgb, target_size)
    frame_hsv = cv2.resize(frame_hsv, target_size)

    frame_rgb = frame_rgb.astype('float32') / 255.0
    frame_hsv = frame_hsv.astype('float32') / 255.0

    return np.expand_dims(frame_rgb, axis=0), np.expand_dims(frame_hsv, axis=0)

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    class_counts = {class_name: 0 for class_name in class_names}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 != 0:  # skip frames to speed up
            continue

        rgb, hsv = preprocess_frame(frame)
        prediction = model.predict([rgb, hsv])
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx] * 100
        class_counts[predicted_class] += 1

        label = f"{predicted_class}: {confidence:.2f}%"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nSummary of video predictions:")
    total = sum(class_counts.values())
    for class_name, count in class_counts.items():
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{class_name}: {count} frames ({percentage:.2f}%)")

if __name__ == "__main__":
    video_path = "testvideos/stamapade5.mp4"
    if os.path.exists(video_path):
        predict_video(video_path)
    else:
        print(f"Video not found at path: {video_path}")
