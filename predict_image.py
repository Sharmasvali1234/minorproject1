import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# Preprocess the image: denoise, convert to RGB & HSV, enhance, resize, normalize
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image {image_path}")
    
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    gaussian = cv2.GaussianBlur(img_rgb, (0, 0), 3.0)
    img_rgb = cv2.addWeighted(img_rgb, 1.5, gaussian, -0.5, 0)
    
    img_rgb = cv2.resize(img_rgb, target_size)
    img_hsv = cv2.resize(img_hsv, target_size)
    
    img_rgb = img_rgb.astype('float32') / 255.0
    img_hsv = img_hsv.astype('float32') / 255.0
    
    return img_rgb, img_hsv

# Load the trained model
model = load_model('best_model.h5')
print("Model loaded successfully!")

# Define class names
class_names = ['Violence', 'Stampade', 'Normal']

# Predict function for a single image
def predict_image(image_path):
    img_rgb, img_hsv = preprocess_image(image_path)
    img_rgb = np.expand_dims(img_rgb, axis=0)
    img_hsv = np.expand_dims(img_hsv, axis=0)
    prediction = model.predict([img_rgb, img_hsv])
    
    predicted_class_idx = np.argmax(prediction[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx] * 100

    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {prediction[0][i] * 100:.2f}%")

    # Show the image with predicted label
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

# Run prediction for a single image
if __name__ == "__main__":
    test_image_path = "test/stampade.png"  # Change path as needed
    if os.path.exists(test_image_path):
        predict_image(test_image_path)
    else:
        print(f"Image path not found: {test_image_path}")
