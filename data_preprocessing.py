import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Directory containing the folders for different classes
dataset_dir = 'augmented_dataset'  # Path to the dataset directory

# Image parameters
img_size = (224, 224)  # Resize all images to 224x224
batch_size = 32

# Initialize lists to store images and their labels
images = []
labels = []
hsv_images = []  # Additional color space

# Define the class labels - corrected spelling
class_labels = {'violence': 0, 'stampade': 1, 'normal': 2}  # You may want to fix 'stampade' to 'stampede' in your folders

# Class counts for monitoring balance
class_counts = {class_name: 0 for class_name in class_labels.keys()}

# Loop through each class folder and process images
for class_name, class_label in class_labels.items():
    class_folder = os.path.join(dataset_dir, class_name)
    
    if not os.path.exists(class_folder):
        print(f"Warning: Class folder '{class_folder}' not found!")
        continue
        
    # Loop through all images in the current class folder
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(class_folder) if f.lower().endswith(valid_extensions)]
    
    print(f"Processing {len(image_files)} images for class '{class_name}'")
    
    for filename in image_files:
        img_path = os.path.join(class_folder, filename)
        
        # Read the image
        img = cv2.imread(img_path)
        if img is None:  # Check if the image was loaded successfully
            print(f"Warning: Could not load image {img_path}")
            continue
        
        # Apply denoising (reduces noise while preserving edges)
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            
        # Convert from BGR to RGB (most models are trained on RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Also create HSV version for additional features
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Apply edge enhancement using unsharp masking
        gaussian = cv2.GaussianBlur(img_rgb, (0, 0), 3.0)
        img_rgb = cv2.addWeighted(img_rgb, 1.5, gaussian, -0.5, 0)
        
        # Resize to the target size
        img_rgb = cv2.resize(img_rgb, img_size)
        img_hsv = cv2.resize(img_hsv, img_size)
        
        # Append the image and its corresponding label to the lists
        images.append(img_rgb)
        hsv_images.append(img_hsv)
        labels.append(class_label)
        class_counts[class_name] += 1

# Print class distribution
print("Class distribution:")
for class_name, count in class_counts.items():
    print(f"  {class_name}: {count} images")

# Convert images and labels to NumPy arrays
images = np.array(images)
hsv_images = np.array(hsv_images)
labels = np.array(labels)

# Normalize images to [0,1] range
images = images.astype('float32') / 255.0
hsv_images = hsv_images.astype('float32') / 255.0

# Split data into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

# Split HSV images as well
X_hsv_train, X_hsv_val, _, _ = train_test_split(
    hsv_images, labels, test_size=0.2, random_state=42, stratify=labels
)

# Calculate class weights to handle imbalance
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print("Class weights to handle imbalance:", class_weight_dict)

# Create more advanced data generators for training
train_datagen = ImageDataGenerator(
    # No rescaling needed since we already normalized
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,  # Increased zoom range
    horizontal_flip=True,
    vertical_flip=False,  # Violence/stampede scenes usually have ground at bottom
    brightness_range=[0.8, 1.2],  # Vary brightness
    channel_shift_range=0.2,  # Slight color variations
    fill_mode='nearest'
)

# Validation generator (no augmentation)
val_datagen = ImageDataGenerator()

# Apply data augmentation to the training set
train_generator = train_datagen.flow(
    X_train, y_train, 
    batch_size=batch_size
)

val_generator = val_datagen.flow(
    X_val, y_val, 
    batch_size=batch_size
)

# Save preprocessed data in multiple formats
np.save('X_train_rgb.npy', X_train)
np.save('X_val_rgb.npy', X_val)
np.save('X_train_hsv.npy', X_hsv_train)
np.save('X_val_hsv.npy', X_hsv_val)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('class_weights.npy', class_weights)

print("Preprocessed data saved successfully!")
print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
print(f"HSV training data shape: {X_hsv_train.shape}, HSV validation data shape: {X_hsv_val.shape}")

# Additional: create a small test set for final evaluation
# This helps prevent overfitting to the validation set
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
print(f"Test data shape: {X_test.shape}")