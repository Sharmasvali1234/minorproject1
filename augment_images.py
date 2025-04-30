import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img

# Set parameters
input_dir = 'dataset'  # Original images directory
output_dir = 'augmented_dataset'  # New folder to save augmented images
img_size = (224, 224)
augmented_images_per_image = 10

# Create ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Class folders
class_names = ['violence', 'stampade', 'normal']

# Create output directories
for class_name in class_names:
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

# Process each class
for class_name in class_names:
    class_input_path = os.path.join(input_dir, class_name)
    class_output_path = os.path.join(output_dir, class_name)

    image_files = [f for f in os.listdir(class_input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"\nProcessing {class_name} ({len(image_files)} original images)...")

    for img_file in image_files:
        img_path = os.path.join(class_input_path, img_file)

        # Load and resize the image
        img = load_img(img_path, target_size=img_size)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)  # Reshape for generator

        # Generate and save augmented images
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=class_output_path,
                                  save_prefix=class_name, save_format='jpg'):
            i += 1
            if i >= augmented_images_per_image:
                break  # Stop after generating 10 augmentations per image

    print(f"Completed augmentation for class '{class_name}'.")

print("\n✅ All augmentations completed! Check 'augmented_dataset' folder.")
