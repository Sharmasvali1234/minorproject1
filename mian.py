import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, concatenate
from tensorflow.keras.applications import MobileNetV2, ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the preprocessed data
X_train = np.load('X_train_rgb.npy')
X_val = np.load('X_val_rgb.npy')
X_test = np.load('X_test.npy')
X_train_hsv = np.load('X_train_hsv.npy')
X_val_hsv = np.load('X_val_hsv.npy')

y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
y_test = np.load('y_test.npy')

# Load class weights
class_weights = np.load('class_weights.npy')
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Convert labels to categorical format for multi-class classification
num_classes = 3  # violence, stampade, normal
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Print dataset information
print(f"Training data shape: {X_train.shape}, Labels: {y_train_cat.shape}")
print(f"Validation data shape: {X_val.shape}, Labels: {y_val_cat.shape}")
print(f"Test data shape: {X_test.shape}, Labels: {y_test_cat.shape}")

# Define image dimensions
img_height, img_width, channels = X_train.shape[1], X_train.shape[2], X_train.shape[3]

# Build a custom model with dual input streams (RGB and HSV)
def build_dual_stream_model():
    # RGB Stream - Using MobileNetV2 as base
    rgb_input = Input(shape=(img_height, img_width, channels), name='rgb_input')
    base_model_rgb = MobileNetV2(include_top=False, weights='imagenet', 
                                input_tensor=rgb_input, input_shape=(img_height, img_width, channels))
    
    # Freeze early layers (transfer learning)
    for layer in base_model_rgb.layers[:100]:
        layer.trainable = False
    
    x_rgb = base_model_rgb.output
    x_rgb = GlobalAveragePooling2D()(x_rgb)
    x_rgb = BatchNormalization()(x_rgb)
    x_rgb = Dense(128, activation='relu')(x_rgb)
    x_rgb = Dropout(0.5)(x_rgb)
    
    # HSV Stream - Using a simpler architecture
    hsv_input = Input(shape=(img_height, img_width, channels), name='hsv_input')
    
    x_hsv = Conv2D(32, (3, 3), activation='relu', padding='same')(hsv_input)
    x_hsv = BatchNormalization()(x_hsv)
    x_hsv = MaxPooling2D((2, 2))(x_hsv)
    
    x_hsv = Conv2D(64, (3, 3), activation='relu', padding='same')(x_hsv)
    x_hsv = BatchNormalization()(x_hsv)
    x_hsv = MaxPooling2D((2, 2))(x_hsv)
    
    x_hsv = Conv2D(128, (3, 3), activation='relu', padding='same')(x_hsv)
    x_hsv = BatchNormalization()(x_hsv)
    x_hsv = MaxPooling2D((2, 2))(x_hsv)
    
    x_hsv = GlobalAveragePooling2D()(x_hsv)
    x_hsv = Dense(64, activation='relu')(x_hsv)
    x_hsv = Dropout(0.5)(x_hsv)
    
    # Combine both streams
    combined = concatenate([x_rgb, x_hsv])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    
    # Output layer
    output = Dense(num_classes, activation='softmax')(combined)
    
    # Create the model
    model = Model(inputs=[rgb_input, hsv_input], outputs=output)
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Alternative: Create a simpler model if memory issues occur
def build_simple_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(img_height, img_width, channels)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Try to build the dual stream model first, fall back to simple model if needed
try:
    print("Building dual stream model...")
    model = build_dual_stream_model()
    dual_stream = True
except Exception as e:
    print(f"Error building dual stream model: {e}")
    print("Falling back to simple model...")
    model = build_simple_model()
    dual_stream = False

# Display model summary
model.summary()

# Set up callbacks for training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

# Train the model
epochs = 50
batch_size = 16

if dual_stream:
    history = model.fit(
        [X_train, X_train_hsv], y_train_cat,
        validation_data=([X_val, X_val_hsv], y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
else:
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

# Save the final model
model.save('final_model.h5')

# Evaluate on test set
if dual_stream:
    test_loss, test_acc = model.evaluate([X_test, X_test], y_test_cat)
    y_pred = model.predict([X_test, X_test])
else:
    test_loss, test_acc = model.evaluate(X_test, y_test_cat)
    y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test_cat, axis=1)

print(f"Test accuracy: {test_acc:.4f}")
print("Classification Report:")
class_names = ['Violence', 'Stampade', 'Normal']
print(classification_report(y_test_classes, y_pred_classes, target_names=class_names))

# Create confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot training history
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.close()

print("Evaluation complete. Results saved to disk.")