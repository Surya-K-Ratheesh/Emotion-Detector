import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Define constants
IMG_SIZE = 48
NUM_CLASSES = 7
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
DATASET_PATH = "fer2013.csv"
MODEL_PATH = "emotion_model.h5"

def generate_synthetic_dataset(path, num_samples=2000):
    """Generates a synthetic FER2013 CSV dataset for immediate pipeline testing."""
    print(f"Dataset '{path}' not found. Generating synthetic dataset with {num_samples} samples...")
    data = []
    for _ in range(num_samples):
        emotion = np.random.randint(0, NUM_CLASSES)
        pixels = " ".join([str(np.random.randint(0, 256)) for _ in range(IMG_SIZE * IMG_SIZE)])
        usage = "Training" if np.random.rand() < 0.8 else "PublicTest"
        data.append([emotion, pixels, usage])
    
    df = pd.DataFrame(data, columns=["emotion", "pixels", "Usage"])
    df.to_csv(path, index=False)
    print(f"Synthetic dataset saved to {path}")

def load_from_directories(train_dir="train", test_dir="test"):
    """Loads images from train and test directories, resizes, and normalizes them."""
    print(f"Loading train dataset from directory '{train_dir}'...")
    class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode='categorical',
        color_mode='grayscale',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=128,
        shuffle=False,
        class_names=class_names
    )
    
    print(f"Loading test dataset from directory '{test_dir}'...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        label_mode='categorical',
        color_mode='grayscale',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=128,
        shuffle=False,
        class_names=class_names
    )
    
    X_list = []
    y_list = []
    
    for images, labels in train_ds:
        X_list.append(images.numpy())
        y_list.append(labels.numpy())
        
    for images, labels in test_ds:
        X_list.append(images.numpy())
        y_list.append(labels.numpy())
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    # Normalize pixels to range [0, 1]
    X /= 255.0
    
    # Split into train/validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
    )
    
    print(f"Data successfully loaded from directories. Training set: {X_train.shape}, Validation set: {X_val.shape}")
    return X_train, X_val, y_train, y_val

def load_and_preprocess_data(path):
    """Loads the FER2013 data. Prioritizes local train/test folders, falling back to CSV."""
    if os.path.exists("train") and os.path.exists("test"):
        return load_from_directories("train", "test")
        
    if not os.path.exists(path):
        generate_synthetic_dataset(path)
        
    print(f"Loading dataset from {path}...")
    df = pd.read_csv(path)
    
    X = []
    y = []
    
    for idx, row in df.iterrows():
        # Split string pixels into list and convert to float32
        pixels = np.fromstring(row['pixels'], dtype=np.float32, sep=' ')
        # Reshape to (48, 48, 1)
        pixels = pixels.reshape(IMG_SIZE, IMG_SIZE, 1)
        
        X.append(pixels)
        y.append(int(row['emotion']))
        
    X = np.array(X, dtype='float32')
    # Normalize pixels to range [0, 1]
    X /= 255.0
    
    # One-hot encode targets
    y = tf.keras.utils.to_categorical(y, num_classes=NUM_CLASSES)
    
    # Split into train/validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1))
    
    print(f"Data successfully loaded. Training set: {X_train.shape}, Validation set: {X_val.shape}")
    return X_train, X_val, y_train, y_val

def build_cnn_model():
    """Builds a robust, deep CNN architecture with Batch Normalization and Dropout."""
    model = Sequential(name="Emotion_Detection_CNN")
    
    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    # Fully Connected Block
    model.add(Flatten())
    
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # Output layer
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    return model

def main():
    X_train, X_val, y_train, y_val = load_and_preprocess_data(DATASET_PATH)
    
    # Build CNN architecture
    model = build_cnn_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # Data Augmentation configuration
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stop, reduce_lr]
    
    # Train the model
    # Note: Using small epochs and batch size by default so it runs fast if testing
    epochs = 30
    batch_size = 64
    
    print("Starting training...")
    model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"Training completed. Best model saved to '{MODEL_PATH}'.")

if __name__ == "__main__":
    main()
