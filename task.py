import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Data Preparation
def prepare_data(dataset_dir):
    """
    Prepares the dataset by resizing images, normalizing pixel values, and splitting data into train/test sets.
    """
    # Define data generators for augmentation and preprocessing
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=0.2
    )

    train_data = datagen.flow_from_directory(
        dataset_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    validation_data = datagen.flow_from_directory(
        dataset_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    return train_data, validation_data

# Step 2: Model Selection and Training
def build_model():
    """
    Builds and compiles the CNN model.
    """
    base_model = applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False  # Freeze the base model layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Model Evaluation and Freezing
def evaluate_and_freeze_model(model, validation_data):
    """
    Evaluates the model and saves the frozen model for deployment.
    """
    # Evaluate model
    val_loss, val_accuracy = model.evaluate(validation_data)
    print(f"Validation Accuracy: {val_accuracy:.2f}, Validation Loss: {val_loss:.2f}")

    # Save model
    model.save('gender_detection_model.h5')
    print("Model saved as gender_detection_model.h5")

# Step 4: Main Function
def main():
    dataset_dir = "path_to_dataset"  # Replace with the actual path to your dataset

    print("Preparing data...")
    train_data, validation_data = prepare_data(dataset_dir)

    print("Building model...")
    model = build_model()

    print("Training model...")
    model.fit(train_data, epochs=10, validation_data=validation_data)

    print("Evaluating and freezing model...")
    evaluate_and_freeze_model(model, validation_data)

if __name__ == "__main__":
    main()
