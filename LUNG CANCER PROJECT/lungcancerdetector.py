import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set parameters
IMAGE_SIZE = (224, 224)  # Many pre-trained models take 224x224 images
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 3  

# Paths to the dataset directories 
TRAIN_DIR = 'data/train'
VALIDATION_DIR = 'data/validation'
TEST_DIR = 'data/test'

# Data Generators for training, validation, and testing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = valid_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Function to build a model given a backbone architecture.
def build_model(architecture):
    """
    Build a transfer learning model using a given pre-trained architecture.
    architecture: one of ['MobileNetV2', 'VGG16', 'VGG19', 'DenseNet201', 'ResNet101']
    """
    if architecture == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=IMAGE_SIZE + (3,),
            include_top=False,
            weights='imagenet'
        )
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    elif architecture == 'VGG16':
        base_model = tf.keras.applications.VGG16(
            input_shape=IMAGE_SIZE + (3,),
            include_top=False,
            weights='imagenet'
        )
        preprocess_input = tf.keras.applications.vgg16.preprocess_input

    elif architecture == 'VGG19':
        base_model = tf.keras.applications.VGG19(
            input_shape=IMAGE_SIZE + (3,),
            include_top=False,
            weights='imagenet'
        )
        preprocess_input = tf.keras.applications.vgg19.preprocess_input

    elif architecture == 'DenseNet201':
        base_model = tf.keras.applications.DenseNet201(
            input_shape=IMAGE_SIZE + (3,),
            include_top=False,
            weights='imagenet'
        )
        preprocess_input = tf.keras.applications.densenet.preprocess_input

    elif architecture == 'ResNet101':
        # Note: TensorFlow provides a ResNet101 model in keras.applications.
        base_model = tf.keras.applications.ResNet101(
            input_shape=IMAGE_SIZE + (3,),
            include_top=False,
            weights='imagenet'
        )
        preprocess_input = tf.keras.applications.resnet.preprocess_input

    else:
        raise ValueError("Unsupported architecture: choose one of MobileNetV2, VGG16, VGG19, DenseNet201, or ResNet101.")
    
    # Freeze the base model so that its weights are not updated during initial training
    base_model.trainable = False

    # Build the model on top of the pre-trained base
    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = preprocess_input(inputs)    # apply the corresponding pre-processing
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)  # helps prevent overfitting
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Function to train and evaluate a model for a specified backbone
def train_and_evaluate(architecture):
    print(f"\nTraining model based on {architecture} architecture...")
    model = build_model(architecture)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Callbacks: EarlyStopping & ModelCheckpoint
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
        ModelCheckpoint(f'best_model_{architecture}.h5', monitor='val_accuracy', save_best_only=True)
    ]
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(validation_generator)
    print(f"{architecture} - Validation Accuracy: {val_acc * 100:.2f}%")
    
    # Optionally, you could also add testing evaluation here if a test set is provided.
    
    # Plot training history (accuracy and loss curves)
    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{architecture} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{architecture} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    return model, history

if __name__ == '__main__':
    # List of models to compare
    architectures = ['MobileNetV2', 'VGG16', 'VGG19', 'DenseNet201', 'ResNet101']
    
    results = {}
    
    for arch in architectures:
        model, history = train_and_evaluate(arch)
        results[arch] = history.history  # store history to later compare if desired

    # Optionally, save all results for further analysis.
    # For example, you might save results in a JSON file or perform a statistical analysis.

    print("Training for all architectures is complete.")
