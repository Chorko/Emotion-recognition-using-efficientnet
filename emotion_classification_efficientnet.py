# Fixing the seed for random number generators
import numpy as np
import random
import tensorflow as tf
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Import necessary modules
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, GlobalAveragePooling2D, Input, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.utils.class_weight import compute_class_weight
import os

# Load EfficientNetV2B2 model and adapt for grayscale input
input_shape = (48, 48, 3)  # EfficientNetV2B2 expects RGB input

# Modify the input layer to accept grayscale images by duplicating the channel
input_tensor = Input(shape=(48, 48, 1))
x = Concatenate(axis=-1)([input_tensor, input_tensor, input_tensor])  # Convert grayscale to RGB

# Load EfficientNetV2B2 and apply to grayscale input
efficient_net_model = EfficientNetV2B2(weights='imagenet', include_top=False, input_shape=input_shape)
x = efficient_net_model(x)

# Global Average Pooling
x = GlobalAveragePooling2D()(x)

# Add Dense layers for classification
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)

x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)

# Output layer for 7 emotion classes
output_layer = Dense(7, activation='softmax')(x)

# Create the model
model = Model(inputs=input_tensor, outputs=output_layer)

# Model Summary
model.summary()

# Data Augmentation
folder_path = 'D:/Emotion_recognition_using_efficientnet/FER/images/'

datagen_train = ImageDataGenerator(
    horizontal_flip=True,
    brightness_range=(0., 2.),
    rescale=1./255,
    shear_range=0.3
)

datagen_validation = ImageDataGenerator(
    rescale=1./255
)

datagen_test = ImageDataGenerator(
    rescale=1./255
)

# Train set
train_set = datagen_train.flow_from_directory(
    folder_path + "train",
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    class_mode='categorical',
    classes=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
    shuffle=True
)

# Validation set
validation_set = datagen_validation.flow_from_directory(
    folder_path + "validation",
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    class_mode='categorical',
    classes=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
    shuffle=True
)

# Test set
test_set = datagen_test.flow_from_directory(
    folder_path + "validation",  
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    class_mode='categorical',
    classes=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
    shuffle=True
)

# Compute class weights dynamically
def get_class_weights(generator):
    class_indices = generator.class_indices
    num_classes = len(class_indices)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=generator.classes
    )
    return dict(enumerate(class_weights))

class_weight_dict = get_class_weights(train_set)

# Callbacks
checkpoint = ModelCheckpoint("D:/Emotion_recognition_using_efficientnet/models/best_model.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
reduce_learningrate = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, min_delta=0.0001)
callbacks_list = [early_stopping, checkpoint, reduce_learningrate]

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss=CategoricalCrossentropy(),
              metrics=['accuracy'])

# Training the model
epochs = 20
history = model.fit(
    train_set,
    validation_data=validation_set,
    epochs=epochs,
    callbacks=callbacks_list,
    class_weight=class_weight_dict  # Apply class weights
)
