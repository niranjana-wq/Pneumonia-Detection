# train_model.py
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
import numpy as np
from tqdm import tqdm
import splitfolders

# Step 1: Preprocess images
def process_images(input_folder, output_folder, target_size=(224, 224)):
    categories = ['NORMAL', 'PNEUMONIA']
    for category in categories:
        category_path = os.path.join(input_folder, category)
        save_path = os.path.join(output_folder, category)
        if not os.path.exists(category_path):
            print(f"Skipping {category}: Folder not found")
            continue
        os.makedirs(save_path, exist_ok=True)
        image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Processing {len(image_files)} images in {category}...")
        for filename in tqdm(image_files, desc=f"Processing {category}"):
            try:
                img_path = os.path.join(category_path, filename)
                save_img_path = os.path.join(save_path, filename)
                if os.path.exists(save_img_path):
                    continue
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                min_dim = min(w, h)
                img = img.crop(((w - min_dim) // 2, (h - min_dim) // 2, (w + min_dim) // 2, (h + min_dim) // 2))
                img = img.resize(target_size)
                img.save(save_img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        print(f"Finished {category}")
    print("✅ All images processed")

# Step 2: Split dataset
def split_dataset(input_folder, output_folder):
    splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.7, 0.2, 0.1))
    print("✅ Dataset split into train, validation, and test sets")

# Step 3: Create data generators
def create_data_generators(train_dir, val_dir, test_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='binary')
    val_generator = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='binary')
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False)
    class_weights = {0: len(os.listdir(os.path.join(train_dir, 'PNEUMONIA'))) / len(os.listdir(os.path.join(train_dir, 'NORMAL'))), 1: 1.0}
    return train_generator, val_generator, test_generator, class_weights

# Step 4: Train model
def train_model(train_generator, val_generator, class_weights):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=10, validation_data=val_generator, class_weight=class_weights, verbose=1)
    for layer in base_model.layers[-4:]:
        layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=5, validation_data=val_generator, class_weight=class_weights, verbose=1)
    return model

# Set paths
input_path = r'C:\Users\niran\OneDrive\Desktop\Machine Learning'
processed_path = r'C:\Users\niran\OneDrive\Desktop\Machine Learning\processed_data'
split_path = r'C:\Users\niran\OneDrive\Desktop\Machine Learning\split_data'
model_path = r'C:\Users\niran\OneDrive\Desktop\Machine Learning\pneumonia_model.h5'

# Run steps
process_images(input_path, processed_path)
split_dataset(processed_path, split_path)
train_dir = os.path.join(split_path, 'train')
val_dir = os.path.join(split_path, 'val')
test_dir = os.path.join(split_path, 'test')
train_generator, val_generator, test_generator, class_weights = create_data_generators(train_dir, val_dir, test_dir)
model = train_model(train_generator, val_generator, class_weights)
model.save(model_path)
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy:.2%}, Test Loss: {test_loss:.4f}")