import tensorflow as tf
import numpy as np
import cv2
import os
import json
import random
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_images_from_folder(folder, file_extensions=(".png", ".bmp")):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(file_extensions):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames

def resize_image(image, target_size=(128, 128)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def preprocess_for_nn(image):
    img = image.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def add_noise(image, noise_factor=0.7):
    noisy = image + noise_factor * np.random.randn(*image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def generate_pairs(avg_images, good_images):
    pairs = []
    for avg_img in avg_images:
        good_img = random.choice(good_images)
        pairs.append((avg_img, good_img, 1))
        noisy_img = add_noise(good_img, noise_factor=0.7)
        pairs.append((avg_img, noisy_img, 0))
    return pairs

def create_base_network(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu')
    ])
    return model

def build_siamese_network(input_shape):
    base_network = create_base_network(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    emb_a = base_network(input_a)
    emb_b = base_network(input_b)
    diff = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]), output_shape=(128,))([emb_a, emb_b])
    similarity = layers.Dense(1, activation='sigmoid')(diff)
    model = models.Model(inputs=[input_a, input_b], outputs=similarity)
    return model

def main_training():
    with open("train_config.json", "r") as file:
        config = json.load(file)
    
    base_folder = config["base_path"]
    good_folders = [f for f in os.listdir(base_folder) if f.startswith("nieuszkodzony_") and f.endswith("Processed")]
    avg_folder = os.path.join(base_folder, "avg")
    
    if not os.path.exists(avg_folder):
        print("Error: 'avg' folder not found.")
        return
    
    avg_images, _ = load_images_from_folder(avg_folder)
    avg_images = [resize_image(img) for img in avg_images]
    
    good_images = []
    for folder in good_folders:
        folder_path = os.path.join(base_folder, folder)
        imgs, _ = load_images_from_folder(folder_path)
        imgs = [resize_image(img) for img in imgs]
        good_images.extend(imgs)
    
    if not good_images:
        print("Error: No good images found.")
        return
    
    pairs = generate_pairs(avg_images, good_images)
    X1, X2, y = [], [], []
    for img1, img2, label in pairs:
        X1.append(preprocess_for_nn(img1))
        X2.append(preprocess_for_nn(img2))
        y.append(label)
    X1, X2, y = np.array(X1), np.array(X2), np.array(y)
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(X1, X2, y, test_size=0.2, random_state=42)
    
    input_shape = (128, 128, 1)
    siamese_model = build_siamese_network(input_shape)
    siamese_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    siamese_model.summary()
    
    history = siamese_model.fit([X1_train, X2_train], y_train,
                                validation_data=([X1_val, X2_val], y_val),
                                epochs=10, batch_size=16)
    
    siamese_model.save("siamese_model.h5")
    print("Model saved as 'siamese_model.h5'.")
    
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Siamese Network Training")
    plt.show()

if __name__ == "__main__":
    main_training()
