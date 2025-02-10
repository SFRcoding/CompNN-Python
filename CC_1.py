import cv2
import numpy as np
import os
import json
import shutil
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Utility Functions
# ---------------------------

def is_black_image(image, threshold=3):
    return np.mean(image) < threshold

def preprocess_image(image_path, scale_value, target_width=1096, target_height=949, crop_border=(50, 50, 50, 50)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    try:
        image = cv2.resize(image, None, fx=scale_value, fy=scale_value, interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"Error resizing {image_path}: {e}")
        return None
    
    left, right, top, bottom = crop_border
    h, w = image.shape
    if h <= (top + bottom) or w <= (left + right):
        print(f"Warning: Image {image_path} is too small to crop.")
        return None
    cropped = image[top:h - bottom, left:w - right]
    processed = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    if is_black_image(processed, threshold=1):
        print(f"Warning: Image {image_path} is nearly black and will be skipped.")
        return None
    return processed

def ensure_clean_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

def process_folder(input_folder, output_folder, scale_value, crop_border=(50, 50, 50, 50)):
    if not os.path.exists(input_folder):
        print(f"Error: Input folder {input_folder} does not exist.")
        return {}
    
    ensure_clean_folder(output_folder)
    processed_images = {}
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.bmp'):
            path = os.path.join(input_folder, filename)
            processed = preprocess_image(path, scale_value, crop_border=crop_border)
            if processed is not None:
                embedding = get_embedding(processed).reshape(64, 64)
                pixelated = cv2.resize(embedding, (1096, 949), interpolation=cv2.INTER_NEAREST)
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, pixelated)  # Save resized pixelated image
                processed_images[filename] = processed
            else:
                print(f"Skipping {filename} due to processing issues.")
    return processed_images

def get_embedding(image, size=(64, 64)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA).flatten()

def pair_images(embeddings, image_ids, similarity_threshold=0.8):
    if len(embeddings) == 0:
        print("Error: No valid embeddings available for pairing.")
        return []
    
    sim_matrix = cosine_similarity(embeddings)
    cost_matrix = 1 - sim_matrix
    np.fill_diagonal(cost_matrix, 1.0)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    paired = []
    for i, j in zip(row_ind, col_ind):
        if sim_matrix[i, j] >= similarity_threshold:
            paired.append((image_ids[i], image_ids[j], sim_matrix[i, j]))
    return paired

def average_pair(image1, image2):
    return ((image1.astype(np.float32) + image2.astype(np.float32)) / 2).astype(np.uint8)

def main():
    try:
        with open("config.json", "r") as file:
            config = json.load(file)
    except FileNotFoundError:
        print("Error: config.json file not found.")
        return
    except json.JSONDecodeError:
        print("Error: config.json contains invalid JSON.")
        return

    input_folders = config.get("input_folders", [])
    scale_factors = config.get("scale_factors", [])
    bad_folder = config.get("bad_folder", "")
    bad_scale = config.get("bad_scale", 1.0)
    output_base_folder = config.get("output_base_folder", "")
    
    good_processed_images = {}
    good_image_ids = []
    
    for i, folder in enumerate(input_folders):
        output_folder = os.path.join(output_base_folder, f"{os.path.basename(folder)} Processed")
        processed_images = process_folder(folder, output_folder, scale_value=scale_factors[i])
        for filename, image in processed_images.items():
            key = f"{os.path.basename(folder)}_{filename}"
            good_processed_images[key] = image
            good_image_ids.append(key)
    
    if not good_image_ids:
        print("Error: No images processed successfully.")
        return
    
    embeddings = np.array([get_embedding(good_processed_images[id]) for id in good_image_ids])
    pairs = pair_images(embeddings, good_image_ids, similarity_threshold=0.8)
    
    if not pairs:
        print("Warning: No valid image pairs found.")
    
    avg_folder = os.path.join(output_base_folder, "avg")
    ensure_clean_folder(avg_folder)
    
    for id1, id2, sim in pairs:
        avg_img = average_pair(good_processed_images[id1], good_processed_images[id2])
        avg_output_path = os.path.join(avg_folder, f"Average_{id1}_{id2}.png")
        cv2.imwrite(avg_output_path, avg_img)
    
    bad_output_folder = os.path.join(output_base_folder, "Bad_Part_Processed")
    process_folder(bad_folder, bad_output_folder, scale_value=bad_scale)
    
    print("Processing complete.")

if __name__ == "__main__":
    main()
