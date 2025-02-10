import cv2
import numpy as np
import os
import json
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Utility Functions
# ---------------------------

def is_black_image(image, threshold=3):
    return np.mean(image) < threshold

def preprocess_image(image_path, scale_value, target_width=796, target_height=749):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    image = cv2.resize(image, None, fx=scale_value, fy=scale_value, interpolation=cv2.INTER_AREA)
    processed = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    if is_black_image(processed):
        return None
    
    return processed

def process_folder(input_folder, output_folder, scale_value):
    os.makedirs(output_folder, exist_ok=True)
    processed_images = {}
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.bmp'):
            path = os.path.join(input_folder, filename)
            processed = preprocess_image(path, scale_value)
            if processed is not None:
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, processed)
                processed_images[filename] = processed
            else:
                print(f"Skipping image {filename}")
    
    return processed_images

def get_embedding(image, size=(64, 64)):
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return small.flatten()

def pair_images(embeddings, image_ids, similarity_threshold=0.8):
    sim_matrix = cosine_similarity(embeddings)
    cost_matrix = 1 - sim_matrix  
    np.fill_diagonal(cost_matrix, 1.0)
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    paired = []
    used = set()
    
    for i, j in zip(row_ind, col_ind):
        if i in used or j in used:
            continue
        if sim_matrix[i, j] >= similarity_threshold:
            paired.append((image_ids[i], image_ids[j], sim_matrix[i, j]))
            used.add(i)
            used.add(j)
    
    return paired

def average_pair(image1, image2):
    return ((image1.astype(np.float32) + image2.astype(np.float32)) / 2).astype(np.uint8)

def clean_output_folders(master_output_folder):
    if not os.path.exists(master_output_folder):
        os.makedirs(master_output_folder, exist_ok=True)
        return
    
    for folder in os.listdir(master_output_folder):
        subfolder_path = os.path.join(master_output_folder, folder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                os.remove(file_path)

def main():
    with open("config.json", "r") as f:
        config = json.load(f)
    
    master_output_folder = config["good_parts"]["output_base_folder"]
    clean_output_folders(master_output_folder)
    
    all_processed_images = {}
    
    for entry in config["good_parts"]["input_folders"]:
        input_folder = entry["folder"]
        scale_value = entry["scale"]
        output_folder = os.path.join(master_output_folder, os.path.basename(input_folder) + " PROCESSED")
        
        processed_images = process_folder(input_folder, output_folder, scale_value)
        all_processed_images.update({f"good_{k}": v for k, v in processed_images.items()})
    
    bad_input_folder = config["bad_parts"]["input_folder"]
    bad_scale = config["bad_parts"]["scale"]
    bad_output_folder = os.path.join(master_output_folder, os.path.basename(bad_input_folder) + " PROCESSED")
    
    processed_bad_images = process_folder(bad_input_folder, bad_output_folder, bad_scale)
    all_processed_images.update({f"bad_{k}": v for k, v in processed_bad_images.items()})
    
    if len(all_processed_images) < 2:
        print("Not enough images to form pairs. Exiting.")
        return
    
    image_ids = list(all_processed_images.keys())
    embeddings = np.array([get_embedding(img) for img in all_processed_images.values()])
    
    pairs = pair_images(embeddings, image_ids, similarity_threshold=0.8)
    if not pairs:
        print("No pairs found that meet the similarity threshold.")
        return
    
    avg_output_folder = os.path.join(master_output_folder, "avg PROCESSED")
    os.makedirs(avg_output_folder, exist_ok=True)
    
    for idx, (id1, id2, sim) in enumerate(pairs):
        img1 = all_processed_images[id1]
        img2 = all_processed_images[id2]
        avg_img = average_pair(img1, img2)
        avg_output_path = os.path.join(avg_output_folder, f"Average_{idx+1}.png")
        cv2.imwrite(avg_output_path, avg_img)
        print(f"Saved averaged image to: {avg_output_path}")
    
if __name__ == "__main__":
    main()
