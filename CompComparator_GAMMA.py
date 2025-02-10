import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, simpledialog

def is_black_image(image, threshold=10):
    """Return True if the image is nearly all black."""
    return np.mean(image) < threshold

def preprocess_image(image_path, scale_value, target_width=1096, target_height=949, crop_border=(50, 50, 50, 50)):
    """
    Load an image in grayscale, rescale it using the provided scale factor,
    crop fixed borders, and resize to target dimensions.
    Returns None if the image is unreadable or nearly black.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    # Rescale using the provided scale factor.
    image = cv2.resize(image, None, fx=scale_value, fy=scale_value, interpolation=cv2.INTER_AREA)

    left, right, top, bottom = crop_border
    h, w = image.shape
    if h <= (top + bottom) or w <= (left + right):
        print(f"Image too small to crop: {image_path}")
        return None
    cropped = image[top:h - bottom, left:w - right]

    # Resize to target dimensions.
    processed = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)

    if is_black_image(processed):
        return None

    return processed

def process_folder(input_folder, output_folder, scale_value):
    """
    Process all BMP images in the input folder:
      - Preprocess each image.
      - Save the processed image to the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.bmp'):
            path = os.path.join(input_folder, filename)
            processed = preprocess_image(path, scale_value)
            if processed is not None:
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, processed)
                print(f"Processed: {filename}")
            else:
                print(f"Skipped: {filename}")

def main():
    # Set up Tkinter dialogs.
    root = tk.Tk()
    root.withdraw()

    input_folder = filedialog.askdirectory(title="Select Folder with Bad Part Images")
    if not input_folder:
        print("No folder selected. Exiting.")
        return

    scale_value = simpledialog.askfloat("Scale Factor", "Enter scale factor (e.g., 1.0):", parent=root, minvalue=0.1)
    if scale_value is None:
        scale_value = 1.0
        print("No scale factor entered; using default 1.0")

    output_folder = filedialog.askdirectory(title="Select Output Folder for Processed Images")
    if not output_folder:
        print("No output folder selected. Exiting.")
        return

    process_folder(input_folder, output_folder, scale_value)
    print("Processing complete.")

if __name__ == "__main__":
    main()
