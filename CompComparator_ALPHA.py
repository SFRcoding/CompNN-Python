import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog

# ------------------------------------------------------------------
# Custom Layer Definition to Replace the Lambda Layer
# ------------------------------------------------------------------
class MyAbsDiff(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        # Compute element-wise absolute difference between the two inputs.
        a, b = inputs
        return tf.abs(a - b)
    
    def compute_output_shape(self, input_shape):
        # The output shape is the same as the shape of the first input.
        return input_shape[0]
    
    def get_config(self):
        # For proper serialization.
        base_config = super(MyAbsDiff, self).get_config()
        return base_config

# ------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------
def preprocess_for_model(image, target_size=(128, 128)):
    """
    Resize a grayscale image to target_size, convert it to float32,
    normalize to [0, 1], and add a channel dimension.
    """
    try:
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    except Exception as e:
        print("Error resizing image:", e)
        return None
    processed = resized.astype("float32") / 255.0
    processed = np.expand_dims(processed, axis=-1)
    return processed

def highlight_differences(baseline_img, candidate_img, diff_thresh=30):
    """
    Compute the absolute difference between two grayscale images,
    threshold it to create a binary image, and return the bounding box
    (x, y, w, h) of the largest difference region, along with the difference image.
    """
    diff = cv2.absdiff(baseline_img, candidate_img)
    ret, thresh = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h), diff
    else:
        return None, diff

# ------------------------------------------------------------------
# Main Comparison Routine
# ------------------------------------------------------------------
def main():
    print("Starting comparison script...")

    model_path = "siamese_model.h5"
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found.")
        sys.exit(1)

    # Load the trained Siamese model.
    # The original model used a Lambda layer (with an anonymous function) that computed tf.abs(a - b).
    # We replace that Lambda with our custom layer by mapping the saved lambda key ("<lambda>")
    # to our custom class MyAbsDiff.
    try:
        siamese_model = tf.keras.models.load_model(
            model_path,
            custom_objects={"<lambda>": MyAbsDiff}
        )
        print("Siamese model loaded successfully from '{}'.".format(model_path))
    except Exception as e:
        print("Error loading the model:", e)
        sys.exit(1)

    # The model was trained on images of size 128x128.
    target_size = (128, 128)

    # Set up Tkinter for file dialogs.
    root = tk.Tk()
    root.withdraw()

    # Ask the user to select a baseline (good-part) image.
    print("Please select a baseline (good-part) image.")
    baseline_path = filedialog.askopenfilename(
        title="Select Baseline Good Part Image",
        filetypes=[("Image Files", "*.png;*.bmp")]
    )
    if not baseline_path:
        print("No baseline image selected. Exiting.")
        sys.exit(1)
    baseline_orig = cv2.imread(baseline_path, cv2.IMREAD_GRAYSCALE)
    if baseline_orig is None:
        print(f"Failed to load baseline image from '{baseline_path}'.")
        sys.exit(1)
    baseline_proc = preprocess_for_model(baseline_orig, target_size)
    if baseline_proc is None:
        print("Failed to preprocess baseline image.")
        sys.exit(1)
    baseline_input = np.expand_dims(baseline_proc, axis=0)
    print("Baseline image loaded and processed.")

    # Ask the user to select the folder containing candidate (bad-part) images.
    print("Please select the folder containing candidate (bad-part) images.")
    candidate_folder = filedialog.askdirectory(title="Select Folder with Candidate (Bad Part) Images")
    if not candidate_folder:
        print("No candidate folder selected. Exiting.")
        sys.exit(1)

    candidate_files = [f for f in os.listdir(candidate_folder) if f.lower().endswith(".bmp")]
    if not candidate_files:
        print(f"No BMP images found in the candidate folder: '{candidate_folder}'.")
        sys.exit(1)
    print("Found {} candidate BMP images.".format(len(candidate_files)))

    # Process each candidate image.
    for filename in candidate_files:
        candidate_path = os.path.join(candidate_folder, filename)
        candidate_orig = cv2.imread(candidate_path, cv2.IMREAD_GRAYSCALE)
        if candidate_orig is None:
            print(f"Failed to load candidate image '{filename}'.")
            continue

        candidate_proc = preprocess_for_model(candidate_orig, target_size)
        if candidate_proc is None:
            print(f"Failed to preprocess candidate image '{filename}'.")
            continue

        candidate_input = np.expand_dims(candidate_proc, axis=0)

        # Compute similarity score using the Siamese model.
        try:
            similarity = siamese_model.predict([baseline_input, candidate_input])[0][0]
        except Exception as e:
            print(f"Error during model prediction for '{filename}': {e}")
            continue
        print("File: '{}', Similarity Score: {:.2f}".format(filename, similarity))

        # For visualization, convert the processed images back to 0-255 grayscale.
        baseline_resized = (baseline_proc * 255).astype("uint8")
        candidate_resized = (candidate_proc * 255).astype("uint8")

        # Compute the difference and determine the largest difference area.
        bbox, diff_map = highlight_differences(baseline_resized, candidate_resized)
        candidate_display = cv2.cvtColor(candidate_resized, cv2.COLOR_GRAY2BGR)
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(candidate_display, (x, y), (x + w, y + h), (0, 0, 255), 2)
            print("  Largest difference area: x={}, y={}, w={}, h={}".format(x, y, w, h))
        else:
            print("  No significant difference area found for '{}'.".format(filename))

        # Display the candidate image with highlighted differences and the difference map.
        cv2.imshow("Candidate: " + filename, candidate_display)
        cv2.imshow("Difference Map: " + filename, diff_map)
        print("Press any key to continue to the next image (Esc to exit)...")
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == 27:  # Esc key pressed
            print("Esc key pressed. Exiting early.")
            break

    print("Comparison complete.")

if __name__ == "__main__":
    main()
