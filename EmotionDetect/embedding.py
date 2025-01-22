import os
import numpy as np
from deepface import DeepFace
from tqdm import tqdm

# Dataset path
dataset_path = "dataset/train"  # Change to your dataset path

# Initialize variables
embeddings = []
labels = []
class_mapping = {}

# Loop over classes (e.g., happy, sad, etc.)
for class_idx, class_name in enumerate(os.listdir(dataset_path)):
    class_mapping[class_idx] = class_name  # Map index to class name
    class_dir = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_dir):
        for img_name in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}"):
            img_path = os.path.join(class_dir, img_name)

            try:
                # Extract embeddings using DeepFace
                embedding = DeepFace.represent(img_path=img_path, model_name="VGG-Face", enforce_detection=False)[0][
                    "embedding"]

                # Append the extracted embedding and label
                embeddings.append(embedding)
                labels.append(class_idx)
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

# Convert embeddings and labels to NumPy arrays
embeddings = np.array(embeddings)
labels = np.array(labels)

# Optionally, save embeddings and labels to disk for later use
np.save("embeddings.npy", embeddings)
np.save("labels.npy", labels)
np.save("class_mapping.npy", class_mapping)
print("Embeddings and labels saved.")
