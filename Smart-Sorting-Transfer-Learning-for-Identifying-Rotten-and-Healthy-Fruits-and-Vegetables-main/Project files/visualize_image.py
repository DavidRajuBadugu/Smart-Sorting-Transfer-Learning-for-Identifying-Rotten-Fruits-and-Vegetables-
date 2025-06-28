import os
import random
from PIL import Image as pil_Image

# Set the path to your training dataset directory
train_dir = 'dataset/train'  # Make sure this matches your structure

# List all class folders inside train directory
classes = os.listdir(train_dir)

# Loop through each class
for cls in classes:
    class_path = os.path.join(train_dir, cls)

    # Get all image files in the class folder
    image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"No images found in {cls}, skipping...")
        continue

    # Randomly choose one image from the class
    random_image = random.choice(image_files)
    image_path = os.path.join(class_path, random_image)

    # Load and show the image
    print(f"Showing sample from class: {cls} → {random_image}")
    img = pil_Image.open(image_path)
    img.show()
