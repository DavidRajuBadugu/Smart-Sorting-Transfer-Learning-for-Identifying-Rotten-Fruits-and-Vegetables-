import os
import shutil
from sklearn.model_selection import train_test_split

# Set the path to the dataset
dataset_dir = 'Fruit And Vegetable Diseases Dataset'
output_dir = 'dataset'

# Get the class names (folder names)
classes = os.listdir(dataset_dir)

# Create train, val, and test directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

# Create class subfolders inside each split folder
for cls in classes:
    os.makedirs(os.path.join(output_dir, 'train', cls), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', cls), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', cls), exist_ok=True)

    # Path to the class images
    class_dir = os.path.join(dataset_dir, cls)
    images = os.listdir(class_dir)

    print(f"{cls}: {len(images)} images")

    # Split into 80% train+val and 20% test
    train_and_val_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

    # Split train_and_val into 75% train and 25% val (=> 60% train, 20% val)
    train_images, val_images = train_test_split(train_and_val_images, test_size=0.25, random_state=42)

    # Copy images to the respective directories
    for img in train_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, 'train', cls, img))

    for img in val_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, 'val', cls, img))

    for img in test_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, 'test', cls, img))

print("✅ Dataset successfully split into train/val/test.")
