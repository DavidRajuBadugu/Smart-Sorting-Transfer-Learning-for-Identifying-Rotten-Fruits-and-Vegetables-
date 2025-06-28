import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# Set your dataset path
train_dir = 'dataset/train'
classes = os.listdir(train_dir)

# Display limit (for example, first 16 classes)
n = min(len(classes), 16)

plt.figure(figsize=(15, 15))

for i, cls in enumerate(classes[:n]):
    class_path = os.path.join(train_dir, cls)
    image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        continue

    img_path = os.path.join(class_path, random.choice(image_files))
    img = Image.open(img_path)

    plt.subplot(4, 4, i + 1)
    plt.imshow(img)
    plt.title(cls, fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.show()
