import os
import pydicom
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np

DATA_DIR = "../data"
PNG_DIR = "../png_images"
os.makedirs(PNG_DIR, exist_ok=True)

labels = pd.read_csv(f"{DATA_DIR}/stage_1_train_labels.csv")
print(f"Total rows in labels CSV: {len(labels)}")
print(f"Unique patientIds: {labels['patientId'].nunique()}")

# Path to actual DICOM files
image_folder = f"{DATA_DIR}/stage_1_train_images/stage_2_train_images"
print(f"Using image folder: {image_folder}")

if not os.path.exists(image_folder):
    print("❌ Image folder not found. Check your dir output again.")
    exit()

# Get all .dcm files
dcm_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.dcm')]
print(f"Found {len(dcm_files)} DICOM files.")

def save_as_png(dcm_path, png_name):
    try:
        ds = pydicom.dcmread(dcm_path)
        img = ds.pixel_array.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        png_path = f"{PNG_DIR}/{png_name}.png"
        pil_img.save(png_path)
        return True
    except:
        return False

print("Starting conversion...")
converted = 0

# Convert all DICOM files - use filename without .dcm as PNG name (handles multiple images per patient)
for dcm_file in tqdm(dcm_files, desc="Converting DICOM to PNG"):
    base_name = os.path.splitext(dcm_file)[0]   # e.g. 0004cfab-14fd-4e49-80ba-63a80b6bddd6
    dcm_path = f"{image_folder}/{dcm_file}"
    if save_as_png(dcm_path, base_name):
        converted += 1

print(f"\n✅ Conversion finished!")
print(f"Successfully converted {converted} images to PNG.")
print(f"Total PNG files now: {len([f for f in os.listdir(PNG_DIR) if f.endswith('.png')])}")
print(f"PNG folder location: {os.path.abspath(PNG_DIR)}")