import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import numpy as np

class PneumoniaDataset(Dataset):
    def __init__(self, df, img_dir="../png_images", augment=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment
        
        self.transform = A.Compose([
            A.Resize(512, 512),  # ResNet50 friendly size
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
            A.ShiftScaleRotate(p=0.5, rotate_limit=10, shift_limit=0.05, scale_limit=0.1),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]) if augment else A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.img_dir}/{row['patientId']}.png"
        image = np.array(Image.open(img_path).convert("RGB"))
        image = self.transform(image=image)["image"]
        
        label = torch.tensor(row['Target'], dtype=torch.float32)
        return image, label