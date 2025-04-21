# dataset.py
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class XrayReportDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, max_length=153):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image1_path = os.path.join(self.image_dir, row['Image1'])
        image2_path = os.path.join(self.image_dir, row['Image2'])

        image1 = Image.open(image1_path).convert('RGB')
        image2 = Image.open(image2_path).convert('RGB')

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        image = torch.cat([image1, image2], dim=0)  # (6, H, W)
        report = row['Report']
        return image, report

    @staticmethod
    def get_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])