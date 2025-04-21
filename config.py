import torch
import os

class Config:
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 5e-5
        self.epochs = 10
        self.max_len = 153
        self.image_size = (224, 224)
        self.train_csv = '/kaggle/input/data-split-csv/Train_Data.csv'
        self.cv_csv = '/kaggle/input/data-split-csv/CV_Data.csv'
        self.test_csv = '/kaggle/input/data-split-csv/Test_Data.csv'
        self.image_dir = '/kaggle/input/image-features-attention/xray_images'
        self.save_model_path = './model_checkpoint.pth'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
