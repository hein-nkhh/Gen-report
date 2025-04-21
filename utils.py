# utils.py
import torch

def collate_fn(batch):
    images, reports = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(reports)

