# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from model import VisionTextModel, VisionEncoder, CrossAttention, TextDecoder
from dataset import XrayReportDataset
from utils import collate_fn
from config import Config
from evaluation import evaluate_model

def adjust_encoder_in_channels(vision_encoder, in_chans=6):
    orig_proj = vision_encoder.encoder.patch_embed.proj
    new_proj = nn.Conv2d(
        in_channels=in_chans,
        out_channels=orig_proj.out_channels,
        kernel_size=orig_proj.kernel_size,
        stride=orig_proj.stride,
        padding=orig_proj.padding,
        bias=(orig_proj.bias is not None)
    )
    with torch.no_grad():
        new_proj.weight[:, :3, :, :].copy_(orig_proj.weight)
        new_proj.weight[:, 3:, :, :].copy_(orig_proj.weight)
        if orig_proj.bias is not None:
            new_proj.bias.copy_(orig_proj.bias)
    vision_encoder.encoder.patch_embed.proj = new_proj

def train():
    os.makedirs(Config.output_dir, exist_ok=True)
    device = torch.device(Config.device)

    transform = XrayReportDataset.get_transform()
    train_ds = XrayReportDataset(Config.train_csv, Config.image_dir, transform=transform, max_length=Config.max_len)
    cv_ds = XrayReportDataset(Config.cv_csv, Config.image_dir, transform=transform, max_length=Config.max_len)
    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, collate_fn=collate_fn)
    cv_loader = DataLoader(cv_ds, batch_size=Config.batch_size, shuffle=False, collate_fn=collate_fn)

    vision_encoder = VisionEncoder(model_name=Config.vision_encoder_name, output_dim=Config.vision_output_dim).to(device)
    adjust_encoder_in_channels(vision_encoder, in_chans=6)
    cross_attention = CrossAttention(hidden_dim=Config.cross_attn_dim, num_heads=Config.cross_attn_heads).to(device)
    text_decoder = TextDecoder(model_name=Config.text_decoder_model).to(device)
    model = VisionTextModel(vision_encoder, text_decoder, cross_attention).to(device)

    optimizer = AdamW(model.parameters(), lr=Config.lr)
    total_steps = len(train_loader) * Config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=Config.warmup_steps, num_training_steps=total_steps)

    best_loss = float('inf')
    for epoch in range(1, Config.epochs+1):
        model.train()
        total_train_loss = 0
        for images, reports in train_loader:
            images = images.to(device)
            input_ids, attention_mask = model.text_decoder.encode_text(reports, max_length=Config.max_len)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(images, reports, labels=input_ids)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, reports in cv_loader:
                images = images.to(device)
                input_ids, attention_mask = model.text_decoder.encode_text(reports, max_length=Config.max_len)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                outputs = model(images, reports, labels=input_ids)
                total_val_loss += outputs.loss.item()
        avg_val_loss = total_val_loss / len(cv_loader)

        print(f"Epoch {epoch}/{Config.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        ckpt_path = os.path.join(Config.output_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), Config.best_model_path)

    # Evaluate on test set
    print("Starting evaluation on test set...")
    test_ds = XrayReportDataset(Config.test_csv, Config.image_dir, transform=transform, max_length=Config.max_len)
    test_loader = DataLoader(test_ds, batch_size=Config.batch_size, shuffle=False, collate_fn=collate_fn)
    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    train()