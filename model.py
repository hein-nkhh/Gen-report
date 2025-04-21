# model.py
import torch
import torch.nn as nn
from timm import create_model
from transformers import AutoModelForCausalLM, AutoTokenizer

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.projection(x)

class VisionEncoder(nn.Module):
    def __init__(self, model_name='swin_base_patch4_window7_224', output_dim=1024):
        super().__init__()
        self.encoder = create_model(model_name, pretrained=True, num_classes=0, features_only=False)
        self.projection = ProjectionHead(self.encoder.num_features, output_dim)

    def forward(self, images):
        feats = self.encoder.forward_features(images)  # (B, H, W, C)
        B, H, W, C = feats.shape
        feats = feats.view(B, H*W, C)
        return self.projection(feats)

class CrossAttention(nn.Module):
    def __init__(self, hidden_dim=1024, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                        dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_embeds, vision_embeds):
        attended, _ = self.attn(query=text_embeds, key=vision_embeds, value=vision_embeds)
        return self.dropout(attended + text_embeds)

class TextDecoder(nn.Module):
    def __init__(self, model_name='microsoft/biogpt'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.decoder = AutoModelForCausalLM.from_pretrained(model_name)
        self.embedding_dim = self.decoder.get_input_embeddings().weight.shape[1]

    def encode_text(self, texts, max_length=153):
        encoding = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return encoding['input_ids'], encoding['attention_mask']

    def get_input_embeddings(self, input_ids):
        return self.decoder.get_input_embeddings()(input_ids)

    def forward(self, input_ids, attention_mask, vision_embeds=None, labels=None):
        if vision_embeds is not None:
            inputs_embeds = self.get_input_embeddings(input_ids)
            vision_token = vision_embeds.mean(dim=1, keepdim=True)
            inputs_embeds = torch.cat([vision_token, inputs_embeds[:,1:,:]], dim=1)
            return self.decoder(inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                labels=labels)
        return self.decoder(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)

    def generate(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                 max_length=50, **kwargs):
        # Ensure inputs_embeds is not empty
        if inputs_embeds is None and input_ids is None:
            raise ValueError("Both inputs_embeds and input_ids cannot be None")

        # Check the sizes of inputs
        if inputs_embeds is not None:
            if inputs_embeds.size(1) == 0:
                raise ValueError("inputs_embeds cannot be empty")

        if input_ids is not None:
            if input_ids.size(1) == 0:
                raise ValueError("input_ids cannot be empty")

        # Generate output
        with torch.no_grad():
            output_ids = self.decoder.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                max_length=max_length,
                **kwargs
            )
            return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

class VisionTextModel(nn.Module):
    def __init__(self, vision_encoder, text_decoder, cross_attention):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.cross_attention = cross_attention

    def forward(self, images, reports, labels=None):
        device = images.device
        vision_embeds = self.vision_encoder(images)
        input_ids, attention_mask = self.text_decoder.encode_text(reports)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        text_embeds = self.text_decoder.get_input_embeddings(input_ids)
        fused = self.cross_attention(text_embeds, vision_embeds)
        return self.text_decoder.decoder(inputs_embeds=fused,
                                        attention_mask=attention_mask,
                                        labels=labels)

    def generate(self, images, input_texts, max_length=50, **kwargs):
        device = images.device
        vision_embeds = self.vision_encoder(images)

        # Ensure that vision_embeds is not empty
        if vision_embeds.size(1) == 0:
            raise ValueError("vision_embeds cannot be empty")

        input_ids, attention_mask = self.text_decoder.encode_text(input_texts)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Ensure that input_ids are valid
        if input_ids.size(1) == 0:
            raise ValueError("input_ids cannot be empty")

        text_embeds = self.text_decoder.get_input_embeddings(input_ids)
        fused = self.cross_attention(text_embeds, vision_embeds)

        # Ensure that fused embeddings are valid
        if fused.size(1) == 0:
            raise ValueError("Fused embeddings cannot be empty")

        return self.text_decoder.generate(inputs_embeds=fused,
                                        attention_mask=attention_mask,
                                        max_length=max_length,
                                        **kwargs)