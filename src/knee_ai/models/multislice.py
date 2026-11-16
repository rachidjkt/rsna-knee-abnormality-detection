from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18

from knee_ai.constants import TARGETS


class MultiSliceClassifier(nn.Module):
    def __init__(self, encoder: str = "resnet18", dropout: float = .2,
                 encoder_weights: str | Path | None = None, freeze_encoder: bool = False):
        super().__init__()
        self.encoder_name = encoder
        self.frozen = freeze_encoder
        if encoder == "resnet18":
            backbone = resnet18(weights=None)  # NEVER download weights as a constructor side effect.
            if encoder_weights:
                state = torch.load(encoder_weights, map_location="cpu", weights_only=True)
                backbone.load_state_dict(state, strict=True)
            backbone.fc = nn.Identity()
            self.encoder, features = backbone, 512
        elif encoder == "tiny":
            if encoder_weights:
                raise ValueError("Tiny encoder is for synthetic fixtures and has no pretrained weights")
            self.encoder = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.ReLU(),
                                         nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
                                         nn.AdaptiveAvgPool2d(1), nn.Flatten())
            features = 16
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, len(TARGETS)))
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])
        if freeze_encoder:
            self.encoder.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, images: torch.Tensor, mask: torch.Tensor,
                encode_chunk_size: int | None = None) -> torch.Tensor:
        if images.ndim != 5 or images.shape[2] != 1 or mask.shape != images.shape[:2]:
            raise ValueError("Expected images [B,S,1,H,W] and mask [B,S]")
        if not mask.bool().any(dim=1).all():
            raise ValueError("Each study needs at least one valid slice")
        batch, slices = images.shape[:2]
        flat_mask = mask.reshape(-1).bool()
        valid = images.reshape(-1, *images.shape[2:])[flat_mask].expand(-1, 3, -1, -1)
        valid = (valid - self.mean) / self.std
        if encode_chunk_size and not self.training:
            encoded = torch.cat([self.encoder(chunk) for chunk in valid.split(encode_chunk_size)])
        else:
            encoded = self.encoder(valid)
        study_index = torch.arange(batch, device=images.device).repeat_interleave(slices)[flat_mask]
        aggregated = encoded.new_zeros(batch, encoded.shape[1]).index_add(0, study_index, encoded)
        aggregated = aggregated / mask.sum(dim=1, keepdim=True).to(encoded.dtype)
        return self.head(aggregated)
