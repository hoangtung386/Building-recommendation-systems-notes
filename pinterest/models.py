"""Models for the shop the look content recommender."""

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """Simple CNN."""

    def __init__(self, filters: Sequence[int], output_size: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_channels = 3
        for f in filters:
            block = nn.ModuleDict({
                "residual_conv": nn.Conv2d(in_channels, f, kernel_size=3, stride=2, padding=1),
                "conv1": nn.Conv2d(in_channels, f, kernel_size=3, stride=2, padding=1),
                "bn1": nn.BatchNorm2d(f),
                "conv2": nn.Conv2d(f, f, kernel_size=1, stride=1),
                "bn2": nn.BatchNorm2d(f),
                "conv3": nn.Conv2d(f, f, kernel_size=1, stride=1),
                "bn3": nn.BatchNorm2d(f),
            })
            self.blocks.append(block)
            in_channels = f
        self.fc = nn.Linear(in_channels, output_size)

    def forward(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        for block in self.blocks:
            residual = block["residual_conv"](x)
            x = block["conv1"](x)
            x = block["bn1"](x) if train else block["bn1"](x)
            x = F.silu(x)
            x = block["conv2"](x)
            x = block["bn2"](x) if train else block["bn2"](x)
            x = F.silu(x)
            x = block["conv3"](x)
            x = block["bn3"](x) if train else block["bn3"](x)
            x = x + residual
            x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = x.mean(dim=(2, 3))
        x = self.fc(x)
        return x


class STLModel(nn.Module):
    """Shop the look model that takes in a scene and item and computes a score for them."""

    def __init__(self, output_size: int):
        super().__init__()
        default_filter = [16, 32, 64, 128]
        self.scene_cnn = CNN(filters=default_filter, output_size=output_size)
        self.product_cnn = CNN(filters=default_filter, output_size=output_size)

    def get_scene_embed(self, scene: torch.Tensor) -> torch.Tensor:
        self.scene_cnn.eval()
        return self.scene_cnn(scene, train=False)

    def get_product_embed(self, product: torch.Tensor) -> torch.Tensor:
        self.product_cnn.eval()
        return self.product_cnn(product, train=False)

    def forward(
        self,
        scene: torch.Tensor,
        pos_product: torch.Tensor,
        neg_product: torch.Tensor,
        train: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        scene_embed = self.scene_cnn(scene, train)

        pos_product_embed = self.product_cnn(pos_product, train)
        pos_score = (scene_embed * pos_product_embed).sum(dim=-1)

        neg_product_embed = self.product_cnn(neg_product, train)
        neg_score = (scene_embed * neg_product_embed).sum(dim=-1)

        return pos_score, neg_score, scene_embed, pos_product_embed, neg_product_embed
