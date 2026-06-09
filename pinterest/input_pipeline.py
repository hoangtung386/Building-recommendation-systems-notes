from typing import Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


def get_transform():
    return transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]),
    ])


def process_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    transform = get_transform()
    return transform(img)


class TripletDataset(Dataset):
    def __init__(self, triplets: Sequence[Tuple[str, str, str]]):
        self.triplets = triplets
        self.transform = get_transform()

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        scene_path, pos_path, neg_path = self.triplets[idx]
        scene = Image.open(scene_path).convert("RGB")
        pos = Image.open(pos_path).convert("RGB")
        neg = Image.open(neg_path).convert("RGB")
        return self.transform(scene), self.transform(pos), self.transform(neg)


class ImageWithIdDataset(Dataset):
    def __init__(self, paths: Sequence[str]):
        self.paths = paths
        self.transform = get_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return path, self.transform(img)


def create_dataloader(
    triplets: Sequence[Tuple[str, str, str]],
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    dataset = TripletDataset(triplets)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
