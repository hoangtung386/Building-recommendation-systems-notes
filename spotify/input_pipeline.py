import glob
import json
import os
from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SpotifyDataset(Dataset):
    def __init__(self, pattern: str):
        filenames = glob.glob(pattern)
        self.data = []
        for fname in filenames:
            with open(fname, "r") as f:
                for line in f:
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            "track_context": torch.tensor(row["track_context"], dtype=torch.long),
            "album_context": torch.tensor(row["album_context"], dtype=torch.long),
            "artist_context": torch.tensor(row["artist_context"], dtype=torch.long),
            "next_track": torch.tensor(row["next_track"], dtype=torch.long),
            "next_album": torch.tensor(row["next_album"], dtype=torch.long),
            "next_artist": torch.tensor(row["next_artist"], dtype=torch.long),
        }


def create_dataloader(pattern: str, batch_size: int = 1, shuffle: bool = False, num_workers: int = 4) -> DataLoader:
    dataset = SpotifyDataset(pattern)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def load_dict(dictionary_path: str, name: str):
    filename = os.path.join(dictionary_path, name)
    with open(filename, "r") as f:
        return json.load(f)


def load_all_tracks(all_tracks_file: str, track_uri_dict, album_uri_dict, artist_uri_dict):
    with open(all_tracks_file, "r") as f:
        all_tracks_json = json.load(f)
    all_tracks_dict = {int(k): v for k, v in all_tracks_json.items()}
    all_tracks_features = {
        k: (track_uri_dict[v["track_uri"]], album_uri_dict[v["album_uri"]], artist_uri_dict[v["artist_uri"]])
        for k, v in all_tracks_dict.items()
    }
    return all_tracks_dict, all_tracks_features


def make_all_tracks_numpy(all_tracks_features):
    all_tracks = []
    all_albums = []
    all_artists = []
    items = sorted(all_tracks_features.items())
    for row in items:
        k, v = row
        all_tracks.append(v[0])
        all_albums.append(v[1])
        all_artists.append(v[2])
    all_tracks = torch.tensor(all_tracks, dtype=torch.long)
    all_albums = torch.tensor(all_albums, dtype=torch.long)
    all_artists = torch.tensor(all_artists, dtype=torch.long)
    return all_tracks, all_albums, all_artists
