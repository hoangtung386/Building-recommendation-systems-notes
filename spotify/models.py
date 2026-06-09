"""Models for the spotify million playlist."""

from typing import Tuple

import torch
import torch.nn as nn


class SpotifyModel(nn.Module):
    """Spotify model that takes a context and predicts the next tracks."""

    def __init__(self, feature_size: int):
        super().__init__()
        self.max_albums = 100000
        self.album_embed = nn.Embedding(self.max_albums, feature_size)
        self.artist_embed = nn.Embedding(295861, feature_size)

    def get_embeddings(self, album: torch.Tensor, artist: torch.Tensor) -> torch.Tensor:
        album_modded = album % self.max_albums
        album_emb = self.album_embed(album_modded)
        artist_emb = self.artist_embed(artist)
        return torch.cat([album_emb, artist_emb], dim=-1)

    def forward(
        self,
        track_context: torch.Tensor,
        album_context: torch.Tensor,
        artist_context: torch.Tensor,
        next_track: torch.Tensor,
        next_album: torch.Tensor,
        next_artist: torch.Tensor,
        neg_track: torch.Tensor,
        neg_album: torch.Tensor,
        neg_artist: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        context_embed = self.get_embeddings(album_context, artist_context)
        next_embed = self.get_embeddings(next_album, next_artist)
        neg_embed = self.get_embeddings(neg_album, neg_artist)

        pos_affinity = torch.max(torch.matmul(next_embed, context_embed.T), dim=-1).values
        pos_affinity = pos_affinity + 0.1 * _isin(next_album, album_context)
        pos_affinity = pos_affinity + 0.1 * _isin(next_artist, artist_context)

        neg_affinity = torch.max(torch.matmul(neg_embed, context_embed.T), dim=-1).values
        neg_affinity = neg_affinity + 0.1 * _isin(neg_album, album_context)
        neg_affinity = neg_affinity + 0.1 * _isin(neg_artist, artist_context)

        all_embeddings = torch.cat([context_embed, next_embed, neg_embed], dim=-2)
        all_embeddings_l2 = torch.sqrt(torch.sum(all_embeddings ** 2, dim=-1))

        context_self_affinity = torch.matmul(torch.flip(context_embed, dims=[-2]), context_embed.T)
        next_self_affinity = torch.matmul(torch.flip(next_embed, dims=[-2]), next_embed.T)
        neg_self_affinity = torch.matmul(torch.flip(neg_embed, dims=[-2]), neg_embed.T)

        return (pos_affinity, neg_affinity,
                context_self_affinity, next_self_affinity, neg_self_affinity,
                all_embeddings_l2)


def _isin(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.unsqueeze(-1) == b.unsqueeze(0)).any(dim=-1).float()
