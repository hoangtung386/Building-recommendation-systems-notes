#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  Trains a model for the Spotify million playlist data set.
"""

import json
import os
import random
from typing import Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
import torch.nn.functional as F
import wandb

import input_pipeline
import models

FLAGS = flags.FLAGS
_TRAIN_PATTERN = flags.DEFINE_string(
    "train_pattern",
    "data/training/00??[0-8].jsonl",
    "Training pattern.")
_TEST_PATTERN = flags.DEFINE_string(
    "test_pattern",
    "data/training/00??9.jsonl",
    "Training pattern.")
_ALL_TRACKS = flags.DEFINE_string(
    "all_tracks",
    "data/training/all_tracks.json",
    "Location of track database.")
_DICTIONARY_PATH = flags.DEFINE_string("dictionaries", "data/dictionaries", "Dictionary path.")

_NUM_NEGATIVES = flags.DEFINE_integer("num_negatives", 64, "Number of negatives to sample.")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
_MOMENTUM = flags.DEFINE_float("momentum", 0.98, "Momentum.")
_REGULARIZATION = flags.DEFINE_float("regularization", 10.0, "Regularization (max l2 norm squared).")
_FEATURE_SIZE = flags.DEFINE_integer("feature_size", 32, "Size of output embedding.")
_LOG_EVERY_STEPS = flags.DEFINE_integer("log_every_steps", 1000, "Log every this step.")
_EVAL_EVERY_STEPS = flags.DEFINE_integer("eval_every_steps", 10000, "Eval every this step.")
_EVAL_STEPS = flags.DEFINE_integer("eval_steps", 1000, "Eval this number of entries.")
_CHECKPOINT_EVERY_STEPS = flags.DEFINE_integer("checkpoint_every_steps", 100000, "Checkpoint every this step.")
_MAX_STEPS = flags.DEFINE_integer("max_steps", 2000000, "Max number of steps.")
_WORKDIR = flags.DEFINE_string("work_dir", "/tmp", "Work directory.")
_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    "spotify_mpl.model", "Model name.")
_RESTORE_CHECKPOINT = flags.DEFINE_bool("restore_checkpoint", False, "If true, restore.")


def compute_loss(result, regularization):
    pos_affinity = result[0]
    neg_affinity = result[1]
    context_self_affinity = result[2]
    next_self_affinity = result[3]
    neg_self_affinity = result[4]
    all_embeddings_l2 = result[5]

    mean_neg_affinity = pos_affinity.mean()
    mean_pos_affinity = pos_affinity.mean()
    mean_triplet_loss = F.relu(1.0 + mean_neg_affinity - mean_pos_affinity)

    max_neg_affinity = neg_affinity.max()
    min_pos_affinity = pos_affinity.min()
    extremal_triplet_loss = F.relu(1.0 + max_neg_affinity - min_pos_affinity)

    context_self_affinity_loss = F.relu(0.5 - context_self_affinity).mean()
    next_self_affinity_loss = F.relu(0.5 - next_self_affinity).mean()
    neg_self_affinity_loss = F.relu(neg_self_affinity).mean()

    reg_loss = F.relu(all_embeddings_l2 - regularization).sum()
    loss = (extremal_triplet_loss + mean_triplet_loss + reg_loss +
            context_self_affinity_loss + next_self_affinity_loss + neg_self_affinity_loss)
    return loss


def train_step(model, x, optimizer, regularization, device):
    model.train()
    x = {k: v.to(device) for k, v in x.items()}
    optimizer.zero_grad()
    result = model(
        x["track_context"], x["album_context"], x["artist_context"],
        x["next_track"], x["next_album"], x["next_artist"],
        x["neg_track"], x["neg_album"], x["neg_artist"])
    loss = compute_loss(result, regularization)
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_step(model, y, all_tracks, all_albums, all_artists, device):
    model.eval()
    with torch.no_grad():
        y = {k: v.to(device) for k, v in y.items()}
        all_tracks = all_tracks.to(device)
        all_albums = all_albums.to(device)
        all_artists = all_artists.to(device)
        result = model(
            y["track_context"], y["album_context"], y["artist_context"],
            y["next_track"], y["next_album"], y["next_artist"],
            all_tracks, all_albums, all_artists)
        all_affinity = result[1]
        top_k_scores, top_k_indices = torch.topk(all_affinity, 500)
        top_tracks = all_tracks[top_k_indices]
        top_artists = all_artists[top_k_indices]
        top_tracks_count = _isin_count(top_tracks, y["next_track"])
        top_artists_count = _isin_count(top_artists, y["next_artist"])

        top_tracks_recall = top_tracks_count / y["next_track"].shape[0]
        top_artists_recall = top_artists_count / y["next_artist"].shape[0]

    return torch.tensor([top_tracks_recall, top_artists_recall])


def _isin_count(a, b):
    return (a.unsqueeze(-1) == b.unsqueeze(0)).any(dim=-1).sum().float()


def sample_negative(x, num_negatives, all_tracks, all_albums, all_artists):
    idx = torch.randint(0, all_tracks.shape[0], (num_negatives,))
    x["neg_track"] = all_tracks[idx]
    x["neg_album"] = all_albums[idx]
    x["neg_artist"] = all_artists[idx]


def main(argv):
    """Main function."""
    del argv

    track_uri_dict = input_pipeline.load_dict(_DICTIONARY_PATH.value, "track_uri_dict.json")
    num_tracks = len(track_uri_dict)
    print("%d tracks loaded" % num_tracks)
    album_uri_dict = input_pipeline.load_dict(_DICTIONARY_PATH.value, "album_uri_dict.json")
    num_albums = len(album_uri_dict)
    print("%d albums loaded" % num_albums)
    artist_uri_dict = input_pipeline.load_dict(_DICTIONARY_PATH.value, "artist_uri_dict.json")
    num_artists = len(artist_uri_dict)
    print("%d artists loaded" % num_artists)
    all_tracks_dict, all_tracks_features = input_pipeline.load_all_tracks(
        _ALL_TRACKS.value, track_uri_dict, album_uri_dict, artist_uri_dict)
    print("10 sample tracks")
    for i in range(10):
        print("Track %d" % i)
        print(all_tracks_dict[i])
        print(all_tracks_features[i])

    all_tracks, all_albums, all_artists = input_pipeline.make_all_tracks_numpy(all_tracks_features)
    print("All tracks features top 10")
    print(all_tracks[:10])
    print(all_albums[:10])
    print(all_artists[:10])

    config = {
        "learning_rate": _LEARNING_RATE.value,
        "regularization": _REGULARIZATION.value,
        "feature_size": _FEATURE_SIZE.value,
        "momentum": _MOMENTUM.value,
    }

    run = wandb.init(
        config=config,
        project="recsys-spotify"
    )
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = input_pipeline.create_dataloader(_TRAIN_PATTERN.value, shuffle=True)
    test_loader = input_pipeline.create_dataloader(_TEST_PATTERN.value, shuffle=False)

    model = models.SpotifyModel(feature_size=config["feature_size"]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"])

    num_negatives = _NUM_NEGATIVES.value

    if _RESTORE_CHECKPOINT.value:
        checkpoint_path = os.path.join(_WORKDIR.value, "checkpoint_latest.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            init_step = checkpoint["step"]
            logging.info("Restored from checkpoint at step %d", init_step)
        else:
            init_step = 0
    else:
        init_step = 0

    train_it = iter(train_loader)
    test_it = iter(test_loader)

    losses = []
    logging.info("Starting at step %d", init_step)
    regularization = config["regularization"]
    eval_steps = _EVAL_STEPS.value

    for i in range(init_step, _MAX_STEPS.value + 1):
        try:
            x = next(train_it)
        except StopIteration:
            train_it = iter(train_loader)
            x = next(train_it)

        sample_negative(x, num_negatives, all_tracks, all_albums, all_artists)
        loss = train_step(model, x, optimizer, regularization, device)
        losses.append(loss)

        if i % _CHECKPOINT_EVERY_STEPS.value == 0 and i > 0:
            logging.info("Saving checkpoint")
            checkpoint_path = os.path.join(_WORKDIR.value, f"checkpoint_{i:06d}.pt")
            torch.save({
                "step": i,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, checkpoint_path)

        metrics = {"step": i}

        if i % _EVAL_EVERY_STEPS.value == 0 and i > 0:
            sum_metrics = torch.zeros(2)
            for j in range(eval_steps):
                try:
                    y = next(test_it)
                except StopIteration:
                    test_it = iter(test_loader)
                    y = next(test_it)
                eval_metrics = eval_step(model, y, all_tracks, all_albums, all_artists, device)
                sum_metrics = sum_metrics + eval_metrics
            avg_metrics = sum_metrics / eval_steps
            metrics.update({
                "eval_track_recall": avg_metrics[0].item(),
                "eval_artist_recall": avg_metrics[1].item(),
            })
            logging.info(metrics)

        if i % _LOG_EVERY_STEPS.value == 0 and i > 0:
            mean_loss = np.mean(losses)
            losses = []
            metrics.update({"train_loss": mean_loss})
            logging.info(metrics)
            wandb.log(metrics)

    logging.info("Saving as %s", _MODEL_NAME.value)
    torch.save(model.state_dict(), _MODEL_NAME.value)
    metadata = dict(config)
    artifact = wandb.Artifact(
        name=_MODEL_NAME.value,
        metadata=metadata,
        type="model")
    artifact.add_file(_MODEL_NAME.value)
    run.log_artifact(artifact)


if __name__ == "__main__":
    app.run(main)
