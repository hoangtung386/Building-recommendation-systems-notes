#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  Trains a model for the shop the look data set.
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
import pin_util

FLAGS = flags.FLAGS
_INPUT_FILE = flags.DEFINE_string(
    "input_file",
    "STL-Dataset/fashion.json",
    "Input cat json file.")
_IMAGE_DIRECTORY = flags.DEFINE_string(
    "image_dir",
    "artifacts/shop_the_look:v1",
    "Directory containing downloaded images from the shop the look dataset.")
_NUM_NEG = flags.DEFINE_integer(
    "num_neg", 5, "How many negatives per positive."
)
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
_REGULARIZATION = flags.DEFINE_float("regularization", 0.1, "Regularization.")
_OUTPUT_SIZE = flags.DEFINE_integer("output_size", 32, "Size of output embedding.")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 16, "Batch size.")
_LOG_EVERY_STEPS = flags.DEFINE_integer("log_every_steps", 100, "Log every this step.")
_EVAL_EVERY_STEPS = flags.DEFINE_integer("eval_every_steps", 2000, "Eval every this step.")
_CHECKPOINT_EVERY_STEPS = flags.DEFINE_integer("checkpoint_every_steps", 100000, "Checkpoint every this step.")
_MAX_STEPS = flags.DEFINE_integer("max_steps", 30000, "Max number of steps.")
_WORKDIR = flags.DEFINE_string("work_dir", "/tmp", "Work directory.")
_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    "pinterest_stl_model", "Model name.")
_RESTORE_CHECKPOINT = flags.DEFINE_bool("restore_checkpoint", False, "If true, restore.")


def generate_triplets(
    scene_product: Sequence[Tuple[str, str]],
    num_neg: int) -> Tuple[list, list]:
    """Generate positive and negative triplets."""
    count = len(scene_product)
    train = []
    test = []
    rng = random.Random(0)
    for i in range(count):
        scene, pos = scene_product[i]
        is_test = i % 10 == 0
        neg_indices = [rng.randint(0, count - 2) for _ in range(num_neg)]
        for neg_idx in neg_indices:
            _, neg = scene_product[neg_idx]
            if is_test:
                test.append((scene, pos, neg))
            else:
                train.append((scene, pos, neg))
    return train, test


def compute_loss(result, regularization, batch_size):
    pos_score, neg_score, scene_embed, pos_embed, neg_embed = result
    triplet_loss = F.relu(1.0 + neg_score - pos_score).sum()

    def reg_fn(embed):
        return F.relu(torch.sqrt(torch.sum(embed ** 2, dim=-1)) - 1.0)

    reg_loss = reg_fn(scene_embed) + reg_fn(pos_embed) + reg_fn(neg_embed)
    reg_loss = reg_loss.sum()
    return (triplet_loss + regularization * reg_loss) / batch_size


def train_step(model, scene, pos_product, neg_product, optimizer, regularization, batch_size, device):
    model.train()
    scene = scene.to(device)
    pos_product = pos_product.to(device)
    neg_product = neg_product.to(device)

    optimizer.zero_grad()
    result = model(scene, pos_product, neg_product, train=True)
    loss = compute_loss(result, regularization, batch_size)
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_step(model, scene, pos_product, neg_product, device):
    model.eval()
    with torch.no_grad():
        scene = scene.to(device)
        pos_product = pos_product.to(device)
        neg_product = neg_product.to(device)
        result = model(scene, pos_product, neg_product, train=False)
        pos_score, neg_score = result[0], result[1]
        triplet_loss = F.relu(1.0 + neg_score - pos_score).sum()
    return triplet_loss.item()


def main(argv):
    """Main function."""
    del argv

    config = {
        "learning_rate": _LEARNING_RATE.value,
        "regularization": _REGULARIZATION.value,
        "output_size": _OUTPUT_SIZE.value
    }

    run = wandb.init(
        config=config,
        project="recsys-pinterest"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    logging.info("Image dir %s, input file %s", _IMAGE_DIRECTORY.value, _INPUT_FILE.value)

    scene_product = pin_util.get_valid_scene_product(_IMAGE_DIRECTORY.value, _INPUT_FILE.value)
    logging.info("Found %d valid scene product pairs." % len(scene_product))

    train, test = generate_triplets(scene_product, _NUM_NEG.value)
    num_train = len(train)
    num_test = len(test)
    logging.info("Train triplets %d", num_train)
    logging.info("Test triplets %d", num_test)

    rng = random.Random(0)
    rng.shuffle(train)
    rng.shuffle(test)

    batch_size = _BATCH_SIZE.value
    train_loader = input_pipeline.create_dataloader(train, batch_size, shuffle=True)
    test_loader = input_pipeline.create_dataloader(test, batch_size, shuffle=False)

    model = models.STLModel(output_size=wandb.config.output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    if _RESTORE_CHECKPOINT.value:
        checkpoint_path = os.path.join(_WORKDIR.value, "checkpoint_latest.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
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
    regularization = wandb.config.regularization
    eval_steps = int(num_test / batch_size)

    for i in range(init_step, _MAX_STEPS.value + 1):
        try:
            batch = next(train_it)
        except StopIteration:
            train_it = iter(train_loader)
            batch = next(train_it)

        scene, pos_product, neg_product = batch
        loss = train_step(model, scene, pos_product, neg_product, optimizer, regularization, batch_size, device)
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
            eval_loss = []
            for j in range(eval_steps):
                try:
                    ebatch = next(test_it)
                except StopIteration:
                    test_it = iter(test_loader)
                    ebatch = next(test_it)
                escene, epos_product, eneg_product = ebatch
                loss = eval_step(model, escene, epos_product, eneg_product, device)
                eval_loss.append(loss)
            eval_loss = np.mean(eval_loss) / batch_size
            metrics.update({"eval_loss": eval_loss})

        if i % _LOG_EVERY_STEPS.value == 0 and i > 0:
            mean_loss = np.mean(losses)
            losses = []
            metrics.update({"train_loss": mean_loss})
            wandb.log(metrics)
            logging.info(metrics)

    logging.info("Saving as %s", _MODEL_NAME.value)
    torch.save(model.state_dict(), _MODEL_NAME.value)
    metadata = {"output_size": wandb.config.output_size}
    artifact = wandb.Artifact(
        name=_MODEL_NAME.value,
        metadata=metadata,
        type="model")
    artifact.add_file(_MODEL_NAME.value)
    run.log_artifact(artifact)


if __name__ == "__main__":
    app.run(main)
