#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  Generates embedding files given a model and a catalog.
"""

import json
import os
from typing import Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

import input_pipeline
import models
import pin_util

FLAGS = flags.FLAGS
_INPUT_FILE = flags.DEFINE_string("input_file", None, "Input cat json file.")
_IMAGE_DIRECTORY = flags.DEFINE_string(
    "image_dir",
    None,
    "Directory containing downloaded images from the shop the look dataset.")
_OUTDIR = flags.DEFINE_string("out_dir", "/tmp", "Output directory.")
_OUTPUT_SIZE = flags.DEFINE_integer("output_size", 64, "Size of embeddings.")
_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    None,
    "Model name.")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 8, "Batch size.")

flags.mark_flag_as_required("model_name")
flags.mark_flag_as_required("image_dir")


def main(argv):
    """Main function."""
    del argv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_product = pin_util.get_valid_scene_product(_IMAGE_DIRECTORY.value, _INPUT_FILE.value)
    logging.info("Found %d valid scene product pairs." % len(scene_product))
    unique_scenes = list(set(x[0] for x in scene_product))
    unique_products = list(set(x[1] for x in scene_product))
    logging.info("Found %d unique scenes.", len(unique_scenes))
    logging.info("Found %d unique products.", len(unique_products))

    model = models.STLModel(output_size=_OUTPUT_SIZE.value).to(device)
    logging.info("Attempting to read model %s", _MODEL_NAME.value)
    model.load_state_dict(torch.load(_MODEL_NAME.value, map_location=device, weights_only=True))
    model.eval()

    batch_size = _BATCH_SIZE.value

    scene_dataset = input_pipeline.ImageWithIdDataset(unique_scenes)
    scene_loader = DataLoader(scene_dataset, batch_size=batch_size, num_workers=4)

    scene_dict = {}
    count = 0
    with torch.no_grad():
        for paths, images in scene_loader:
            count += 1
            if count % 100 == 0:
                logging.info("Created %d scene embeddings", count * batch_size)
            images = images.to(device)
            result = model.get_scene_embed(images)
            for i in range(len(paths)):
                scene_dict[paths[i]] = result[i].cpu().numpy().tolist()

    scene_filename = os.path.join(_OUTDIR.value, "scene_embed.json")
    with open(scene_filename, "w") as scene_file:
        json.dump(scene_dict, scene_file)

    product_dataset = input_pipeline.ImageWithIdDataset(unique_products)
    product_loader = DataLoader(product_dataset, batch_size=batch_size, num_workers=4)

    product_dict = {}
    count = 0
    with torch.no_grad():
        for paths, images in product_loader:
            count += 1
            if count % 100 == 0:
                logging.info("Created %d product embeddings", count * batch_size)
            images = images.to(device)
            result = model.get_product_embed(images)
            for i in range(len(paths)):
                product_dict[paths[i]] = result[i].cpu().numpy().tolist()

    product_filename = os.path.join(_OUTDIR.value, "product_embed.json")
    with open(product_filename, "w") as product_file:
        json.dump(product_dict, product_file)


if __name__ == "__main__":
    app.run(main)
