#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  Given embedding files makes recommendations.
"""

import json
import os
from typing import Any, Dict, Tuple

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch

import pin_util

FLAGS = flags.FLAGS
_PRODUCT_EMBED_ = flags.DEFINE_string("product_embed", None, "Product embedding json.")
_SCENE_EMBED_ = flags.DEFINE_string("scene_embed", None, "Scene embedding json.")
_TOP_K = flags.DEFINE_integer("top_k", 10, "Number of top scoring products to return per scene.")
_OUTPUT_DIR = flags.DEFINE_string("output_dir", "/tmp", "Location to write output.")
_MAX_RESULTS = flags.DEFINE_integer("max_results", 100, "Max scenes to score.")

flags.mark_flag_as_required("product_embed")
flags.mark_flag_as_required("scene_embed")


def find_top_k(
    scene_embedding: torch.Tensor,
    product_embeddings: torch.Tensor,
    k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
  Finds the top K nearest product embeddings to the scene embedding.
  """
    scores = (scene_embedding * product_embeddings).sum(dim=-1)
    top_k_scores, top_k_indices = torch.topk(scores, k)
    return top_k_scores, top_k_indices


def local_file_to_pin_url(filename):
    """Converts a local filename to a pinterest url."""
    key = filename.split("/")[-1]
    key = key.split(".")[0]
    url = pin_util.key_to_url(key)
    result = "<img src=\"%s\">" % url
    return result


def save_results(
    filename: str,
    scene_key: str,
    scores_and_indices: Tuple[torch.Tensor, torch.Tensor],
    index_to_key: Dict[int, str]):
    """
  Save results of a scoring run as a html document.
  """
    scores, indices = scores_and_indices
    scores = scores.cpu().numpy()
    indices = indices.cpu().numpy()
    with open(filename, "w") as f:
        f.write("<HTML>\n")
        scene_img = local_file_to_pin_url(scene_key)
        f.write("Nearest neighbors to %s<br>\n" % scene_img)
        for i in range(scores.shape[0]):
            idx = indices[i]
            product_key = index_to_key[idx]
            product_img = local_file_to_pin_url(product_key)
            f.write("Rank %d Score %f<br>%s<br>\n" % (i + 1, scores[i], product_img))
        f.write("</HTML>\n")


def main(argv):
    """Main function."""
    del argv

    with open(_PRODUCT_EMBED_.value, "r") as f:
        product_dict = json.load(f)
    with open(_SCENE_EMBED_.value, "r") as f:
        scene_dict = json.load(f)

    index_to_key = {}
    product_embeddings = []
    for index, kv in enumerate(product_dict.items()):
        key, vec = kv
        index_to_key[index] = key
        product_embeddings.append(np.array(vec))
    product_embeddings = torch.tensor(np.stack(product_embeddings, axis=0), dtype=torch.float32)

    for index, kv in enumerate(scene_dict.items()):
        scene_key, scene_vec = kv
        scene_embed = torch.tensor(scene_vec, dtype=torch.float32).unsqueeze(0)
        scores_and_indices = find_top_k(scene_embed, product_embeddings, _TOP_K.value)
        filename = os.path.join(_OUTPUT_DIR.value, "%05d.html" % index)
        save_results(filename, scene_key, scores_and_indices, index_to_key)
        if index > _MAX_RESULTS.value:
            break


if __name__ == "__main__":
    app.run(main)
