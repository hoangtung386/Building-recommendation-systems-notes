#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  Trains the co-occurrence matrix.
  See the GloVe paper for the math.
  https://nlp.stanford.edu/pubs/glove.pdf
"""

import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
import wandb

from cooccurrence_matrix import CooccurrenceGenerator, create_dataloader
from models import Glove
from token_dictionary import TokenDictionary


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "train_input_pattern",
    "data/wikipedia.cooccur.pb.b64.bz2/part-?????.bz2",
    "Input cooccur.pb.b64.bz2 file pattern.")
flags.DEFINE_string(
    "token_dictionary",
    "data/dictionaries/token.tstat.pb.b64.bz2",
    "The token dictionary file.")
flags.DEFINE_integer("max_terms", 20, "Max terms per row to dump")
flags.DEFINE_integer("embedding_dim", 64,
                     "Embedding dimension.")
flags.DEFINE_integer("batch_size", 2048,
                     "Batch size")
flags.DEFINE_integer("seed", 1701,
                     "Random number seed.")
flags.DEFINE_integer("shuffle_buffer_size", 5000000,
                     "Shuffle buffer size")
flags.DEFINE_string(
    "terms",
    "news,apple,computer,physics,neural,democracy,singapore,livermore",
    "CSV of terms to dump")
flags.DEFINE_string(
    "checkpoint_dir",
    "data/wikipedia_training",
    "Location to save checkpoints.")
flags.DEFINE_integer("checkpoint_every_epochs", 20, "Number of epochs to checkpoint.")
flags.DEFINE_string("resume_checkpoint", None, "If not None, resume from this checkpoint.")
flags.DEFINE_integer("steps_per_epoch", 10000,
                     "Number of training steps per epoch")
flags.DEFINE_integer("num_epochs", 20,
                     "Number of epochs")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")

flags.mark_flag_as_required("train_input_pattern")


def glove_loss(model, inputs, target):
    predicted = model(inputs)
    ones = torch.ones_like(target)
    weight = torch.minimum(ones, target / 100.0)
    weight = torch.pow(weight, 0.75)
    log_target = torch.log10(1.0 + target)
    loss = torch.mean(torch.square(log_target - predicted) * weight)
    return loss


def find_knn(model, token):
    model.eval()
    with torch.no_grad():
        scores = model.score_all(token)
        indices = torch.argsort(scores, dim=0)
    return scores, indices


def train_epoch(model, optimizer, steps_per_epoch, train_it, device):
    model.train()
    epoch_loss = []
    for i in range(steps_per_epoch):
        inputs, targets = next(train_it)
        inputs = [t.to(device) for t in inputs]
        targets = targets.to(device)
        optimizer.zero_grad()
        loss = glove_loss(model, inputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
    train_loss = np.mean(epoch_loss)
    return train_loss


def dump_knn(model, tokens, token_dictionary, device):
    model.eval()
    with torch.no_grad():
        tokens = tokens.to(device)
        for i in range(tokens.shape[0]):
            token = tokens[i:i + 1]
            query_word = token_dictionary.get_token_from_embedding_index(token.item())
            scores = model.score_all(token)
            indices = torch.argsort(scores, dim=0)
            knn = []
            for j in range(10):
                idx = indices[-j - 1].item()
                word = token_dictionary.get_token_from_embedding_index(idx)
                score = scores[idx].item()
                knn.append("%s:%f" % (word, score))
            logging.info("Nearest neighbors for %s: %s", query_word, " ".join(knn))


def save_state(model, optimizer, step, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = os.path.join(checkpoint_dir, "checkpoint-%05d.pt" % step)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, filename)


def main(argv):
    """Main function."""
    del argv
    init_config = dict(
        seed=FLAGS.seed,
        learning_rate=FLAGS.learning_rate)
    run = wandb.init(config=init_config)
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    token_dictionary = TokenDictionary(FLAGS.token_dictionary)
    num_tokens = token_dictionary.get_embedding_dictionary_size()

    debug_tokens = []
    for word in FLAGS.terms.split(','):
        token = token_dictionary.get_embedding_index(word)
        debug_tokens.append(token)
    debug_tokens = torch.tensor(debug_tokens, dtype=torch.long)

    model = Glove(num_embeddings=num_tokens, features=FLAGS.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_data = CooccurrenceGenerator(FLAGS.train_input_pattern)
    train_iterator = train_data.get_batch(FLAGS.batch_size, FLAGS.shuffle_buffer_size)

    if FLAGS.resume_checkpoint:
        logging.info("Resuming from %s", FLAGS.resume_checkpoint)
        checkpoint = torch.load(FLAGS.resume_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for step in range(FLAGS.num_epochs):
        logging.info("Step %d", step)
        if step % FLAGS.checkpoint_every_epochs == 0:
            save_state(model, optimizer, step, FLAGS.checkpoint_dir)
        dump_knn(model, debug_tokens, token_dictionary, device)
        train_loss = train_epoch(model, optimizer, FLAGS.steps_per_epoch, train_iterator, device)
        logging.info("Training loss %f", train_loss)
        wandb.log({"train_loss": train_loss})

    os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)
    art = wandb.Artifact(f"glove-wikipedia-{wandb.run.id}", type="model")
    model_path = os.path.join(FLAGS.checkpoint_dir, "glove.pt")
    torch.save(model.state_dict(), model_path)
    art.add_file(model_path)
    wandb.log_artifact(art)


if __name__ == "__main__":
    app.run(main)
