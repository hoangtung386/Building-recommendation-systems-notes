#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  Trains the text to url and url 2 url.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from absl import app
from absl import flags
from token_dictionary import TokenDictionary
from ioutil import proto_generator, shuffle_generator
import nlp_pb2 as nlp_pb

FLAGS = flags.FLAGS
flags.DEFINE_string("txt2url_train_input_pattern", None, "Input sdoc.pb.b64.bz2 file pattern.")
flags.DEFINE_string("txt2url_validation_input_pattern", None, "Input sdoc.pb.b64.bz2 file pattern.")
flags.DEFINE_string("url2url_train_input_pattern", None, "Input coccur.pb.b64.bz2 file pattern.")
flags.DEFINE_string("url2url_validation_input_pattern", None, "Input coccur.pb.b64.bz2 file pattern.")
flags.DEFINE_string("token_dictionary", None, "The token dictionary file.")
flags.DEFINE_string("title_dictionary", None, "The title dictionary file.")
flags.DEFINE_string("word_embedding", None, "HDF5 model of the word embedding.")
flags.DEFINE_integer("sentence_length", 64, "Max number of words in a sentence.")
flags.DEFINE_integer("max_sentence_per_example", 8, "Max number sentences per example.")
flags.DEFINE_integer("max_terms", 20, "Max terms per row to dump")
flags.DEFINE_integer("word_embedding_dim", 64, "Embedding dimension for the words.")
flags.DEFINE_integer("url_embedding_dim", 8, "Embedding dimension for the urls.")
flags.DEFINE_integer("rnn_size", 128, "RNN cell size.")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("shuffle_buffer_size", 1000, "Shuffle buffer size")
flags.DEFINE_string("sentence_csv", None, "CSV of terms to dump")
flags.DEFINE_string("tensorboard_dir", None, "Location to store training logs.")
flags.DEFINE_string("checkpoint_dir", None, "Location to save checkpoints.")
flags.DEFINE_string("loss_type", "MSE", "Type of loss")
flags.DEFINE_integer("steps_per_epoch", 1000, "Number of steps per epoch")
flags.DEFINE_integer("num_epochs", 100, "Number of epochs")
flags.DEFINE_integer("validation_steps", 100, "Number of validation steps")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_float("learning_rate_decay", 0.9, "Learning rate decay")
flags.DEFINE_float("url_max_norm", 10.0, "Max norm for url embedding")
flags.DEFINE_float("text_l2", 0.0, "Text l2 regularization.")
flags.DEFINE_float("margin", 0.1, "Margin for text similarity.")


class Txt2UrlModel(nn.Module):
    def __init__(self, num_word_tokens, word_embedding_dim, url_embedding_dim, rnn_size,
                 sentence_length, max_title_embedding, url_max_norm, text_l2, margin=0.1):
        super().__init__()
        self.word_embedding = nn.Embedding(num_word_tokens, word_embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(word_embedding_dim, rnn_size, batch_first=True)
        self.sentence_to_url = nn.Linear(rnn_size, url_embedding_dim)
        self.url_embedding = nn.Embedding(max_title_embedding, url_embedding_dim)
        self.url_max_norm = url_max_norm
        self.text_l2 = text_l2
        self.sentence_length = sentence_length
        self.margin = margin

    def _clip_url_embedding(self):
        with torch.no_grad():
            self.url_embedding.weight.data = F.normalize(
                self.url_embedding.weight.data, p=2, dim=-1
            ) * self.url_max_norm

    def encode_sentence(self, sentence_input):
        emb = self.word_embedding(sentence_input)
        lstm_out, _ = self.lstm(emb)
        sentence_repr = lstm_out[:, -1, :]
        return self.sentence_to_url(sentence_repr)

    def forward(self, url_near_text_input, sentence_input, url_near1_input, url_near2_input):
        self._clip_url_embedding()
        url_near_text = self.url_embedding(url_near_text_input).squeeze(1)
        sentence_emb = self.encode_sentence(sentence_input)
        url_near1 = self.url_embedding(url_near1_input).squeeze(1)
        url_near2 = self.url_embedding(url_near2_input).squeeze(1)

        text_sim = torch.sum(sentence_emb * url_near_text, dim=-1)
        text_loss = F.relu(self.margin - text_sim) ** 2

        url_dice = torch.sum(url_near1 * url_near2, dim=-1)

        return text_loss, url_dice


def url_triplet_generator(input_pattern, title_dictionary):
    gen = proto_generator(input_pattern, nlp_pb.CooccurrenceRow)
    while True:
        row = next(gen)
        main_count = title_dictionary.get_doc_frequency(row.index)
        for j in range(len(row.other_index)):
            idx = row.other_index[j]
            joint_count = row.count[j]
            doc_count = title_dictionary.get_doc_frequency(idx)
            dice = 2.0 * joint_count / (doc_count + main_count)
            yield (row.index, idx, dice)


def txt2url_generator(input_pattern, sentence_length, max_sentence_per_example):
    gen = proto_generator(input_pattern, nlp_pb.SparseDocument)
    while True:
        sdoc = next(gen)
        num_tokens_in_page = len(sdoc.token_index)
        length_diff = sentence_length - num_tokens_in_page
        tokens = list(sdoc.token_index)
        if length_diff == 0:
            yield (sdoc.primary_index, tokens)
        elif length_diff > 0:
            for i in range(length_diff):
                tokens.append(0)
            yield (sdoc.primary_index, tokens)
        else:
            for i in range(max_sentence_per_example):
                idx = np.random.randint(0, num_tokens_in_page - sentence_length)
                yield (sdoc.primary_index, tokens[idx:idx + sentence_length])


def triplet_generator(
        url2url_pattern, txt2url_pattern, batch_size, shuffle_size,
        sentence_length, max_sentence_per_example, title_dictionary):
    url_triplet = url_triplet_generator(url2url_pattern, title_dictionary)
    if shuffle_size > 0:
        url_triplet = shuffle_generator(url_triplet, shuffle_size)
    txt2url_triplet = txt2url_generator(txt2url_pattern, sentence_length, max_sentence_per_example)
    while True:
        url_near_text = []
        sentence_input = []
        url1 = []
        url2 = []
        score = []
        for i in range(batch_size):
            url_near, tokens = next(txt2url_triplet)
            url_near_text.append(url_near)
            sentence_input.append(tokens)
            a, b, c = next(url_triplet)
            url1.append(a)
            url2.append(b)
            score.append(c)

        x = {
            "url_near_text": torch.tensor(url_near_text, dtype=torch.long),
            "sentence_input": torch.tensor(sentence_input, dtype=torch.long),
            "url1": torch.tensor(url1, dtype=torch.long),
            "url2": torch.tensor(url2, dtype=torch.long),
        }
        y_text = torch.zeros(batch_size, dtype=torch.float32)
        y_url = torch.sqrt(torch.tensor(score, dtype=torch.float32))
        yield x, (y_text, y_url)


def main(argv):
    """Main function."""
    del argv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    token_dictionary = TokenDictionary(FLAGS.token_dictionary)
    title_dictionary = TokenDictionary(FLAGS.title_dictionary)
    max_title_embedding = title_dictionary.get_dictionary_size()
    num_word_tokens = token_dictionary.get_embedding_dictionary_size()

    model = Txt2UrlModel(
        num_word_tokens=num_word_tokens,
        word_embedding_dim=FLAGS.word_embedding_dim,
        url_embedding_dim=FLAGS.url_embedding_dim,
        rnn_size=FLAGS.rnn_size,
        sentence_length=FLAGS.sentence_length,
        max_title_embedding=max_title_embedding,
        url_max_norm=FLAGS.url_max_norm,
        text_l2=FLAGS.text_l2,
        margin=FLAGS.margin,
    ).to(device)

    if FLAGS.word_embedding:
        state_dict = torch.load(FLAGS.word_embedding, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=FLAGS.learning_rate)

    train_iterator = triplet_generator(
        FLAGS.url2url_train_input_pattern,
        FLAGS.txt2url_train_input_pattern,
        FLAGS.batch_size,
        FLAGS.shuffle_buffer_size,
        FLAGS.sentence_length,
        FLAGS.max_sentence_per_example,
        title_dictionary)

    validation_iterator = triplet_generator(
        FLAGS.url2url_validation_input_pattern,
        FLAGS.txt2url_validation_input_pattern,
        FLAGS.batch_size,
        0,
        FLAGS.sentence_length,
        FLAGS.max_sentence_per_example,
        title_dictionary)

    loss_fn_url = nn.MSELoss() if FLAGS.loss_type == "MSE" else nn.L1Loss()

    for epoch in range(FLAGS.num_epochs):
        model.train()
        epoch_loss = []
        for step in range(FLAGS.steps_per_epoch):
            x, (y_text, y_url) = next(train_iterator)
            x = {k: v.to(device) for k, v in x.items()}
            y_text = y_text.to(device)
            y_url = y_url.to(device)

            optimizer.zero_grad()
            text_loss, url_dice = model(
                x["url_near_text"], x["sentence_input"], x["url1"], x["url2"])
            text_loss_val = F.l1_loss(text_loss, y_text)
            url_loss_val = loss_fn_url(url_dice, y_url)
            total_loss = text_loss_val + url_loss_val
            total_loss.backward()
            optimizer.step()
            epoch_loss.append(total_loss.item())

        lr = FLAGS.learning_rate * (FLAGS.learning_rate_decay ** (epoch + 1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        print("Epoch %d, loss: %.4f" % (epoch, np.mean(epoch_loss)))

        model.eval()
        val_loss = []
        with torch.no_grad():
            for step in range(FLAGS.validation_steps):
                x, (y_text, y_url) = next(validation_iterator)
                x = {k: v.to(device) for k, v in x.items()}
                y_text = y_text.to(device)
                y_url = y_url.to(device)
                text_loss, url_dice = model(
                    x["url_near_text"], x["sentence_input"], x["url1"], x["url2"])
                text_loss_val = F.l1_loss(text_loss, y_text)
                url_loss_val = loss_fn_url(url_dice, y_url)
                val_loss.append((text_loss_val + url_loss_val).item())
        print("Validation loss: %.4f" % np.mean(val_loss))

        if FLAGS.checkpoint_dir:
            os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(FLAGS.checkpoint_dir, "model_epoch_%03d.pt" % epoch))


if __name__ == "__main__":
    app.run(main)
