#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  Debugging utilities for PyTorch models.
"""

import numpy as np
import torch


def dump_word_knn(model, tokens, token_dictionary, max_terms, device):
    model.eval()
    with torch.no_grad():
        tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        target_embeddings = model.word_embedding(tokens)
        all_indices = torch.arange(model.word_embedding.num_embeddings, device=device)
        all_embeddings = model.word_embedding(all_indices)
        results = torch.matmul(target_embeddings, all_embeddings.T)
        results = results.cpu().numpy()

        num_embeddings = model.word_embedding.num_embeddings
        count = min(max_terms, num_embeddings)
        for i in range(len(tokens)):
            far_to_near_indices = np.argsort(results[i])
            result_list = []
            for j in range(count):
                idx = far_to_near_indices[num_embeddings - 1 - j]
                sim = results[i][idx]
                other_token = token_dictionary.get_token_from_embedding_index(idx)
                display = '%s:%3f' % (other_token, sim)
                result_list.append(display)
            print('Nearest to %s: %s' % (token_dictionary.get_token_from_embedding_index(tokens[i].item()),
                                          ','.join(result_list)))


def dump_sentence_knn(model, sentences, max_sentence_length, max_terms,
                      token_dictionary, title_dictionary, device):
    model.eval()
    with torch.no_grad():
        indices = []
        for sentence in sentences:
            tokens = token_dictionary.simple_tokenize(sentence)
            idx = token_dictionary.get_embedding_indices(tokens)
            if len(idx) > max_sentence_length:
                idx = idx[:max_sentence_length]
            else:
                while len(idx) < max_sentence_length:
                    idx.append(0)
            indices.append(idx)
        indices = torch.tensor(indices, dtype=torch.long, device=device)

        sentence_embeddings = model.encode_sentence(indices)

        num_embeddings = title_dictionary.get_dictionary_size()
        all_indices = torch.arange(num_embeddings, device=device)
        url_embeddings = model.url_embedding(all_indices)

        distances = -torch.matmul(sentence_embeddings, url_embeddings.T)
        distances = distances.cpu().numpy()

        count = min(max_terms, num_embeddings)
        for i in range(len(sentences)):
            print('Nearest to %s' % sentences[i])
            near_to_far_indices = np.argsort(distances[i])
            result_list = []
            for j in range(count):
                idx = near_to_far_indices[j]
                sim = -distances[i][idx]
                other_token = title_dictionary.get_token(idx)
                display = '%s:%3f' % (other_token, sim)
                result_list.append(display)
            print('Nearest to %s: %s' % (sentences[i], ','.join(result_list)))
