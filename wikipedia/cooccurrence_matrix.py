#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Cooccurrence matrix library.
"""

import base64
import bz2
import glob
from random import shuffle

import nlp_pb2 as nlp_pb
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader


class CooccurrenceMatrix:

    def __reset(self):
        self.__matrix = {}

    def debug_print(self, max_rows, token_dictionary, num_terms):
        count = 0
        for key in self.__matrix.keys():
            row = sorted(self.__matrix[key], key=lambda x: x[1], reverse=True)
            num_others = len(row)
            token = token_dictionary.get_token_from_embedding_index(key)
            print('Token [%s]' % token)
            nt = min(num_terms, num_others)
            for i in range(nt):
                token = token_dictionary.get_token_from_embedding_index(row[i][0])
                print(' %s : %f' % (token, row[i][1]))
            count = count + 1
            if count > max_rows:
                break

    def load(self, input_file):
        self.__reset()
        with bz2.open(input_file, 'rb') as file:
            for line in file:
                line = line[:-1]
                serialized = base64.b64decode(line)
                proto = nlp_pb.CooccurrenceRow()
                proto.ParseFromString(serialized)
                if proto.index not in self.__matrix:
                    self.__matrix[proto.index] = []
                for i in range(len(proto.other_index)):
                    self.__matrix[proto.index].append((proto.other_index[i], proto.count[i]))

    def __init__(self, input_file):
        self.load(input_file)


class CooccurrenceGenerator:
    def __init__(self, input_pattern):
        self._input_files = glob.glob(input_pattern)
        self._total_files = len(self._input_files)

    def get_item(self):
        while True:
            file_epoch = 0
            for input_file in self._input_files:
                file_epoch += 1
                print('Opening %s (%d of %d)' % (input_file, file_epoch, self._total_files))
                with bz2.open(input_file, 'rb') as file:
                    for line in file:
                        line = line[:-1]
                        serialized = base64.b64decode(line)
                        proto = nlp_pb.CooccurrenceRow()
                        proto.ParseFromString(serialized)
                        count = len(proto.other_index)
                        for i in range(count):
                            yield (proto.index, proto.other_index[i], proto.count[i])

    def get_shuffled_items(self, num_items):
        iterator = self.get_item()
        while True:
            items = [next(iterator) for _ in range(num_items)]
            np.random.shuffle(items)
            for item in items:
                yield item

    def get_batch(self, batch_size, shuffle_size=0):
        if shuffle_size:
            iterator = self.get_shuffled_items(shuffle_size)
        else:
            iterator = self.get_item()
        while True:
            token1 = []
            token2 = []
            token_count = []
            for _ in range(batch_size):
                item = next(iterator)
                token1.append(item[0])
                token2.append(item[1])
                token_count.append(item[2])
            x = [np.asarray(token1, dtype=np.int32),
                 np.asarray(token2, dtype=np.int32)]
            y = np.asarray(token_count, dtype=np.float32)
            yield (x, y)


class CooccurrenceIterableDataset(IterableDataset):
    def __init__(self, generator, batch_size, shuffle_size=0):
        self._generator = generator
        self._batch_size = batch_size
        self._shuffle_size = shuffle_size

    def __iter__(self):
        batch_iter = self._generator.get_batch(self._batch_size, self._shuffle_size)
        while True:
            try:
                x, y = next(batch_iter)
                yield (
                    torch.tensor(x[0], dtype=torch.long),
                    torch.tensor(x[1], dtype=torch.long),
                ), torch.tensor(y, dtype=torch.float32)
            except StopIteration:
                break


def create_dataloader(generator, batch_size, shuffle_size=0, num_workers=0):
    dataset = CooccurrenceIterableDataset(generator, batch_size, shuffle_size)
    return DataLoader(dataset, batch_size=None, num_workers=num_workers, pin_memory=True)
