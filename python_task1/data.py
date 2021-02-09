import json
from dataclasses import dataclass
from typing import List, NamedTuple

import pytorch_lightning as pl
import torch
from sentencepiece import SentencePieceProcessor
import pandas as pd
import numpy as np
import re
import codecs
# Imports

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset, random_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import codecs

def create_vocab(data):
    """
    Creating a corpus of all the tokens used
    """
    tokenized_corpus = [] # Let us put the tokenized corpus in a list

    for sentence in data:

        tokenized_sentence = []

        for token in sentence.split(' '): # simplest split is

            tokenized_sentence.append(token)

        tokenized_corpus.append(tokenized_sentence)

    # Create single list of all vocabulary
    vocabulary = []  # Let us put all the tokens (mostly words) appearing in the vocabulary in a list

    for sentence in tokenized_corpus:

        for token in sentence:

            if token not in vocabulary:

                if True:
                    vocabulary.append(token)

    return vocabulary, tokenized_corpus

def word2idx(embeddings,joint_vocab):
    # We create representations for our tokens
    wvecs = []  # word vectors
    word2idx = []  # word2index
    idx2word = []

    # This is a large file, it will take a while to load in the memory!
    with codecs.open(embeddings, 'r', 'utf-8') as f:
        index = 1
        for line in f.readlines():
            # Ignore the first line - first line typically contains vocab, dimensionality
            if len(line.strip().split()) > 3:
                word = line.strip().split()[0]
                if word in joint_vocab:
                    (word, vec) = (word,
                                   list(map(float, line.strip().split()[1:])))
                    wvecs.append(vec)
                    word2idx.append((word, index))
                    idx2word.append((index, word))
                    index += 1

    wvecs = np.array(wvecs)
    word2idx = dict(word2idx)
    idx2word = dict(idx2word)
    return wvecs, word2idx, idx2word



def collate_fn_padd(batch):
    '''
    We add padding to our minibatches and create tensors for our model
    '''

    batch_labels = [l for f, l in batch]
    batch_features = [f for f, l in batch]

    batch_features_len = [len(f) for f, l in batch]

    seq_tensor = torch.zeros((len(batch), max(batch_features_len))).long()

    for idx, (seq, seqlen) in enumerate(zip(batch_features, batch_features_len)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

    batch_labels = torch.FloatTensor(batch_labels)

    return seq_tensor, batch_labels

class Task1Dataset(Dataset):

    def __init__(self, train_data, labels):
        self.x_train = train_data
        self.y_train = labels

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, item):
        return self.x_train[item], self.y_train[item]

if __name__ == '__main__':
    from models import *
    from data import *
    import pickle

    train_proportion = 0.8
    epochs = 10
    EMBEDDING_DIM = 100
    BATCH_SIZE = 32

    # Load data
    train_df = pd.read_csv('../data/task-1/train.csv')
    test_df = pd.read_csv('../data/task-1/dev.csv')
    training_data = train_df['original']
    test_data = test_df['original']

    training_vocab, training_tokenized_corpus = create_vocab(training_data)
    test_vocab, test_tokenized_corpus = create_vocab(test_data)

    # Creating joint vocab from test and train:
    joint_vocab, joint_tokenized_corpus = create_vocab(pd.concat([training_data, test_data]))

    embeddings = 'glove.6B.100d.txt'
    wvecs, word2idx, idx2word = word2idx(embeddings, joint_vocab)

    vectorized_seqs = [[word2idx[tok] for tok in seq if tok in word2idx] for seq in training_tokenized_corpus]
    vectorized_seqs = [x if len(x) > 0 else [0] for x in vectorized_seqs]

    train_and_dev = Task1Dataset(vectorized_seqs, train_df['meanGrade'])


    train_examples = round(len(train_and_dev) * train_proportion)
    dev_examples = len(train_and_dev) - train_examples
    train_dataset, dev_dataset = random_split(train_and_dev,
                                              (train_examples,
                                               dev_examples))

    with open('train_dataset.pickle', 'wb') as handle:
        pickle.dump(train_dataset, handle, protocol = pickle.HIGHEST_PROTOCOL)
    with open('dev_dataset.pickle', 'wb') as handle:
        pickle.dump(dev_dataset, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open('wvecs.pickle', 'wb') as handle:
        pickle.dump(wvecs, handle, protocol = pickle.HIGHEST_PROTOCOL)

