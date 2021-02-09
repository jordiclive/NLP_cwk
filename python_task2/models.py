

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from torch.utils.data import Dataset, random_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import codecs

class BiLSTM_classification(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, batch_size, device):
        super(BiLSTM_classification, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.device = device
        self.batch_size = batch_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2label = nn.Linear(hidden_dim * 2, 3)
        self.hidden = self.init_hidden()
        self.loss_fn = nn.CrossEntropyLoss()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim)
        return torch.zeros(2, self.batch_size, self.hidden_dim).to(self.device), \
               torch.zeros(2, self.batch_size, self.hidden_dim).to(self.device)

    def forward(self, sentence):
        embedded = self.embedding(sentence)
        embedded = embedded.permute(1, 0, 2)

        lstm_out, self.hidden = self.lstm(
            embedded.view(len(embedded), self.batch_size, self.embedding_dim), self.hidden)

        out = self.hidden2label(lstm_out[-1])
        return out


def non_embed_approach(train_df,test_df,train_proportion):

    train_and_dev = train_df['edit1']

    training_data, dev_data, training_y, dev_y = train_test_split(train_df['edit1'], train_df['label'],
                                                                  test_size = (1 - train_proportion),
                                                                  random_state = 42)

    # We train a Tf-idf model
    count_vect = CountVectorizer(stop_words = 'english')
    train_counts = count_vect.fit_transform(training_data)
    transformer = TfidfTransformer().fit(train_counts)
    train_counts = transformer.transform(train_counts)
    naive_model = MultinomialNB().fit(train_counts, training_y)

    # Train predictions
    predicted_train = naive_model.predict(train_counts)

    # Calculate Tf-idf using train and dev, and validate on dev:
    test_and_test_counts = count_vect.transform(train_and_dev)
    transformer = TfidfTransformer().fit(test_and_test_counts)

    test_counts = count_vect.transform(dev_data)

    test_counts = transformer.transform(test_counts)

    # Dev predictions
    predicted = naive_model.predict(test_counts)

    return predicted_train, training_y, predicted, dev_y

