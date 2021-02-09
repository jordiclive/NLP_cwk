
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
device = 'cpu'


def train(optimizer,train_iter, dev_iter, model, number_epoch):
    """
    Training loop for the model, which calls on eval to evaluate after each epoch
    """

    print("Training model.")

    for epoch in range(1, number_epoch + 1):

        model.train()
        epoch_loss = 0
        epoch_sse = 0
        no_observations = 0  # Observations used for training so far

        for batch in train_iter:
            feature, target = batch

            feature, target = feature.to(device), target.to(device)

            # for RNN:
            model.batch_size = target.shape[0]
            no_observations = no_observations + target.shape[0]
            model.hidden = model.init_hidden()

            predictions = model(feature).squeeze(1)

            optimizer.zero_grad()

            loss = model.loss_fn(predictions, target)

            sse, __ = model_performance(predictions.detach().cpu().numpy(), target.detach().cpu().numpy())

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * target.shape[0]
            epoch_sse += sse

        valid_loss, valid_mse, __, __ = eval(dev_iter, model)

        epoch_loss, epoch_mse = epoch_loss / no_observations, epoch_sse / no_observations
        print(
            f'| Epoch: {epoch:02} | Train Loss: {epoch_loss:.2f} | Train MSE: {epoch_mse:.2f} | Train RMSE: {epoch_mse ** 0.5:.2f} | \
        Val. Loss: {valid_loss:.2f} | Val. MSE: {valid_mse:.2f} |  Val. RMSE: {valid_mse ** 0.5:.2f} |')

# We evaluate performance on our dev set
def eval(data_iter, model):
    """
    Evaluating model performance on the dev set
    """
    model.eval()
    epoch_loss = 0
    epoch_sse = 0
    pred_all = []
    trg_all = []
    no_observations = 0

    with torch.no_grad():
        for batch in data_iter:
            feature, target = batch

            feature, target = feature.to(device), target.to(device)

            # for RNN:
            model.batch_size = target.shape[0]
            no_observations = no_observations + target.shape[0]
            model.hidden = model.init_hidden()

            predictions = model(feature).squeeze(1)
            loss = model.loss_fn(predictions, target)

            # We get the mse
            pred, trg = predictions.detach().cpu().numpy(), target.detach().cpu().numpy()
            sse, __ = model_performance(pred, trg)

            epoch_loss += loss.item()*target.shape[0]
            epoch_sse += sse
            pred_all.extend(pred)
            trg_all.extend(trg)

    return epoch_loss/no_observations, epoch_sse/no_observations, np.array(pred_all), np.array(trg_all)


# How we print the model performance
def model_performance(output, target, print_output=False):
    """
    Returns SSE and MSE per batch (printing the MSE and the RMSE)
    """

    sq_error = (output - target)**2

    sse = np.sum(sq_error)
    mse = np.mean(sq_error)
    rmse = np.sqrt(mse)

    if print_output:
        print(f'| MSE: {mse:.2f} | RMSE: {rmse:.2f} |')

    return sse, mse

if __name__ == '__main__':

    import pickle
    from models import *
    from data import *

    # Approach 2
    # Load data
    train_df = pd.read_csv('../data/task-1/train.csv')
    test_df = pd.read_csv('../data/task-1/dev.csv')

    predicted_train, training_y, predicted, dev_y = non_embed_approach(train_df,test_df,0.8)
    # We run the evaluation:
    print("\nTrain performance:")
    sse, mse = model_performance(predicted_train, training_y, True)

    print("\nDev performance:")
    sse, mse = model_performance(predicted, dev_y, True)


    # Approach 1
    with open('train_dataset.pickle', 'rb') as handle:
        train_dataset = pickle.load(handle)
    with open('dev_dataset.pickle', 'rb') as handle:
        dev_dataset = pickle.load(handle)

    with open('wvecs.pickle', 'rb') as handle:
        wvecs = pickle.load(handle)

    INPUT_DIM = len(wvecs)# 5306
    EMBEDDING_DIM = 100
    BATCH_SIZE = 32
    epochs = 5
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle = True, batch_size = BATCH_SIZE,
                                               collate_fn = collate_fn_padd)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size = BATCH_SIZE, collate_fn = collate_fn_padd)



    model = BiLSTM(EMBEDDING_DIM, 50, INPUT_DIM, BATCH_SIZE, device)
    model.embedding.weight.data.copy_(torch.from_numpy(wvecs))

    optimizer = torch.optim.AdamW(model.parameters())

    train(optimizer,train_loader, dev_loader, model, epochs)