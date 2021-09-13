from dataloader import MonitorLearningDataset
from net import Net

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

import argparse
import pickle
import random


window_size = 5
characteristics = 9
learning_rate = 1e-4
log_batch = 100

batch_size = 2
n_workers = 8
train_ratio = 0.8


def train():
    model = Net(window_size, characteristics)
    model = model.double()
    if gpu_exists:
        model = model.cuda()

    traces_train = MonitorLearningDataset(train_set)
    traces_val = MonitorLearningDataset(val_set)
    validation_set = DataLoader(traces_val,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=n_workers)
    training_set = DataLoader(traces_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=n_workers)
    traces_dataset = { 'validation': validation_set, 'train': training_set }

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    max_loss = 1e20
    for epoch in range(args.epochs):
        for phase in ['train', 'validation']:
            cumu_loss = 0
            break_line = '-' * 20
            print(f'{break_line}\n{phase} at epoch no. {epoch + 1}\n{break_line}')
            for i, sample in enumerate(traces_dataset[phase]):
                x, y = sample

                if gpu_exists:
                    x = x.cuda()
                    y = y.cuda()

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                cumu_loss += loss
                if phase == 'train':
                    model.zero_grad()

                    loss.backward()
                    optimizer.step()

                if i != 0 and i % log_batch == 0:
                    print(f'batch {i}: loss = {cumu_loss / log_batch}')
                    cumu_loss = 0

            # save_model = model.module if torch.cuda.device_count() > 1 else model
            save_model = model
            torch.save(save_model.state_dict(), chkpt_path)
            if phase == 'validation':
                if cumu_loss / len(traces_dataset['validation']) < max_loss:
                    max_loss = cumu_loss
                    torch.save(save_model.state_dict(), best_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', help='Number of Epochs', type=int, default=5)
    parser.add_argument('-t', '--train', help='.pkl file containing the training traces', default='training_data.pkl')
    parser.add_argument('-c', '--chkpt', help='path to save checkpoint model to (.pth)', default='chkpt.pth')
    parser.add_argument('-b', '--best', help='path to save best model to (.pth)', default='best.pth')
    args = parser.parse_args()

    with open(args.train, 'rb') as f:
        training_data = pickle.load(f)

    random.shuffle(training_data)
    split = int(len(training_data) * train_ratio)
    train_set = training_data[:split]
    val_set = training_data[split:]
    print(len(train_set), len(val_set))

    gpu_exists = torch.cuda.device_count() > 0
    print(f'GPU: {int(gpu_exists)}')

    chkpt_path = args.chkpt
    best_path = args.best

    train()
