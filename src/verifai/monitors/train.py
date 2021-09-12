from dataloader import MonitorLearningDataset
from net import Net

from torch.utils.data import DataLoader

import argparse


window_size = 5
characteristics = 10
learning_rate = 1e-4
log_batch = 100

batch_size = 5
n_workers = 8


def train():
    model = Net(window_size, characteristics)
    if gpu_exists:
        model = model.cuda()

    traces_val = MonitorLearningDataset(args.val)
    traces_train = MonitorLearningDataset(args.train)
    validation_set = DatasetLoader(traces_val,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=n_workers)
    train_set = DatasetLoader(traces_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=n_workers)
    traces_dataset = { 'validation': validation_set, 'train': train_set }

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    max_loss = 1e20
    for epoch in range(args.epochs):
        for phase in ['train', 'validation']:
            cumu_loss = 0
            print(f'{phase} at epoch no. {epoch}\n', '-' * 20)
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

                if i % log_batch == 0:
                    print(f'batch {i}: loss = {loss}')

            save_model = model.module if torch.cuda.device_count() > 1 else model
            torch.save(save_model.state_dict(), chkpt_path)
            if phase == 'validation':
                if cumu_loss / len(traces_dataset['validation']) < max_loss:
                    max_loss = cumu_loss
                    torch.save(save_model.state_dict(), best_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_args('-e', '--epochs', help='Number of Epochs', type=int, default=1000)
    parser.add_args('-t', '--train', help='.pkl file containing the training traces', default='data/train.pkl')
    parser.add_args('-v', '--val', help='.pkl file containing the validation traces', default='data/val.pkl')
    parser.add_args('-c', '--chkpt', help='path to save checkpoint model to (.pth)', default='chkpt.pth')
    parser.add_args('-b', '--best', help='path to save best model to (.pth)', default='best.pth')
    args = parser.parse_args()

    gpu_exists = torch.cuda.device_count() > 0

    chkpt_path = args.chkpt
    best_path = args.bes

    train()
