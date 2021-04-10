import torch
import argparse
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import os

from model import *
from dataset import roadDataset
from test import validate

def train(args):
    train_data = roadDataset('/home/carla_challenge/Desktop/wayne/VerifAI/examples/carla/overtake_control/simpath/_out','train')
    test_data  = roadDataset('/home/carla_challenge/Desktop/wayne/VerifAI/examples/carla/overtake_control/simpath/_out','test')
    train_loader = DataLoader(train_data, shuffle=True, num_workers=8, batch_size=args.batch)
    test_loader = DataLoader(test_data, shuffle=False, num_workers=8, batch_size=args.batch)
    net = mobilenet_v2(480) 
    best_loss = float('inf')
    if torch.cuda.device_count() >1:
        print("Using", torch.cuda.device_count(), "GPUs")
        net = nn.DataParallel(net).cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    mseloss = nn.MSELoss()
    if args.load == 1:
        net.load_state_dict(torch.load(args.save_dir +'/net.pth'))
        optimizer.load_state_dict(torch.load(args.save_dir  + '/opt.pth'))
        best_loss = validate(args, net, test_loader)
        print("Model loaded! best loss: {}".format(best_loss))
    
    for epoch in range(args.n_epoch):
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            #images = Variable(images.cuda())
            #labels = Variable(labels.cuda().float())
            print (i)
            images = Variable(images.float())
            labels = Variable(labels.float())

            optimizer.zero_grad()
             
            outputs = net(images)
            loss = mseloss(outputs, labels)

            loss.backward()
            optimizer.step()
        val_loss = validate(args, net, test_loader)
        if val_loss < best_loss:
            torch.save(net.state_dict(), args.save_dir + 'net'+str(epoch) + ".pth")
            torch.save(optimizer.state_dict(), args.save_dir  + 'opt'+str(epoch)+'.pth')
            best_mRecall = val_mRecall
            print(f'Epoch [{epoch + 1}/{args.n_epoch}][saved] Loss: {loss.data}')
        else:
            print(f'Epoch [{epoch + 1}/{args.n_epoch}][-----] Loss: {loss.data}')
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hyperparams")
    parser.add_argument('--n_epoch', nargs='?', type=int, default=2, help='# of the epochs')
    parser.add_argument('--save_dir', type=str, default="/home/carla_challenge/Desktop/wayne/VerifAI/examples/carla/overtake_control/simpath/_out/saved_models")
    parser.add_argument('--load', nargs='?', type=int)
    parser.add_argument('--batch', nargs='?', type=int, default=32, help='Batch Size')
    args = parser.parse_args()
    train(args)
