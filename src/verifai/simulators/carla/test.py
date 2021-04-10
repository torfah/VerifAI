import sys
import torch
import argparse
import numpy as np
import torch.nn as nn

from torch.autograd import Variable
from torch.utils import data
from PIL import Image
import random
#import torchvision.transforms as transforms

from model import *
#import torchvision
from sklearn import metrics

def validate(args, model, val_loader):
    model.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            #images = Variable(images.cuda())
            #labels = Variable(labels.cuda().float())
            images = Variable(images)
            labels = Variable(labels.float())
            outputs = model(images)
            loss = mseloss(outputs, labels)
    return loss 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--load_path', nargs='?', type=str, default='/home/carla_challenge/Desktop/wayne/VerifAI/examples/carla/overtake_control/simpath/_out/saved_models',
                        help='Model path')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    args = parser.parse_args()

    # Setup Dataloader
    #img_transform = transforms.Compose([
    #    transforms.Resize((256, 256)),
    #    transforms.ToTensor(),
    #    normalize,
    #])

    test_data  = roadDataset('/home/carla_challenge/Desktop/wayne/VerifAI/examples/carla/overtake_control/simpath/_out','test')
    test_loader = data.DataLoader(test_data, batch_size=args.batch_size, num_workers=8, shuffle=False)

    print(len(test_loader))

    net = mobilenet_v2(True, True, 480) 

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net).cuda()

    net.load_state_dict(torch.load(args.load_path + args.dataset + '/' +
                                     args.arch + ".pth"))
    print("Model Loaded!")

    loss = validate(args, net, test_loader)
    print("loss on test set: %.4f" % ( loss ))
