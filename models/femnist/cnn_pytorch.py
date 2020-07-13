#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 7/12/20 3:11 PM 2020

@author: Anirban Das
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_pytorch import Model
import numpy as np


IMAGE_SIZE = 28


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc1 = nn.Linear(in_features=(7*7*64), out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, inp):
        inp = inp.reshape(-1, 1, 28, 28)
        inp = F.relu(self.pool1(self.conv1(inp)))
        inp = F.relu(self.pool2(self.conv2(inp)))
        inp = inp.view(-1, 7 * 7 * 64)
        inp = F.relu(self.fc1(inp))
        inp = self.fc2(inp)
        return inp

class Net3(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, num_classes),
            #nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = self.conv_layers(x)
        x = x.view(-1, 320)
        x = self.fc_layers(x)
        return x

class Net2(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #output = F.log_softmax(x, dim=1)
        return x
"""
torch.Size([1, 32, 14, 14])
torch.Size([1, 64, 7, 7])
torch.Size([1, 3136])
torch.Size([1, 2048])
===============================================================
           Kernel Shape     Output Shape     Params  Mult-Adds
Layer                                                         
0_conv1   [1, 32, 5, 5]  [1, 32, 28, 28]      832.0     627.2k
1_pool1               -  [1, 32, 14, 14]          -          -
2_conv2  [32, 64, 5, 5]  [1, 64, 14, 14]    51.264k   10.0352M
3_pool2               -    [1, 64, 7, 7]          -          -
4_fc1      [3136, 2048]        [1, 2048]  6.424576M  6.422528M
5_fc2        [2048, 10]          [1, 10]     20.49k     20.48k
---------------------------------------------------------------
                          Totals
Total params           6.497162M
Trainable params       6.497162M
Non-trainable params         0.0
Mult-Adds             17.105408M
===============================================================

"""


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes):
        self.num_classes = num_classes
        print(self, seed)
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self, lr, momentum=0):
        """Model function for CNN."""
        net = Net(self.num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        return net, criterion, optimizer

    def process_x(self, raw_x_batch):
        return torch.tensor(raw_x_batch)

    def process_y(self, raw_y_batch):
        return torch.tensor(raw_y_batch)
