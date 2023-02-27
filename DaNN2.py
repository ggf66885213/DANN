# -*- coding: utf-8 -*-

import torch.nn as nn


class DaNN(nn.Module):
    def __init__(self, n_input=11, n_hidden=256, n_class=2):
        super(DaNN, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.layer_input2 = nn.Linear(n_hidden, 128)
        self.layer_hidden = nn.Linear(128, n_class)
        self.soft = nn.Softmax(dim=1)
    def forward(self, src, tar):
        x_src = self.layer_input(src)
        x_tar = self.layer_input(tar)
        x_src_mmd1 = self.relu(x_src)
        x_tar_mmd1 = self.relu(x_tar)
        x_src = self.layer_input2(x_src_mmd1)
        x_tar = self.layer_input2(x_tar_mmd1)
        x_src = self.dropout(x_src)
        x_tar = self.dropout(x_tar)
        x_src_mmd2 = self.relu(x_src)
        x_tar_mmd2 = self.relu(x_tar)
        y_src = self.layer_hidden(x_src_mmd2)
        y_src = self.soft(y_src)
        return y_src, x_src_mmd1, x_tar_mmd1
class DaNN1(nn.Module):
    def __init__(self, n_input=11, n_hidden=256, n_class=2):
        super(DaNN1, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.layer_input2 = nn.Linear(n_hidden, 128)
        self.layer_hidden = nn.Linear(128, n_class)
        self.soft = nn.Softmax(dim=1)
    def forward(self, src, tar):
        x_src = self.layer_input(src)
        x_tar = self.layer_input(tar)
        x_src_mmd1 = self.relu(x_src)
        x_tar_mmd1 = self.relu(x_tar)
        x_src = self.layer_input2(x_src_mmd1)
        x_tar = self.layer_input2(x_tar_mmd1)
        x_src = self.dropout(x_src)
        x_tar = self.dropout(x_tar)
        x_src_mmd2 = self.relu(x_src)
        x_tar_mmd2 = self.relu(x_tar)
        y_src = self.layer_hidden(x_src_mmd2)
        y_src = self.soft(y_src)
        return y_src, x_src_mmd1, x_tar_mmd1, x_src_mmd2, x_tar_mmd2
class DaNN2(nn.Module):
    def __init__(self, n_input=11, n_hidden=256, n_class=2):
        super(DaNN2, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.layer_hidden = nn.Linear(n_hidden, n_class)
        self.soft = nn.Softmax(dim=1)
    def forward(self, src, tar):
        x_src = self.layer_input(src)
        x_tar = self.layer_input(tar)
        x_src = self.dropout(x_src)
        x_tar = self.dropout(x_tar)
        x_src_mmd = self.relu(x_src)
        x_tar_mmd = self.relu(x_tar)
        y_src = self.layer_hidden(x_src_mmd)
        y_src = self.soft(y_src)
        return y_src, x_src_mmd, x_tar_mmd