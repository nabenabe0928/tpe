import math
import torch.nn as nn
import torch.nn.functional as F
import typing 
from typing import NamedTuple


class HyperParameters(
    NamedTuple("_HyperParameters",
               [("batch_size", int),
                ("lr", float),
                ("momentum", float),
                ("weight_decay", float),
                ("ch1", int),
                ("ch2", int),
                ("ch3", int),
                ("ch4", int),
                ("drop_rate", float)])):
    pass


def get_hyperparameters(hp_parser):
    type_hints = typing.get_type_hints(HyperParameters)
    var_names = list(type_hints.keys())
    hp = {var_name: getattr(hp_parser, var_name) for var_name in var_names}

    return HyperParameters(**hp)


class CNN(nn.Module):
    def __init__(self, hp_parser):
        super(CNN, self).__init__()
        self.hyperparameters = get_hyperparameters(hp_parser)
        self.batch_size = self.hyperparameters.batch_size
        self.lr = self.hyperparameters.lr
        self.momentum = self.hyperparameters.momentum
        self.weight_decay = self.hyperparameters.weight_decay
        self.drop_rate = self.hyperparameters.drop_rate
        self.ch1 = self.hyperparameters.ch1
        self.ch2 = self.hyperparameters.ch2
        self.ch3 = self.hyperparameters.ch3
        self.ch4 = self.hyperparameters.ch4
        self.nesterov = False
        self.epochs = 160
        self.lr_decay = 1
        self.lr_step = [1]
        self.c1 = nn.Conv2d(3, self.ch1, 5, padding=2)
        self.c2 = nn.Conv2d(self.ch1, self.ch2, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(self.ch2)
        self.c3 = nn.Conv2d(self.ch2, self.ch3, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(self.ch3)
        self.full_conn1 = nn.Linear(self.ch3 * 3 ** 2, self.ch4)
        self.full_conn2 = nn.Linear(self.ch4, 100)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        h = self.c1(x)
        h = F.relu(F.max_pool2d(h, 3, stride=2))
        h = self.c2(h)
        h = F.relu(h)
        h = self.bn2(F.avg_pool2d(h, 3, stride=2))
        h = self.c3(h)
        h = F.relu(h)
        h = self.bn3(F.avg_pool2d(h, 3, stride=2))

        h = h.view(h.size(0), -1)
        h = self.full_conn1(h)
        h = F.dropout2d(h, p=self.drop_rate, training=self.training)
        h = self.full_conn2(h)
        return F.log_softmax(h, dim=1)
