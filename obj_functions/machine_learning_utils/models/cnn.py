import torch.nn as nn
import torch.nn.functional as F
import math


class CNN(nn.Module):
    """
    Parameters
    ----------
    batch_size: int
        batch size of image dataset
    lr: float
        The learning rate of inner weight parameter of CNN.
    momentum: float
        momentum coefficient for Stochastic Gradient Descent (SGD)
    weight_decay: float
        the coefficients of a regularization term for cross entropy
    drop_rate: float
        The probability of dropout a weight of connections.
    ch1, ch2, ch3, ch4: int
        the number of kernel feature maps
    nesterov: bool
        Whether using nesterov or not in SGD.
    epochs: int
        The number of training throughout one learning process.
    lr_step: list of float
        When to decrease the learning rate.
        The learning rate will decline at lr_step[k] * epochs epoch.
    lr_decay: float
        How much make learning rate decline at epochs * lr_step[k]
    n_cls: int
        The number of classes on a given task.
    """

    def __init__(self,
                 batch_size=128,
                 lr=5.0e-2,
                 momentum=0.9,
                 weight_decay=5.0e-4,
                 ch1=32,
                 ch2=32,
                 ch3=32,
                 ch4=32,
                 drop_rate=0.5,
                 nesterov=False,
                 lr_decay=1.,
                 lr_step=[1.],
                 n_cls=100
                 ):
        super(CNN, self).__init__()

        # Hyperparameter Configuration for CNN.
        self.batch_size = int(batch_size)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.drop_rate = drop_rate
        self.ch1 = int(ch1)
        self.ch2 = int(ch2)
        self.ch3 = int(ch3)
        self.ch4 = int(ch4)
        self.nesterov = nesterov
        self.lr_decay = lr_decay
        self.lr_step = lr_step
        self.epochs = 160

        # Architecture of CNN.
        self.c1 = nn.Conv2d(3, self.ch1, 5, padding=2)
        self.c2 = nn.Conv2d(self.ch1, self.ch2, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(self.ch2)
        self.c3 = nn.Conv2d(self.ch2, self.ch3, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(self.ch3)
        self.full_conn1 = nn.Linear(self.ch3 * 3 ** 2, self.ch4)
        self.full_conn2 = nn.Linear(self.ch4, n_cls)
        self.init_inner_params()

    def init_inner_params(self):
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
