from tqdm import tqdm
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cnn.metrics import accuracy, AvgrageMeter
from cnn.hyperparameters import Hyperparameters


def conv_block(in_channels: int, out_channels: int,
               kernel_size: int = 3, stride: int = 1, padding: int = 1,
               batch_normalization: bool = True) -> nn.Sequential:
    """
    The function to return the convolution block

    Args:
        in_channels (int): The number of channels at the input
        out_channels (int): The number of channels at the output
        kernel_size (int): The size of kernel filters
        stride (int): The stride of the convolution
        padding (int): The padding size for the convolution
        batch_normalization (bool): Whether we apply batch normalization

    Returns:
        (nn.Sequential): The convolution block
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(out_channels) if batch_normalization else nn.Identity(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


def get_conv_layers(config: Hyperparameters, in_channels: int, out_channels_list: List[int]) -> nn.Sequential:
    """
    The function to return the convolution layers.

    Args:
        config (ModelConfig): The hyperparameter configuration for this model
        in_channels (int): The number of channels at the input
        out_channels_list (List[int]): The number of channels at each layer

    Returns:
        (nn.Sequential): The convolution layers
    """

    conv_layers = []
    kernel_size = config.kernel_size

    for i in range(config.n_conv_layers):
        out_channels = out_channels_list[i]
        padding = (kernel_size - 1) // 2
        conv_layers.append(
            conv_block(in_channels, out_channels, kernel_size=kernel_size,
                       padding=padding, batch_normalization=config.batch_normalization)
        )
        in_channels = out_channels

    return nn.Sequential(
        *conv_layers,
        nn.AdaptiveAvgPool2d(1) if config.global_avg_pooling else nn.Identity()
    )


def get_fc_layers(config: Hyperparameters, in_features: int, out_features_list: List[int],
                  num_classes: int) -> nn.Sequential:
    """
    The function to return the fully-connected layers.

    Args:
        config (ModelConfig): The hyperparameter configuration for this model
        in_features (int): The size of input features
        out_features_list (List[int]): The size of output at each layer
        num_classes (int): The number of classes in the dataset

    Returns:
        (nn.Sequential): The fully-connected layers
    """
    fc_layers = []
    for i in range(config.n_fc_layers):
        out_features = out_features_list[i]
        fc_layers.append(
            nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Dropout(p=config.dropout_rate),
            )
        )
        in_features = out_features

    fc_layers.append(nn.Linear(in_features, num_classes))

    return nn.Sequential(*fc_layers)


class CNN(nn.Module):
    def __init__(self, config: Hyperparameters, input_shape: Tuple[int, ...] = (3, 16, 16),
                 num_classes: int = 17):
        super(CNN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        in_channels = input_shape[0]
        out_channels_list = [getattr(config, f'n_channels_conv{i}') for i in range(3)]
        out_features_list = [getattr(config, f'n_channels_fc{i}') for i in range(3)]
        self.num_classes = num_classes

        self.conv_layers = get_conv_layers(
            config=config,
            in_channels=in_channels,
            out_channels_list=out_channels_list
        ).to(self.device)

        in_features = self._get_conv_output(input_shape)
        self.fc_layers = get_fc_layers(
            config=config,
            in_features=in_features,
            out_features_list=out_features_list,
            num_classes=num_classes
        ).to(self.device)

    def _get_conv_output(self, shape: Tuple[int, ...]) -> int:
        """ The method to compute the output size of conv layers """
        in_features = torch.rand(1, *shape).to(self.device)
        out_features = self.conv_layers(in_features)
        return out_features.data.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def network_size(self) -> float:
        return sum(p.numel() for p in self.parameters())

    def train_fn(self, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module,
                 loader: DataLoader) -> Tuple[float, float]:
        """
        The method to train the model (self) on a given dataset

        Args:
            optimizer (torch.optim.Optimizer): The gradient descent method
            criterion (torch.nn.Module): The loss function for the training
            loader (DataLoader): Training dataset

        Returns:
            accs.avg, losses.avg (Tuple[float, float]):
                The average of accuracy and that of the loss value
                over the provided dataloader
        """

        accs, losses = AvgrageMeter(), AvgrageMeter()
        self.train()

        t = tqdm(loader)
        for images, labels in t:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            logits = self(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            acc = accuracy(logits, labels)

            losses.update(loss.item(), images.size(0))
            accs.update(acc, images.size(0))

            t.set_description('(=> Training) Loss, Acc.: {:.4f}, {:.4f}'.format(losses.avg, accs.avg))

        return accs.avg, losses.avg

    def eval_fn(self, criterion: torch.nn.Module, loader: DataLoader) -> Tuple[float, float]:
        """
        The method to evaluate the model (self) on a given dataset

        Args:
            criterion (torch.nn.Module): The loss function for the training
            loader (DataLoader): Dataloader to be evaluated

        Returns:
            accs.avg, losses.avg (Tuple[float, float]):
                The average of accuracy and that of the loss value
                over the provided dataloader
        """

        accs, losses = AvgrageMeter(), AvgrageMeter()
        self.eval()

        t = tqdm(loader)
        with torch.no_grad():  # no gradient needed
            for images, labels in t:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self(images)
                loss = criterion(logits, labels)
                acc = accuracy(logits, labels)

                accs.update(acc, images.size(0))
                losses.update(loss.item(), images.size(0))

                t.set_description('(=> Validation) Loss, Acc.: {:.4f}, {:.4f}'.format(losses.avg, accs.avg))

        return accs.avg, losses.avg
