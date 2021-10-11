import json
import os
import numpy as np
from argparse import ArgumentParser
from logging import Logger
from typing import Any, Callable, Dict

import ConfigSpace as CS

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from cnn.cnn import CNN
from cnn.hyperparameters import BudgetConfig, Hyperparameters
from util.utils import extract_hyperparameter, get_config_space, get_logger, ParameterSettings


def train(epochs: int, model: CNN, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module,
          logger: Logger, train_loader: DataLoader, val_loader: DataLoader) -> float:
    """
    Training routine

    Args:
        epochs (int): The number of epochs to train
        model (CNN): The target for the optimization
        optimizer (torch.optim.Optimizer): The optimizer for the neural network
        loss_fn (torch.nn.Module): The loss function for the training
        train_loader (DataLoader): The dataloader for the training dataset
        val_loader (DataLoader): The dataloader for the validation dataset

    Returns:
        acc (float):
            The validation accuracy at the end
    """
    best_val_acc, best_val_loss = 0.0, np.inf
    for epoch in range(epochs):
        logger.info('Epoch [{}/{}]'.format(epoch + 1, epochs))
        train_acc, train_loss = model.train_fn(optimizer, loss_fn, train_loader)
        val_acc, val_loss = model.eval_fn(loss_fn, val_loader)
        logger.info('Train Acc.: {:.4f}, Train loss: {:.4f}, Val. Acc.: {:.4f}, Val. loss: {:.4f}'.format(
            train_acc, train_loss, val_acc, val_loss
        ))

        best_val_acc = max(best_val_acc, val_acc)
        best_val_loss = min(best_val_loss, val_loss)

    return best_val_acc


def get_objective_func(
    logger: Logger,
    searching_space: Dict[str, ParameterSettings],
    config_space: CS.ConfigurationSpace,
    hp_module_path: str = 'cnn',
    data_dir: str = f'{os.environ["HOME"]}/research/micro17flower/'
) -> Callable:
    """
    Args:
        searching_space (Dict[str, ParameterSettings]):
            Searching space of a provided function (i.e. json file).
        config_space (CS.ConfigurationSpace):
            Searching space by config space
        data_dir (str): The directory path of the dataset

    Returns:
        objective_func (Callable):
            The objective func that returns the val acc given a configuration and budget
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        # transforms.Resize(),
        transforms.ToTensor()
    ])
    train_data = ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_data = ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    def objective_func(config: Dict[str, Any], budget: Dict[str, Any] = {}) -> float:
        """
        Args:
            config (Dict[str, Any]): The hyperparameter configuration for the CNN and the training
            budget (Dict[str, Any]): The variables that control the budget size

        Returns:
            acc (float): The validation accuracy at the end
        """
        config = extract_hyperparameter(config, config_space, searching_space, hp_module_path)
        _config, _budget = Hyperparameters(**config), BudgetConfig(**budget)
        batch_size, learning_rate = _config.batch_size, _config.learning_rate
        input_shape = (3, _budget.input_size, _budget.input_size)
        epochs, num_classes = _budget.epochs, _budget.num_classes
        opt, loss_fn = _config.optimizer.value, _config.loss_fn.value().to(device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

        model = CNN(_config, input_shape=input_shape, num_classes=num_classes).to(device)
        optimizer = opt(model.parameters(), lr=learning_rate)

        val_acc = train(epochs=epochs, model=model, optimizer=optimizer, loss_fn=loss_fn, logger=logger,
                        train_loader=train_loader, val_loader=val_loader)

        logger.info('Val Acc.: {:.2f}%'.format(val_acc * 100))

        return 1.0 - val_acc

    return objective_func


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--filename', type=str, default='example')
    parser.add_argument('--saved_config', type=str, default=None)
    args = parser.parse_args()

    # Logger setting
    logger = get_logger(file_name=args.filename, logger_name=args.filename)

    js = open('cnn/params.json')
    searching_space: Dict[str, ParameterSettings] = json.load(js)
    config_space = get_config_space(searching_space, hp_module_path='cnn')
    objective_func = get_objective_func(
        logger=logger,
        searching_space=searching_space,
        config_space=config_space
    )

    results = objective_func(
        config=Hyperparameters().__dict__,
        budget=BudgetConfig().__dict__
    )
