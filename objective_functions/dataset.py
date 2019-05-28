import torch
import random
from torchvision import datasets
from torchvision import transforms
import numpy as np

def get_data(batch_size):
   
    normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                          std=[0.2471, 0.2435, 0.2616])

    transform_train = transforms.Compose([
                                transforms.Pad(4, padding_mode = 'reflect'),
                                transforms.RandomCrop(32),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), 
                                normalize])
    transform_test = transforms.Compose([
                                transforms.ToTensor(), 
                                normalize])

    train_dataset = datasets.CIFAR100(
                        root = "cifar",
                        train = True, 
                        download = True,
                        transform = transform_train
                        )
    test_dataset = datasets.CIFAR100(
                        root = "cifar",
                        train = False, 
                        download = False,
                        transform = transform_test
                        )

    train_data = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 1, pin_memory = True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 1, pin_memory = True)
    
    return train_data, test_data
