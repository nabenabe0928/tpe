import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms


"""
dataset: 5-d array
1d: the index of image (default: < 50000 (cifar), 73257 (svhn))
2d: pixel information (0) or label (1)
3d: the index of RGB (0, 1, 2)
4d: the index of a row vector of pixel information (default: < 32)
5d: the index of the i-th row vector (default: < 32)
"""


transform_train = transforms.Compose([transforms.ToTensor()])
transform_test = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.CIFAR10(root="cifar",
                                 train=True,
                                 download=True,
                                 transform=transform_train)
test_dataset = datasets.CIFAR10(root="cifar",
                                train=False,
                                download=False,
                                transform=transform_test)


def see_cifar(dataset, idx=0):
    classes = dataset.dataset.classes[:]

    npimg = dataset[idx][0].numpy()
    print(classes[dataset[idx][1]])
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def see_svhn(dataset, idx=0):
    npimg = dataset[idx][0].numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
