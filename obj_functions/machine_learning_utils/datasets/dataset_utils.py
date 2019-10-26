import torch
import numpy as np
from obj_functions.machine_learning_utils.datasets.cifar import get_cifar
from obj_functions.machine_learning_utils.datasets.svhn import get_svhn
from obj_functions.machine_learning_utils.datasets.imagenet import get_imagenet
from torch.utils.data.dataset import Subset


def get_dataset(dataset_name,
                n_cls=10,  # The number of class in training and testing
                image_size=None,  # pixel size
                data_frac=None,  # How much percentages of training we use in an experiment. [0, 1]
                biased_cls=None  # len(biased_cls) must be n_cls. Each element represents the percentages.
                ):

    train_raw_dataset, test_raw_dataset, raw_n_cls = get_raw_dataset(dataset_name, image_size, n_cls)

    n_cls = None if n_cls == raw_n_cls else n_cls
    train_dataset, test_dataset = process_raw_dataset(train_raw_dataset,
                                                      test_raw_dataset,
                                                      raw_n_cls,
                                                      dataset_name,
                                                      n_cls,
                                                      data_frac,
                                                      biased_cls)

    return train_dataset, test_dataset


def get_data(train_dataset, test_dataset, batch_size):
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_data, test_data


def get_raw_dataset(dataset_name, image_size=None, n_cls=10):
    if dataset_name.upper() == "CIFAR":
        if 2 <= n_cls <= 10:
            nc = 10
        elif 11 <= n_cls <= 100:
            nc = 100
        else:
            raise ValueError("n_cls must be between 2 and 100.")

        return get_cifar(nc) if image_size is None else get_cifar(nc, image_size)
    elif dataset_name.upper() == "SVHN":
        return get_svhn() if image_size is None else get_svhn(image_size)
    elif dataset_name.upper() == "IMAGENET":
        return get_imagenet() if image_size is None else get_imagenet(image_size)


def process_raw_dataset(train_raw_dataset,
                        test_raw_dataset,
                        raw_n_cls,
                        dataset_name,
                        n_cls=None,
                        data_frac=None,
                        biased_cls=None):
    """
    Parameters
    ----------
    train_raw_dataset
        The dataset from DataLoader
    test_raw_dataset
        The dataset from DataLoader
    raw_n_cls: int
        The number of classes the raw dataset has.
    dataset_name: str
        The name of dataset i.e. "cifar", "svhn", "imagenet"
    n_cls: int
        The number of classes you want the learning model to solve
    data_frac: float
        How many proportions the learning model uses to train itself. (0. to 1.)
    biased_cls: list of float (n_cls, )
        The index corresponds to the index of classes.
        How many data to use in training. Each element of the list must be 0. to 1.

    Returns
    -------
    train_dataset, test_dataset
    """

    if n_cls is None and data_frac is None and biased_cls is None:
        return train_raw_dataset, test_raw_dataset
    else:
        print("Processing raw dataset...")
        if dataset_name.upper() == "CIFAR":
            train_labels = np.array([train_raw_dataset.targets, list(range(len(train_raw_dataset)))])
            test_labels = np.array([test_raw_dataset.targets, list(range(len(test_raw_dataset)))])
        elif dataset_name.upper() == "SVHN":
            train_labels = np.array([train_raw_dataset.labels, list(range(len(train_raw_dataset)))])
            test_labels = np.array([test_raw_dataset.labels, list(range(len(test_raw_dataset)))])
        print("Start processing...")
        print("")

        if n_cls is not None:
            print("The number of classes: {} -> {}".format(raw_n_cls, n_cls))
            train_labels, test_labels = get_small_class(train_labels, test_labels, n_cls)
        if data_frac is not None:
            n_subtrain = int(np.ceil(len(train_labels[0]) * data_frac))
            print("Subsampling: {} images".format(n_subtrain))
            train_labels = np.array([tl[:n_subtrain] for tl in train_labels])
        if biased_cls is not None:
            print("Biased labels")
            train_labels = get_biased_class(train_labels, biased_cls, n_cls, raw_n_cls)

        return Subset(train_raw_dataset, train_labels[1]), Subset(test_raw_dataset, test_labels[1])


def get_small_class(train_labels, test_labels, n_cls):
    """
    Parameters
    ----------
    train_labels: list of int
        Each element is the answer of the corresponding images.
    test_labels: list of int
        Same as above.

    Returns
    -------
    (train_idxs, test_idxs): tuple of list of int (2,:)
        Each list has the index of images that will be used in an experiment.
    """

    train_indexes, test_indexes = [], []

    for idx, label in enumerate(train_labels[0]):
        if label < n_cls:
            train_indexes.append(idx)
    for idx, label in enumerate(test_labels[0]):
        if label < n_cls:
            test_indexes.append(idx)

    train_indexes, test_indexes = map(np.asarray, [train_indexes, test_indexes])

    return np.array([tl[train_indexes] for tl in train_labels]), np.array([tl[test_indexes] for tl in test_labels])


def get_biased_class(train_labels, biased_cls, n_cls, raw_n_cls):
    if len(biased_cls) != n_cls and len(biased_cls) != raw_n_cls:
        raise ValueError("The length of biased_cls must be n_cls(={}) or {}, but {} was given.".format(n_cls, raw_n_cls, len(biased_cls)))
    n_cls = raw_n_cls if n_cls is None else n_cls

    labels_to_idx = [train_labels[1][np.where(train_labels[0] == n)[0]] for n in range(len(n_cls))]
    labels_to_idx = [idx[:int(len(idx) * biased_cls[n])] for n, idx in enumerate(labels_to_idx)]

    return_idx = []
    for idx in labels_to_idx:
        return_idx += idx

    return_idx = np.array(return_idx)

    return np.array([np.sort(return_idx), list(len(return_idx))])
