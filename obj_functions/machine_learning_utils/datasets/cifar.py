from torchvision import datasets, transforms


# https://gluon-cv.mxnet.io/build/examples_classification/demo_cifar10.html
# https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151


def get_cifar(n_cls=10, image_size=32):
    if n_cls == 10:
        return get_cifar10(image_size)
    elif n_cls == 100:
        return get_cifar100(image_size)
    else:
        raise ValueError("The number of class for CIFAR must be 2 to 100.")
        print("But, {} was given.".format(n_cls))


def get_cifar10(image_size=32):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2470, 0.2435, 0.2616])

    transform_train = transforms.Compose([transforms.Pad(4, padding_mode='reflect'),
                                          transforms.RandomCrop(32),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = datasets.CIFAR10(root="cifar",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
    test_dataset = datasets.CIFAR10(root="cifar",
                                    train=False,
                                    download=False,
                                    transform=transform_test)

    return train_dataset, test_dataset, 10


def get_cifar100(image_size=32):
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])

    transform_train = transforms.Compose([transforms.Pad(4, padding_mode='reflect'),
                                          transforms.RandomResizedCrop(image_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])
    transform_test = transforms.Compose([transforms.Resize(32),
                                         transforms.CenterCrop(image_size),
                                         transforms.ToTensor(),
                                         normalize])

    train_dataset = datasets.CIFAR100(root="cifar",
                                      train=True,
                                      download=True,
                                      transform=transform_train)
    test_dataset = datasets.CIFAR100(root="cifar",
                                     train=False,
                                     download=False,
                                     transform=transform_test)

    return train_dataset, test_dataset, 100
