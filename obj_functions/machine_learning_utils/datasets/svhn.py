from torchvision import datasets, transforms
import warnings


# https://github.com/Coderx7/SimpleNet_Pytorch/issues/3


def get_svhn(image_size=32):
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
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

    train_dataset = datasets.SVHN(root="svhn",
                                  split="train",
                                  download=True,
                                  transform=transform_train)
    test_dataset = datasets.SVHN(root="svhn",
                                 split="test",
                                 download=True,
                                 transform=transform_test)

    return train_dataset, test_dataset, 10
