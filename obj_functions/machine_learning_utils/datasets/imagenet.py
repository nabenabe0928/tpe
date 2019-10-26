import torch
from torchvision import datasets, transforms


# https://github.com/Coderx7/SimpleNet_Pytorch/issues/3


def get_imagenet(batch_size):
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])

    transform_train = transforms.Compose([transforms.Pad(4, padding_mode='reflect'),
                                          transforms.RandomCrop(32),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = datasets.SVHN(root="svhn",
                                  split="train",
                                  download=True,
                                  transform=transform_train)
    test_dataset = datasets.SVHN(root="svhn",
                                 split="test",
                                 download=True,
                                 transform=transform_test)

    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_data, test_data
