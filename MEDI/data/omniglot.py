import torch
import torchvision
from torchvision import transforms
import numpy as np

def My_Omniglot(split='train'):
    binary_flip = transforms.Lambda(lambda x: 1 - x)
    normalize = transforms.Normalize((0.086,), (0.235,))
    trainset = torchvision.datasets.Omniglot(
        root='data', download=True, background=True,
        transform=transforms.Compose(
           [#transforms.RandomResizedCrop(32, (0.85, 1.)),
            transforms.Resize(32),
            transforms.ToTensor(),
            binary_flip,
            normalize]
        ))
    testset = torchvision.datasets.Omniglot(
        root='data', download=True, background=False,
        transform=transforms.Compose(
          [transforms.Resize(32),
           transforms.ToTensor(),
           binary_flip,
           normalize]
        ))
    if split == 'train':
        return trainset
    else:
        return testset
    