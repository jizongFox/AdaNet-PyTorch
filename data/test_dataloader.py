from unittest import TestCase
from torch.utils.data import DataLoader
from deepclustering.augment.tensor_augment import RandomHorizontalFlip, RandomCrop
from deepclustering.augment.augment import ToTensor
from deepclustering.augment import transforms
import torch

try:
    from .cifar_dataloader import CIFAR10
except ImportError:
    from cifar_dataloader import CIFAR10


class TestCIFAR_Dataset(TestCase):
    def test_init(self):
        img_transform = transforms.Compose([
            RandomHorizontalFlip(),
            RandomCrop((28, 28)),
            ToTensor()
        ])
        labeled_set = CIFAR10('label', img_transform=img_transform)
        print(labeled_set[0][0].shape)
        print(labeled_set[0][1])

    def test_dataloader(self):
        img_transform = transforms.Compose([
            RandomHorizontalFlip(),
            RandomCrop((28, 28)),
            ToTensor()
        ])
        labeled_set = CIFAR10('label', img_transform=img_transform)
        dataloader = DataLoader(labeled_set, batch_size=4, shuffle=True)
        iter(dataloader).__next__()[0].shape == torch.Size([4, 3, 28, 28])
