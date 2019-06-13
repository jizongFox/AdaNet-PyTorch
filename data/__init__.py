__all__ = ['get_dataloader']
from copy import deepcopy as dcp

from .cifar_dataloader import Cifar10SemiSupervisedDatasetInterface, default_cifar10_transformation
from .svhn_dataloader import SVHNDatasetSemiSupervisedInterface, default_svhn_transform


def get_dataloader(name: str = None, DataLoader_DICT: dict = {}):
    DataLoader_DICT = dcp(DataLoader_DICT)
    assert name in ('cifar10', 'svhn')
    if name == 'cifar10':
        SemiDatasetHandler = Cifar10SemiSupervisedDatasetInterface(
            tra_img_transformation=default_cifar10_transformation["train"],
            val_img_transformation=default_cifar10_transformation['val'],
            verbose=True
        )
    else:
        SemiDatasetHandler = SVHNDatasetSemiSupervisedInterface(
            tra_img_transformation=default_svhn_transform["train"],
            val_img_transformation=default_svhn_transform['val'],
            verbose=True
        )
    DataLoader_DICT.pop('name', None)
    label_loader, unlabel_loader, val_loader = SemiDatasetHandler.SemiSupervisedDataLoaders(**DataLoader_DICT)
    return label_loader, unlabel_loader, val_loader
