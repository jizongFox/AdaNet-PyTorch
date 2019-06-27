__all__ = ['get_dataloader']
from copy import deepcopy as dcp

from .cifar_dataloader import Cifar10SemiSupervisedDatasetInterface, default_cifar10_transformation, \
    default_cifar10_aug_transformation
from .svhn_dataloader import SVHNSemiSupervisedDatasetInterface, default_svhn_transformation, \
    default_svhn_aug_transformation


def get_dataloader(name: str = None, aug: bool = False, DataLoader_DICT: dict = {}):
    DataLoader_DICT = dcp(DataLoader_DICT)
    assert name in ('cifar10', 'svhn')
    print(f'data aug: {bool(aug)}.')
    if name == 'cifar10':

        SemiDatasetHandler = Cifar10SemiSupervisedDatasetInterface(
            tra_img_transformation=
            default_cifar10_aug_transformation["train"] if aug else
            default_cifar10_transformation['train'],
            val_img_transformation=
            default_cifar10_aug_transformation['val'] if aug else
            default_cifar10_transformation['val'],
            verbose=True
        )
    else:
        SemiDatasetHandler = SVHNSemiSupervisedDatasetInterface(
            tra_img_transformation=default_svhn_aug_transformation["train"] if aug else \
                default_svhn_transformation['train'],
            val_img_transformation=default_svhn_aug_transformation['val'] if aug else \
                default_svhn_transformation['val'],
            verbose=True
        )
    DataLoader_DICT.pop('name', None)
    label_loader, unlabel_loader, val_loader = SemiDatasetHandler.SemiSupervisedDataLoaders(**DataLoader_DICT)
    return label_loader, unlabel_loader, val_loader
