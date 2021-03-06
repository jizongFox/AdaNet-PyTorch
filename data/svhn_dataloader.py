__all__ = [
    "SVHN",
    "SVHNSemiSupervisedDatasetInterface",
    "default_svhn_transformation",
    "default_svhn_aug_transformation",
]
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch
from deepclustering.augment.pil_augment import ToTensor
from deepclustering.augment.tensor_augment import RandomCrop, Compose
from deepclustering.dataset.classification.semi_helper import SemiDatasetInterface
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class SVHN(Dataset):
    data_splits = {
        "label": "labeled_train.npz",
        "unlabel": "unlabeled_train.npz",
        "test": "test.npz",
        "label_val": "labeled_train_valid.npz",
        "unlabel_val": "unlabeled_train_valid.npz",
        "test_val": "test_valid.npz",
    }

    def __init__(
        self,
        split: str,
        img_transform: Callable = None,
        target_transform: Callable = None,
        seed: int = 1,
    ) -> None:
        """
        img_transform should be for PyTorch Tensor, not PIL images
        :param split:
        :param img_transform: for Tensor
        :param target_transform: for Tensor
        :param seed:
        """
        super().__init__()
        assert split in (
            self.data_splits
        ), f'arg `split` should be in \
        {", ".join(self.data_splits.keys())}, given {split}.'
        data = np.load(
            str(
                Path(__file__).parent
                / "svhnss"
                / f"seed{seed}"
                / self.data_splits[split]
            )
        )
        self.images: np.ndarray = data["images"]
        self.images = self.images.reshape((self.images.shape[0], 3, 32, 32))
        self.target = data["labels"]
        assert self.images.shape[0] == len(self.target)
        self.img_transform = img_transform
        self.target_transform = target_transform

    def __getitem__(self, index: int):
        img = torch.from_numpy(self.images[index]).float()
        target = torch.from_numpy(np.array(self.target[index])).long()
        if self.img_transform:
            img = self.img_transform(img.unsqueeze(0)).squeeze()
        if self.target_transform:
            target = self.target_transform(target)
        assert img.dtype == torch.float
        assert target.dtype == torch.long

        return img, target

    def __len__(self) -> int:
        return len(self.images)


class SVHNSemiSupervisedDatasetInterface(SemiDatasetInterface):
    def __init__(
        self,
        is_validation: bool = False,
        tra_img_transformation: Callable = None,
        val_img_transformation: Callable = None,
        target_transformation: Callable = None,
        verbose: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            SVHN, "", 1000, None, target_transformation, verbose, *args, **kwargs
        )
        self.is_validation = is_validation
        self.tra_img_transformation = tra_img_transformation
        self.val_img_transformation = val_img_transformation

    def SemiSupervisedDataLoaders(
        self,
        batch_size=4,
        shuffle=True,
        drop_last=False,
        num_workers=1,
        *args,
        **kwargs,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        _split = "_val" if self.is_validation else ""
        labeled_set = SVHN(
            split=f"label{_split}",
            img_transform=self.tra_img_transformation,
            target_transform=self.target_transform,
        )
        unlabel_set = SVHN(
            split=f"unlabel{_split}",
            img_transform=self.tra_img_transformation,
            target_transform=self.target_transform,
        )
        test_set = SVHN(
            split=f"test{_split}",
            img_transform=self.val_img_transformation,
            target_transform=self.target_transform,
        )
        label_loader = DataLoader(
            labeled_set,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
        unlabel_loader = DataLoader(
            unlabel_set,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )
        return label_loader, unlabel_loader, test_loader

    def _init_train_and_test_test(
        self, transform, target_transform, *args, **kwargs
    ) -> Tuple[Dataset, Dataset]:
        pass


default_svhn_transformation = {
    "train": Compose([ToTensor()]),
    "val": Compose([ToTensor()]),
}
default_svhn_aug_transformation = {
    "train": Compose([RandomCrop((32, 32), padding=(2, 2, 2, 2)), ToTensor()]),
    "val": Compose([ToTensor()]),
}
