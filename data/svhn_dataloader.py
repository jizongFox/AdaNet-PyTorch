#
#
#   file description
#
#
__all__ = ['SVHNDatasetSemiSupervisedInterface', 'default_svhn_transform']
from pathlib import Path
from typing import Callable, Tuple

from deepclustering.dataset.classification.semi_helper import SemiDatasetInterface
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import SVHN


class SVHNDatasetSemiSupervisedInterface(SemiDatasetInterface):

    def __init__(self, labeled_sample_num: int = 1000, tra_img_transformation: Callable = None,
                 val_img_transformation: Callable = None,
                 target_transformation: Callable = None,
                 verbose: bool = True,
                 *args, **kwargs) -> None:
        super().__init__(SVHN, str(Path(__file__).parent / 'SVHN'), labeled_sample_num, None,
                         target_transformation, verbose,
                         *args, **kwargs)
        self.tra_img_transformation = tra_img_transformation
        self.val_img_transformation = val_img_transformation

    def _init_train_and_test_test(self, transform, target_transform, *args, **kwargs) -> Tuple[Dataset, Dataset]:
        train_set = SVHN(
            root=self.data_root,
            split='train',
            transform=self.tra_img_transformation,
            target_transform=self.target_transform,
            download=True
        )
        test_set = SVHN(
            root=self.data_root,
            split=f"test",
            transform=self.val_img_transformation,
            target_transform=self.target_transform,
            download=True
        )
        return train_set, test_set


default_svhn_transform = {
    'train': transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    ),
    'val': transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
}
