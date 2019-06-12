#
#
#   The main program for the training of AdaNet.
#
#

from deepclustering.augment.augment import ToTensor
from deepclustering.augment.tensor_augment import RandomHorizontalFlip, RandomCrop, Compose
from deepclustering.manager import ConfigManger
from deepclustering.model import Model

from arch import _register_arch
from data.cifar_dataloader import Cifar10SemiSupervisedDatasetInterface
from scheduler import CustomScheduler
from trainer import AdaNetTrainer

_ = _register_arch  # to enable the registration
DEFAULT_CONFIG_PATH = 'config.yaml'
config = ConfigManger(DEFAULT_CONFIG_PATH, verbose=True, integrality_check=False).merged_config
model = Model(config.get('Arch'), config.get('Optim'), config.get('Scheduler'))
# print(model)
img_transform = Compose([
    RandomHorizontalFlip(),
    RandomCrop((32, 32), padding=(2, 2)),
    ToTensor()
])
val_img_transform = Compose([
    ToTensor()
])
SemiDatasetHandler = Cifar10SemiSupervisedDatasetInterface(
    img_transformation=img_transform,
    target_transformation=None,
    verbose=True
)
label_loader, unlabel_loader, val_loader = SemiDatasetHandler.SemiSupervisedDataLoaders(**config.get('DataLoader'))
val_loader.dataset.img_transform: Compose = val_img_transform
scheduler = CustomScheduler(max_epoch=config['Trainer']['max_epoch'])
trainer = AdaNetTrainer(
    model=model,
    labeled_loader=label_loader,
    unlabeled_loader=unlabel_loader,
    val_loader=val_loader,
    config=config,
    grl_scheduler=scheduler,
    **config['Trainer']
)
trainer.start_training()
trainer.clean_up()
