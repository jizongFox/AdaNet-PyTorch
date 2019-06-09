#
#
#   The main program for the training of AdaNet.
#
#
from pprint import pprint

from deepclustering.dataset.classification import Cifar10SemiSupervisedDatasetInterface, default_cifar10_img_transform
from deepclustering.manager import ConfigManger
from deepclustering.model import Model
from arch import _register_arch

from trainer import AdaNetTrainer

DEFAULT_CONFIG_PATH = 'config.yaml'
config = ConfigManger(DEFAULT_CONFIG_PATH, verbose=True, integrality_check=False).merged_config
pprint(config)
model = Model(config.get('Arch'), config.get('Optim'), config.get('Scheduler'))
# print(model)
SemiDatasetHandler = Cifar10SemiSupervisedDatasetInterface(
    data_root='/home/jizong/Workspace/deep-clustering-toolbox/.data',
    labeled_sample_num=1000,
    img_transformation=default_cifar10_img_transform['tf1'],
    target_transformation=None,
    verbose=True
)
label_loader, unlabel_loader, val_loader = SemiDatasetHandler.SemiSupervisedDataLoaders(**config.get('DataLoader'))
trainer = AdaNetTrainer(
    model=model,
    labeled_loader=label_loader,
    unlabeled_loader=unlabel_loader,
    val_loader=val_loader,
    config=config,
    **config['Trainer']
)
trainer.start_training()
