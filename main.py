from deepclustering.manager import ConfigManger
from deepclustering.model import Model, to_Apex

from arch import _register_arch
from data import get_dataloader
from scheduler import CustomScheduler
from trainer import AdaNetTrainer, VAT_Trainer

_ = _register_arch  # to enable the network registration

DEFAULT_CONFIG_PATH = 'config.yaml'
config = ConfigManger(DEFAULT_CONFIG_PATH, verbose=True, integrality_check=False).config
model = Model(config.get('Arch'), config.get('Optim'), config.get('Scheduler'))
model = to_Apex(model, opt_level='O1')

label_loader, unlabel_loader, val_loader = get_dataloader(
    config['DataLoader'].get('name'),
    config['DataLoader'].get('aug', False),
    config.get('DataLoader'))
scheduler = CustomScheduler(max_epoch=config['Trainer']['max_epoch'])
assert config['Trainer'].get('name') in ('vat', 'ada')

Trainer = VAT_Trainer if config['Trainer']['name'] == 'vat' else AdaNetTrainer
trainer = Trainer(
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
