#
#
#   This is the trainer class to realize the AdaNet based on the corresponding paper
#
#
from pathlib import Path
from typing import List

import torch
from deepclustering import ModelMode
from deepclustering.loss import Entropy
from deepclustering.loss import KL_div
from deepclustering.meters import MeterInterface, AverageValueMeter, ConfusionMatrix
from deepclustering.model import Model
from deepclustering.trainer import _Trainer
from deepclustering.utils import DataIter, tqdm, tqdm_, class2one_hot, flatten_dict, nice_dict, simplex
from torch.distributions import Beta
from torch.utils.data import DataLoader

from scheduler import CustomScheduler
from utils import VATLoss

PROJECT_PATH = str(Path(__file__).parent)


class AdaNetTrainer(_Trainer):
    RUN_PATH = str(Path(PROJECT_PATH) / 'runs')
    ARCHIVE_PATH = str(Path(PROJECT_PATH) / 'archives')

    def __init__(self, model: Model,
                 labeled_loader: DataLoader,
                 unlabeled_loader: DataLoader,
                 val_loader: DataLoader,
                 max_epoch: int = 100,
                 grl_scheduler: CustomScheduler = None,
                 save_dir: str = 'adanet',
                 checkpoint_path: str = None,
                 device='cpu',
                 config: dict = None,
                 **kwargs) -> None:
        super().__init__(model, None, val_loader, max_epoch, save_dir, checkpoint_path, device, config,
                         **kwargs)
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.kl_criterion = KL_div()
        self.beta_distr: Beta = Beta(torch.tensor([1.0]), torch.tensor([1.0]))
        self.grl_scheduler = grl_scheduler
        self.grl_scheduler.epoch = self._start_epoch

    def __init_meters__(self) -> List[str]:
        METER_CONFIG = {'tra_reg_total': AverageValueMeter(),
                        'tra_sup_label': AverageValueMeter(),
                        'tra_sup_mixup': AverageValueMeter(),
                        'tra_cls': AverageValueMeter(),
                        'tra_conf': ConfusionMatrix(num_classes=10),
                        'val_conf': ConfusionMatrix(num_classes=10)}
        self.METERINTERFACE = MeterInterface(METER_CONFIG)
        return [
            'tra_reg_total_mean',
            'tra_sup_label_mean',
            'tra_sup_mixup_mean',
            'tra_cls_mean',
            'tra_conf_acc',
            'val_conf_acc'
        ]

    def start_training(self):
        for epoch in range(self._start_epoch, self.max_epoch):
            self._train_loop(
                labeled_loader=self.labeled_loader,
                unlabeled_loader=self.unlabeled_loader,
                epoch=epoch,
            )
            with torch.no_grad():
                current_score = self._eval_loop(self.val_loader, epoch)
            self.METERINTERFACE.step()
            self.model.schedulerStep()
            self.grl_scheduler.step()
            # save meters and checkpoints
            for k, v in self.METERINTERFACE.aggregated_meter_dict.items():
                v.summary().to_csv(self.save_dir / f'meters/{k}.csv')
            self.METERINTERFACE.summary().to_csv(self.save_dir / self.wholemeter_filename)
            self.writer.add_scalars('Scalars', self.METERINTERFACE.summary().iloc[-1].to_dict(), global_step=epoch)
            self.drawer.call_draw()
            self.model.torchnet.lambd = self.grl_scheduler.value
            self.save_checkpoint(self.state_dict, epoch, current_score)

    def _train_loop(
            self,
            labeled_loader: DataLoader = None,
            unlabeled_loader: DataLoader = None,
            epoch: int = 0,
            mode=ModelMode.TRAIN,
            *args, **kwargs
    ):
        super(AdaNetTrainer, self)._train_loop(*args, **kwargs)  # warnings
        self.model.set_mode(mode)
        assert self.model.training

        labeled_loader_ = DataIter(labeled_loader)
        unlabeled_loader_ = DataIter(unlabeled_loader)
        batch_num: tqdm = tqdm_(range(unlabeled_loader.__len__()))

        for _batch_num, ((label_img, label_gt), (unlabel_img, _), _) in enumerate(
                zip(labeled_loader_, unlabeled_loader_, batch_num)):
            label_img, label_gt, unlabel_img = label_img.to(self.device), \
                                               label_gt.to(self.device), unlabel_img.to(self.device)

            label_pred, _ = self.model(label_img)
            self.METERINTERFACE.tra_conf.add(label_pred.max(1)[1], label_gt)
            sup_loss = self.kl_criterion(label_pred,
                                         class2one_hot(label_gt.unsqueeze(1).unsqueeze(1), 10).squeeze().float())
            self.METERINTERFACE.tra_sup_label.add(sup_loss.item())

            reg_loss = self._trainer_specific_loss(label_img, label_gt, unlabel_img)
            self.METERINTERFACE.tra_reg_total.add(reg_loss.item())
            self.model.zero_grad()
            (sup_loss + reg_loss).backward()
            self.model.step()
            report_dict = self._training_report_dict
            batch_num.set_postfix(report_dict)
        print(f'Validating epoch {epoch}: {nice_dict(report_dict)}')

    def _eval_loop(self, val_loader: DataLoader = None, epoch: int = 0, mode=ModelMode.EVAL, *args, **kwargs) -> float:
        self.model.set_mode(mode)
        assert not self.model.training
        val_loader_: tqdm = tqdm_(val_loader)

        for _batch_num, (img, label) in enumerate(val_loader_):
            img, label = img.to(self.device), label.to(self.device)
            pred, _ = self.model(img)
            self.METERINTERFACE.val_conf.add(pred.max(1)[1], label)
            report_dict = self._eval_report_dict
            val_loader_.set_postfix(report_dict)

        print(f'Validating epoch {epoch}: {nice_dict(report_dict)}')

        return self.METERINTERFACE.val_conf.summary()['acc']

    @property
    def _training_report_dict(self):
        return {'tra_sup_l': self.METERINTERFACE.tra_sup_label.summary()['mean'],
                'tra_sup_m': self.METERINTERFACE.tra_sup_mixup.summary()['mean'],
                'tra_cls': self.METERINTERFACE.tra_cls.summary()['mean'],
                'tra_acc': self.METERINTERFACE.tra_conf.summary()['acc']}

    @property
    def _eval_report_dict(self):
        return flatten_dict({'val': self.METERINTERFACE.val_conf.summary()}, sep='_')

    def _trainer_specific_loss(self, label_img, label_gt, unlab_img, *args, **kwargs):
        super(AdaNetTrainer, self)._trainer_specific_loss(*args, **kwargs)  # warning
        assert label_img.shape == unlab_img.shape, f"Shapes of labeled and unlabeled images should be the same," \
            f"given {label_img.shape} and {unlab_img.shape}."
        self.model.eval()
        with torch.no_grad():
            pseudo_label = self.model.torchnet(unlab_img)[0]
        self.model.train()
        mixup_img, mixup_label, mix_indice = self._mixup(label_img, label_gt, unlab_img, pseudo_label)

        pred, cls = self.model(mixup_img)
        assert simplex(pred) and simplex(cls)
        reg_loss1 = self.kl_criterion(pred, mixup_label)
        adv_loss = self.kl_criterion(cls, mix_indice)
        self.METERINTERFACE.tra_sup_mixup.add(reg_loss1.item())
        self.METERINTERFACE.tra_cls.add(adv_loss.item())

        # Discriminator
        return reg_loss1 + adv_loss

    def _mixup(self, label_img, label_gt, unlab_img, pseudo_label):
        bn, *shape = label_img.shape
        alpha = self.beta_distr.sample((bn,)).squeeze(1).to(self.device)
        _alpha = alpha.view(bn, 1, 1, 1).repeat(1, *shape)
        assert _alpha.shape == label_img.shape
        mixup_img = label_img * _alpha + unlab_img * (1 - _alpha)
        mixup_label = class2one_hot(label_gt.unsqueeze(dim=1).unsqueeze(dim=2),
                                    C=self.model.arch_dict['num_classes']).squeeze().float() * alpha.view(bn, 1) \
                      + pseudo_label * (1 - alpha).view(bn, 1)
        mixup_index = torch.stack([alpha, 1 - alpha], dim=1).to(self.device)

        assert mixup_img.shape == label_img.shape
        assert mixup_label.shape == pseudo_label.shape
        assert mixup_index.shape[0] == bn
        assert simplex(mixup_index)

        return mixup_img, mixup_label, mixup_index


class VAT_Trainer(AdaNetTrainer):

    def __init__(self, model: Model, labeled_loader: DataLoader, unlabeled_loader: DataLoader, val_loader: DataLoader,
                 max_epoch: int = 100, weight: float = 1, use_entropy: bool = True, save_dir: str = 'vat',
                 checkpoint_path: str = None,
                 device='cpu', config: dict = None, **kwargs) -> None:
        super().__init__(model, labeled_loader, unlabeled_loader, val_loader, max_epoch, weight, save_dir,
                         checkpoint_path, device, config, **kwargs)
        self.use_entropy = use_entropy

    def _trainer_specific_loss(self, label_img, label_gt, unlab_img, *args, **kwargs):
        adversarial_loss, *_ = VATLoss(xi=1, eps=10.0, prop_eps=1)(self.model.torchnet, unlab_img)
        entropy = 0
        if self.use_entropy:
            unlabled_predicts, *_ = self.model(unlab_img)
            entropy = Entropy(reduce=True)(unlabled_predicts)
        return entropy + adversarial_loss
