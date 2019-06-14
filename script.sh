#!/usr/bin/env bash
# Semi-supervised Learning without augmentation
python main.py Trainer.save_dir=semi-no_aug/cifar \
DataLoader.name=cifar10 \
DataLoader.aug=False \
Trainer.max_epoch=500 \
Trainer.epoch_decay_start=460 \
Trainer.eps=10.0 \
Trainer.use_entropy=False

python main.py Trainer.save_dir=semi-no_aug/svhn \
DataLoader.name=svhn \
DataLoader.aug=False \
Trainer.max_epoch=120 \
Trainer.epoch_decay_start=80 \
Trainer.eps=2.5 \
Arch.top_bn=True \
Trainer.use_entropy=False

# Semi-supervised Learning with augmentation
python main.py Trainer.save_dir=semi-aug/cifar \
DataLoader.name=cifar10 \
DataLoader.aug=True \
Trainer.max_epoch=500 \
Trainer.epoch_decay_start=460 \
Trainer.eps=8.0 \
Trainer.use_entropy=False

python main.py Trainer.save_dir=semi-aug/svhn \
DataLoader.name=svhn \
DataLoader.aug=True \
Trainer.max_epoch=120 \
Trainer.epoch_decay_start=80 \
Trainer.eps=3.5 \
Arch.top_bn=True \
Trainer.use_entropy=False

#Semi-supervised Learning with augmentation + entropy minimization
python main.py Trainer.save_dir=semi-aug-ent/cifar \
DataLoader.name=cifar10 \
DataLoader.aug=True \
Trainer.max_epoch=500 \
Trainer.epoch_decay_start=460 \
Trainer.eps=8.0 \
Trainer.use_entropy=True

python main.py Trainer.save_dir=semi-aug-ent/svhn \
DataLoader.name=svhn \
DataLoader.aug=True \
Trainer.max_epoch=120 \
Trainer.epoch_decay_start=80 \
Trainer.eps=3.5 \
Arch.top_bn=True \
Trainer.use_entropy=True