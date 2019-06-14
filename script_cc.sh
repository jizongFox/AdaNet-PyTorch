#!/usr/bin/env bash
wrapper(){
    hour=$1
    command=$2
    module load python/3.6
    source $HOME/torchenv36/bin/activate
    module load scipy-stack
    echo ${command} > tmp.sh
    sed -i '1i\#!/bin/bash' tmp.sh
    sbatch  --job-name="${commend}" \
     --nodes=1  \
     --gres=gpu:1 \
     --cpus-per-task=6  \
     --mem=16000M \
     --time=0-${hour}:00 \
     --account=rrg-mpederso \
     --mail-user=jizong.peng.1@etsmtl.net \
     --mail-type=ALL   \
    ./tmp.sh
    rm ./tmp.sh
}

# Semi-supervised Learning without augmentation
declare -a StringArray=(
"python main.py Trainer.save_dir=semi-no_aug/cifar \
DataLoader.name=cifar10 \
DataLoader.aug=False \
Trainer.max_epoch=500 \
Trainer.epoch_decay_start=460 \
Trainer.eps=10.0 \
Trainer.use_entropy=False"
\
"python main.py Trainer.save_dir=semi-no_aug/svhn \
DataLoader.name=svhn \
DataLoader.aug=False \
Trainer.max_epoch=120 \
Trainer.epoch_decay_start=80 \
Trainer.eps=2.5 \
Arch.top_bn=True \
Trainer.use_entropy=False"
\
"python main.py Trainer.save_dir=semi-aug/cifar \
DataLoader.name=cifar10 \
DataLoader.aug=True \
Trainer.max_epoch=500 \
Trainer.epoch_decay_start=460 \
Trainer.eps=8.0 \
Trainer.use_entropy=False"
\
"python main.py Trainer.save_dir=semi-aug/svhn \
DataLoader.name=svhn \
DataLoader.aug=True \
Trainer.max_epoch=120 \
Trainer.epoch_decay_start=80 \
Trainer.eps=3.5 \
Arch.top_bn=True \
Trainer.use_entropy=False"
\
"python main.py Trainer.save_dir=semi-aug/cifar \
DataLoader.name=cifar10 \
DataLoader.aug=True \
Trainer.max_epoch=500 \
Trainer.epoch_decay_start=460 \
Trainer.eps=8.0 \
Trainer.use_entropy=True"
\
"python main.py Trainer.save_dir=semi-aug/svhn \
DataLoader.name=svhn \
DataLoader.aug=True \
Trainer.max_epoch=120 \
Trainer.epoch_decay_start=80 \
Trainer.eps=3.5 \
Arch.top_bn=True \
Trainer.use_entropy=True"
)

time=1

for cmd in "${StringArray[@]}"
do
echo $cmd
#$cmd
#wrapper "${time}" "${cmd}"
done