
import numpy as np
from deepclustering.scheduler import Scheduler


class CustomScheduler(Scheduler):

    def __init__(self, max_epoch):
        super().__init__()
        self.max_epoch = int(max_epoch)
        self.epoch = 0
        self.lmd_func = lambda x: 1.0 * (2. / (1. + np.exp(-10 * x)) - 1.)

    @staticmethod
    def get_lr(function, epoch, max_epoch):
        return function(float(epoch / max_epoch))

    @property
    def value(self):
        return self.get_lr(self.lmd_func, self.epoch, self.max_epoch)

    def step(self):
        self.epoch += 1
