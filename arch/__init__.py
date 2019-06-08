#
#
#   This is the interface for networks
#
#
from deepclustering.arch import get_arch, ARCH_CALLABLES
from deepclustering.utils.general import _register

from torch import nn
from torch.nn import functional as F


class SimpleNet(nn.Module):

    def __init__(self, in_channel, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_channel, out_features=10)
        self.fc2 = nn.Linear(10, num_classes)

    def forward(self, input):
        out = self.fc1(input)
        out = F.relu(out, inplace=True)
        return self.fc2(out)


_register('simplenet', SimpleNet, CALLABLE_DICT=ARCH_CALLABLES)
param = {'in_channel': 1, 'num_classes': 10}
net = get_arch('simplenet', param)
print(net)
