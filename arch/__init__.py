#
#
#   This is the interface for networks
#
#
from deepclustering.arch import _register_arch

from .large_conv import LargeConvNet, SimpleNet

_register_arch('simplenet', SimpleNet)
_register_arch('largeconvnet', LargeConvNet)
