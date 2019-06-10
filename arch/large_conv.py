#   Networks adapted from https://github.com/qinenergy/adanet/blob/master/convlarge/cnn.py
import torch
from torch import nn
from torch.nn import functional as F


def conv_block(input_dim, out_dim, kernel_size=3, stride=1, padding=1, lrelu_slope=0.01):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=input_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(inplace=True, negative_slope=lrelu_slope)
    )


class identical(nn.Module):

    def forward(self, input):
        return input


class LargeConvNet(nn.Module):
    def __init__(self, input_dim=3, num_classes=10, stochastic=True, top_bn=False):
        super().__init__()
        self.top_bn = top_bn
        self.block1 = conv_block(input_dim, 128, 3, 1, padding=1, lrelu_slope=0.1)
        self.block2 = conv_block(128, 128, 3, 1, 1, 0.1)
        self.block3 = conv_block(128, 128, 3, 1, 1, 0.1)

        self.block4 = conv_block(128, 256, 3, 1, 1, 0.1)
        self.block5 = conv_block(256, 256, 3, 1, 1, 0.1)
        self.block6 = conv_block(256, 256, 3, 1, 1, 0.1)

        self.block7 = conv_block(256, 512, 3, 1, 1, 0.1)
        self.block8 = conv_block(512, 256, 3, 1, 1, 0.1)
        self.block9 = conv_block(256, 128, 3, 1, 1, 0.1)

        self.AveragePooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

        dropout = nn.Dropout2d() if stochastic else identical()

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            dropout
        )

    def forward(self, input):
        out = self.block1(input)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bottleneck(out)

        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.bottleneck(out)

        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        feature = self.AveragePooling(out)

        out = self.fc(feature.view(feature.shape[0], -1))
        if self.top_bn:
            out = nn.BatchNorm1d(10)(out)
        return out, feature.view(feature.shape[0], -1)


class Discriminator(nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, input):
        out = F.relu(self.fc1(input), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)
        return out


# class GradReverse(Function):
#     def forward(self, x):
#         return x
#
#     def backward(self, grad_output):
#         return (-grad_output)

# def grad_reverse(x):
#     return GradReverse()(x)


class SimpleNet(nn.Module):

    def __init__(self, in_channel, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_channel, out_features=10)
        self.fc2 = nn.Linear(10, num_classes)

    def forward(self, input):
        out = self.fc1(input)
        out = F.relu(out, inplace=True)
        return self.fc2(out)


if __name__ == '__main__':
    net = LargeConvNet(3, 10)
    img = torch.randn(10, 3, 32, 32)
    out = net(img)
