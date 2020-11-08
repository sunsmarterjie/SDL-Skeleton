import torch
import torch.nn as nn
import torch.nn.functional as F

OPS = {
    'skip': lambda C, stride: Identity(),
    'conv1': lambda C, stride: Conv(C, C, 1, stride, 0),
    'conv3': lambda C, stride: Conv(C, C, 3, stride, 1),
    'conv5': lambda C, stride: Conv(C, C, 5, stride, 2),
    'dconv3_2': lambda C, stride: DilConv(C, C, 3, stride, 2, 2),
    'dconv3_4': lambda C, stride: DilConv(C, C, 3, stride, 4, 4),
    'dconv3_8': lambda C, stride: DilConv(C, C, 3, stride, 8, 8),
    'dconv5_2': lambda C, stride: DilConv(C, C, 5, stride, 4, 2),
    'dconv5_4': lambda C, stride: DilConv(C, C, 5, stride, 8, 4),
    'dconv5_8': lambda C, stride: DilConv(C, C, 5, stride, 16, 8),
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, inplace=True, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=inplace),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding),
            # nn.BatchNorm2d(C_out)
        )
        # for layer in self.op.modules():
        #     if isinstance(layer, nn.Conv2d):
        #         nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, inplace=False, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=inplace),
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            # nn.BatchNorm2d(C_out, affine=affine),
        )
        # for layer in self.op.modules():
        #     if isinstance(layer, nn.Conv2d):
        #         nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.op(x)


class Conv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, inplace=False, affine=True):
        super(Conv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=inplace),
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.BatchNorm2d(C_out),  # memory overflow if use this BN
        )
        # for layer in self.op.modules():
        #     if isinstance(layer, nn.Conv2d):
        #         nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, inplace=True, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=inplace)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0)
        # self.bn = nn.BatchNorm2d(C_out)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        # out = self.bn(out)
        return out
