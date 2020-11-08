import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.binary_cross_entropy import bce2d
cfg = {'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
       'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], }


class VGGfs(nn.Module):
    def __init__(self, cfg):
        super(VGGfs, self).__init__()
        self.cfg = cfg
        self.features = self.make_layers()

    def forward(self, x):
        i = 0
        c = []
        for v in self.cfg:
            if v == 'M':
                c.append(x)
                x = self.features[i](x)
                i = i + 1
            else:
                x = self.features[i](x)
                x = self.features[i + 1](x)
                i = i + 2
        return tuple(c)

    def make_layers(self):
        layers = []
        in_channels = 3
        for i, v in enumerate(self.cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                if i == 0:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=35)
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.ModuleList(layers)


# definition of HED module
class Network(nn.Module):
    def __init__(self, pretrained_model=None):
        # define VGG architecture and layers
        super(Network, self).__init__()

        # define fully-convolutional layers
        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(128, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn5 = nn.Conv2d(512, 2, 1)
        self.fuse = nn.Conv2d(5, 1, 1)
        self.fuse4_5 = nn.Conv2d(2, 2, 1)
        self.fuse3_4 = nn.Conv2d(2, 2, 1)
        self.fuse2_3 = nn.Conv2d(2, 2, 1)
        self.fuse1_2 = nn.Conv2d(2, 1, 1)

        self.upscore2 = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        self.upscore3 = lambda x: F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        self.upscore4 = lambda x: F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        self.upscore5 = lambda x: F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=False)
        self.up5_4 = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        self.up4_3 = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        self.up3_2 = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        self.up2_1 = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # initialize weights of layers
        for m in self.named_modules():
            if m[0] == 'fuse':
                m[1].weight.data.fill_(0.2)
                m[1].bias.data.zero_()
            elif isinstance(m[1], nn.Conv2d):
                m[1].weight.data.zero_()
                m[1].bias.data.zero_()
        self.fuse4_5.weight.data.fill_(0.25)
        self.fuse3_4.weight.data.fill_(0.25)
        self.fuse2_3.weight.data.fill_(0.25)
        self.fuse1_2.weight.data.fill_(0.25)

        self.VGG16fs = VGGfs(cfg['D'])
        if pretrained_model is not None:
            print("Loading pretrained weights from %s" % (pretrained_model))
            state_dict = torch.load(pretrained_model)
            self.VGG16fs.load_state_dict({k: v for k, v in state_dict.items() if k in self.VGG16fs.state_dict()})

    # define the computation graph
    def forward(self, *input):
        x = input[0]
        size = x.size()[2:4]
        # get output from VGG model
        conv1, conv2, conv3, conv4, conv5 = self.VGG16fs(x)

        ## side output
        slice5_1, slice5_2 = self.slice(self.dsn5(conv5))
        dsn5_up = self.upscore5(slice5_1)
        d5 = self.crop_2(dsn5_up, (34, 34) + size)

        dsn4 = self.dsn4(conv4)
        cat4_5 = torch.cat((dsn4, self.crop_1(self.up5_4(slice5_2), dsn4.size()[2:4])), dim=1)
        slice4_1, slice4_2 = self.slice(self.fuse4_5(cat4_5))
        dsn4_up = self.upscore4(slice4_1)
        d4 = self.crop_2(dsn4_up, (34, 34) + size)

        dsn3 = self.dsn3(conv3)
        cat3_4 = torch.cat((dsn3, self.crop_1(self.up4_3(slice4_2), dsn3.size()[2:4])), dim=1)
        slice3_1, slice3_2 = self.slice(self.fuse3_4(cat3_4))
        dsn3_up = self.upscore3(slice3_1)
        d3 = self.crop_2(dsn3_up, (34, 34) + size)

        dsn2 = self.dsn2(conv2)
        cat2_3 = torch.cat((dsn2, self.crop_1(self.up3_2(slice3_2), dsn2.size()[2:4])), dim=1)
        slice2_1, slice2_2 = self.slice(self.fuse2_3(cat2_3))
        dsn2_up = self.upscore2(slice2_1)
        d2 = self.crop_2(dsn2_up, (34, 34) + size)

        dsn1 = self.dsn1(conv1)
        cat1_2 = torch.cat((dsn1, self.crop_1(self.up2_1(slice2_2), dsn1.size()[2:4])), dim=1)
        d1 = self.crop_2(self.fuse1_2(cat1_2), (34, 34) + size)

        d6 = self.fuse(torch.cat((d1, d2, d3, d4, d5), 1))

        if self.training:
            loss1 = bce2d(d1, input[1])
            loss2 = bce2d(d2, input[1])
            loss3 = bce2d(d3, input[1])
            loss4 = bce2d(d4, input[1])
            loss5 = bce2d(d5, input[1])
            loss6 = bce2d(d6, input[1])
            return loss1 + loss2 + loss3 + loss4 + loss5 + loss6

        else:
            d1 = torch.sigmoid(d1)
            d2 = torch.sigmoid(d2)
            d3 = torch.sigmoid(d3)
            d4 = torch.sigmoid(d4)
            d5 = torch.sigmoid(d5)
            d6 = torch.sigmoid(d6)
        return d6

    # function to crop the padding pixels
    def crop_1(self, d, size):  # use to keep same to the out the former layer
        d_h, d_w = d.size()[2:4]
        g_h, g_w = size[0], size[1]
        d1 = d[:, :, int(math.floor((d_h - g_h) / 2.0)):int(math.floor((d_h - g_h) / 2.0)) + g_h,
               int(math.floor((d_w - g_w) / 2.0)):int(math.floor((d_w - g_w) / 2.0)) + g_w]
        return d1

    def crop_2(self, d, region):  # use for crop the keep to input data
        x, y, h, w = region
        d1 = d[:, :, x:x + h, y:y + w]
        return d1

    def slice(self, data):
        length = len(data[0])
        return data[:, :length // 2, :, :], data[:, length // 2:, :, :]

    def parameters(net, lr):
        parameters = [{'params': net.dsn1.weight, 'lr': lr * 0.01},
                      {'params': net.dsn1.bias, 'lr': lr * 0.02, 'weight_decay': 0},
                      {'params': net.dsn2.weight, 'lr': lr * 0.01},
                      {'params': net.dsn2.bias, 'lr': lr * 0.02, 'weight_decay': 0},
                      {'params': net.dsn3.weight, 'lr': lr * 0.01},
                      {'params': net.dsn3.bias, 'lr': lr * 0.02, 'weight_decay': 0},
                      {'params': net.dsn4.weight, 'lr': lr * 0.01},
                      {'params': net.dsn4.bias, 'lr': lr * 0.02, 'weight_decay': 0},
                      {'params': net.dsn5.weight, 'lr': lr * 0.01},
                      {'params': net.dsn5.bias, 'lr': lr * 0.02, 'weight_decay': 0},
                      {'params': net.fuse.weight, 'lr': lr * 0.001},
                      {'params': net.fuse.bias, 'lr': lr * 0.002, 'weight_decay': 0},
                      {'params': net.fuse1_2.weight, 'lr': lr * 0.05},
                      {'params': net.fuse1_2.bias, 'lr': lr * 0.002, 'weight_decay': 0},
                      {'params': net.fuse2_3.weight, 'lr': lr * 0.05},
                      {'params': net.fuse2_3.bias, 'lr': lr * 0.002, 'weight_decay': 0},
                      {'params': net.fuse3_4.weight, 'lr': lr * 0.05},
                      {'params': net.fuse3_4.bias, 'lr': lr * 0.002, 'weight_decay': 0},
                      {'params': net.fuse4_5.weight, 'lr': lr * 0.05},
                      {'params': net.fuse4_5.bias, 'lr': lr * 0.002, 'weight_decay': 0},
                      ]
        for i, layer in enumerate(net.VGG16fs.features):
            if isinstance(layer, nn.Conv2d):
                if i < 24:
                    parameters.extend([{'params': layer.weight, 'lr': lr * 1},
                                       {'params': layer.bias, 'lr': lr * 2, 'weight_decay': 0}])
                else:
                    parameters.extend([{'params': layer.weight, 'lr': lr * 100},
                                       {'params': layer.bias, 'lr': lr * 200, 'weight_decay': 0}])
        return parameters


if __name__ == "__main__":
    data = torch.randn(1, 3, 64, 64)
    data2 = torch.randn(1, 2, 64, 64)
    target = torch.randn(1, 1, 64, 64)
    d1 = torch.cat((data, data2), dim=1)

    net = Network()
    net(data, target)
