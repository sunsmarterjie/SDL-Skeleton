import torch
import torch.nn as nn
from torch.autograd import Variable
from modules.weighted_euclidean_loss import wel
from modules.binary_cross_entropy import bce2d
import torch.nn.functional as F
import math
from modules.vgg import VGGfs, cfg


class Network(nn.Module):
    def __init__(self, pretrained_model=None):
        # define VGG architecture and layers
        super(Network, self).__init__()

        self.VGG16fs = VGGfs(cfg['D'])
        self.dsn3 = nn.Conv2d(256, 256, 1)
        self.dsn4 = nn.Conv2d(512, 256, 1)
        self.dsn5 = nn.Conv2d(512, 256, 1)
        self.dsn3_relu = nn.ReLU(inplace=True)
        self.dsn4_relu = nn.ReLU(inplace=True)
        self.dsn5_relu = nn.ReLU(inplace=True)

        self.d2conv5 = nn.Conv2d(512, 128, kernel_size=3, padding=2, dilation=2)
        self.d4conv5 = nn.Conv2d(512, 128, kernel_size=3, padding=4, dilation=4)
        self.d8conv5 = nn.Conv2d(512, 128, kernel_size=3, padding=8, dilation=8)
        self.d16conv5 = nn.Conv2d(512, 128, kernel_size=3, padding=16, dilation=16)
        self.d2_relu = nn.ReLU(inplace=True)
        self.d4_relu = nn.ReLU(inplace=True)
        self.d8_relu = nn.ReLU(inplace=True)
        self.d16_relu = nn.ReLU(inplace=True)

        self.sdconv = nn.Conv2d(128 * 4, 256, 1)
        self.sd_relu = nn.ReLU(inplace=True)

        self.fconv1 = nn.Conv2d(4 * 256, 512, 1)
        self.fconv2 = nn.Conv2d(512, 512, 1)
        self.fconv3 = nn.Conv2d(512, 2, 1)  # for wel loss
        # self.fconv3 = nn.Conv2d(512, 1, 1)  # for bce loss
        self.fconv1_relu = nn.ReLU(inplace=True)
        self.fconv2_relu = nn.ReLU(inplace=True)

        self.up2 = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        self.up4 = lambda x: F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        self.init_parameters()
        if pretrained_model is not None:
            print("Loading pretrained weights from %s" % (pretrained_model))
            state_dict = torch.load(pretrained_model)
            self.VGG16fs.load_state_dict({k: v for k, v in state_dict.items() if k in self.VGG16fs.state_dict()})

    def forward(self, *input):
        size = input[0].size()[2:4]
        conv1, conv2, conv3, conv4, conv5 = self.VGG16fs(input[0])

        dsn3_conv = self.dsn3(conv3)
        dsn3_relu = self.dsn3_relu(dsn3_conv)

        dsn4_conv = self.dsn4(conv4)
        dsn4_relu = self.dsn4_relu(dsn4_conv)
        up_dsn4 = self.up2(dsn4_relu)
        dsn4 = self.crop_2(up_dsn4, (0, 0) + conv3.size()[2:4])

        dsn5_conv = self.dsn5(conv5)
        dsn5_relu = self.dsn5_relu(dsn5_conv)
        up_dsn5 = self.up4(dsn5_relu)
        dsn5 = self.crop_2(up_dsn5, (0, 0) + conv3.size()[2:4])

        d2_conv = self.d2conv5(conv5)
        d2_relu = self.d2_relu(d2_conv)
        d4_conv = self.d4conv5(conv5)
        d4_relu = self.d4_relu(d4_conv)
        d8_conv = self.d8conv5(conv5)
        d8_relu = self.d8_relu(d8_conv)
        d16_conv = self.d16conv5(conv5)
        d16_relu = self.d16_relu(d16_conv)
        fuse_d = torch.cat([d2_relu, d4_relu, d8_relu, d16_relu], dim=1)
        sd_conv = self.sdconv(fuse_d)
        sd_relu = self.sd_relu(sd_conv)
        ups4 = self.up4(sd_relu)
        sdcrop = self.crop_2(ups4, (0, 0) + conv3.size()[2:4])

        fuse = torch.cat([dsn3_relu, dsn4, dsn5, sdcrop], dim=1)
        f1_conv = self.fconv1(fuse)
        f1_relu = self.fconv1_relu(f1_conv)
        f2_conv = self.fconv2(f1_relu)
        f2_relu = self.fconv2_relu(f2_conv)
        f3 = self.fconv3(f2_relu)

        fup = self.up4(f3)
        # fup = self.upscore4(f3)
        fcrop = self.crop_2(fup, (34, 34) + size)
        if self.training:
            loss = wel(fcrop, input[1], input[2])
            # loss = bce2d(fcrop, input[1])
            return loss
        else:
            return fcrop

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

    def parameters(net, lr):
        parameters = [
            {'params': net.dsn3.weight, 'lr': lr * 10}, {'params': net.dsn3.bias, 'lr': lr * 20, 'weight_decay': 0},
            {'params': net.dsn4.weight, 'lr': lr * 10}, {'params': net.dsn4.bias, 'lr': lr * 20, 'weight_decay': 0},
            {'params': net.dsn5.weight, 'lr': lr * 10}, {'params': net.dsn5.bias, 'lr': lr * 20, 'weight_decay': 0},
            {'params': net.d2conv5.weight, 'lr': lr * 10},
            {'params': net.d2conv5.bias, 'lr': lr * 20, 'weight_decay': 0},
            {'params': net.d4conv5.weight, 'lr': lr * 10},
            {'params': net.d4conv5.bias, 'lr': lr * 20, 'weight_decay': 0},
            {'params': net.d8conv5.weight, 'lr': lr * 10},
            {'params': net.d8conv5.bias, 'lr': lr * 20, 'weight_decay': 0},
            {'params': net.d16conv5.weight, 'lr': lr * 10},
            {'params': net.d16conv5.bias, 'lr': lr * 20, 'weight_decay': 0},
            {'params': net.sdconv.weight, 'lr': lr * 10}, {'params': net.sdconv.bias, 'lr': lr * 20, 'weight_decay': 0},
            {'params': net.fconv1.weight, 'lr': lr * 10}, {'params': net.fconv1.bias, 'lr': lr * 20, 'weight_decay': 0},
            {'params': net.fconv2.weight, 'lr': lr * 10}, {'params': net.fconv2.bias, 'lr': lr * 20, 'weight_decay': 0},
            {'params': net.fconv3.weight, 'lr': lr * 10}, {'params': net.fconv3.bias, 'lr': lr * 20, 'weight_decay': 0},
        ]
        for i, layer in enumerate(net.VGG16fs.features):
            if isinstance(layer, nn.Conv2d):
                parameters.extend([
                    {'params': layer.weight, 'lr': lr * 1},
                    {'params': layer.bias, 'lr': lr * 2, 'weight_decay': 0}
                ])
        return parameters

    def init_parameters(self):
        torch.nn.init.normal_(self.dsn3.weight, std=0.0001)
        torch.nn.init.normal_(self.dsn4.weight, std=0.001)
        torch.nn.init.normal_(self.dsn5.weight, std=0.01)
        torch.nn.init.xavier_normal_(self.d2conv5.weight)
        torch.nn.init.xavier_normal_(self.d4conv5.weight)
        torch.nn.init.xavier_normal_(self.d8conv5.weight)
        torch.nn.init.xavier_normal_(self.d16conv5.weight)
        torch.nn.init.normal_(self.sdconv.weight, std=0.01)
        torch.nn.init.xavier_normal_(self.fconv1.weight)
        torch.nn.init.xavier_normal_(self.fconv2.weight)
        torch.nn.init.xavier_normal_(self.fconv3.weight)
