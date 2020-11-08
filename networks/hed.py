import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.binary_cross_entropy import bce2d
from modules.vgg import VGGfs, cfg


class Network(nn.Module):
    def __init__(self, pretrained_model=None):
        super(Network, self).__init__()

        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(128, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn5 = nn.Conv2d(512, 1, 1)
        self.fuse = nn.Conv2d(5, 1, 1)

        self.upscore2 = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        self.upscore3 = lambda x: F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        self.upscore4 = lambda x: F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        self.upscore5 = lambda x: F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=False)

        # initialize weights of layers
        for m in self.named_modules():
            # print(m)
            if m[0] == 'fuse':
                m[1].weight.data.fill_(0.2)
                m[1].bias.data.zero_()
            elif isinstance(m[1], nn.Conv2d):
                m[1].weight.data.zero_()
                m[1].bias.data.zero_()

        self.VGG16fs = VGGfs(cfg['D'])
        if pretrained_model is not None:
            print("Loading pretrained weights from %s" % (pretrained_model))
            state_dict = torch.load(pretrained_model)
            self.VGG16fs.load_state_dict({k: v for k, v in state_dict.items() if k in self.VGG16fs.state_dict()})

    def forward(self, *input):
        size = input[0].size()[2:4]

        conv1, conv2, conv3, conv4, conv5 = self.VGG16fs(input[0])

        dsn5_up = self.upscore5(self.dsn5(conv5))
        d5 = self.crop(dsn5_up, (34, 34) + size)
        dsn4_up = self.upscore4(self.dsn4(conv4))
        d4 = self.crop(dsn4_up, (34, 34) + size)
        dsn3_up = self.upscore3(self.dsn3(conv3))
        d3 = self.crop(dsn3_up, (34, 34) + size)
        dsn2_up = self.upscore2(self.dsn2(conv2))
        d2 = self.crop(dsn2_up, (34, 34) + size)
        dsn1 = self.dsn1(conv1)
        d1 = self.crop(dsn1, (34, 34) + size)

        d6 = self.fuse(torch.cat((d1, d2, d3, d4, d5), 1))

        if self.training:
            loss1 = bce2d(d1, input[1])
            loss2 = bce2d(d2, input[1])
            loss3 = bce2d(d3, input[1])
            loss4 = bce2d(d4, input[1])
            loss5 = bce2d(d5, input[1])
            loss6 = bce2d(d6, input[1])
            return loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            # return loss5
        else:
            d1 = torch.sigmoid(d1)
            d2 = torch.sigmoid(d2)
            d3 = torch.sigmoid(d3)
            d4 = torch.sigmoid(d4)
            d5 = torch.sigmoid(d5)
            d6 = torch.sigmoid(d6)
            # return d1, d2, d3, d4, d5, d6
            return d6

    def crop(self, d, region):  # use for crop the keep to input data
        x, y, h, w = region
        d1 = d[:, :, x:x + h, y:y + w]
        return d1

    def parameters(net, lr):
        parameters = [
            {'params': net.dsn1.weight, 'lr': lr * 0.01}, {'params': net.dsn1.bias, 'lr': lr * 0.02, 'weight_decay': 0},
            {'params': net.dsn2.weight, 'lr': lr * 0.01}, {'params': net.dsn2.bias, 'lr': lr * 0.02, 'weight_decay': 0},
            {'params': net.dsn3.weight, 'lr': lr * 0.01}, {'params': net.dsn3.bias, 'lr': lr * 0.02, 'weight_decay': 0},
            {'params': net.dsn4.weight, 'lr': lr * 0.01}, {'params': net.dsn4.bias, 'lr': lr * 0.02, 'weight_decay': 0},
            {'params': net.dsn5.weight, 'lr': lr * 0.01}, {'params': net.dsn5.bias, 'lr': lr * 0.02, 'weight_decay': 0},
            {'params': net.fuse.weight, 'lr': lr * 0.001},
            {'params': net.fuse.bias, 'lr': lr * 0.002, 'weight_decay': 0},
        ]
        for i, layer in enumerate(net.VGG16fs.features):
            if isinstance(layer, nn.Conv2d):
                if i < 24:
                    parameters.extend([
                        {'params': layer.weight, 'lr': lr * 1},
                        {'params': layer.bias, 'lr': lr * 2, 'weight_decay': 0}
                    ])
                else:
                    parameters.extend([
                        {'params': layer.weight, 'lr': lr * 100},
                        {'params': layer.bias, 'lr': lr * 200, 'weight_decay': 0}
                    ])
        return parameters


if __name__ == '__main__':
    net = Network()
    x = torch.randn(1, 3, 256, 256)
    t = torch.randn(1, 1, 256, 256)
    out = net(x, t)
    print(out)
