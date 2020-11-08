import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.binary_cross_entropy import bce2d
from modules.upsample import Upsample

cfg = {'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
       'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], }


class VGGfs_hifi(nn.Module):
    def __init__(self, cfg):
        super(VGGfs_hifi, self).__init__()
        self.cfg = cfg
        self.features = self.make_layers()

    def forward(self, x):
        i = 0
        c = []
        for v in self.cfg:
            if v == 'M':
                x = self.features[i](x)
                i = i + 1
            else:
                x = self.features[i](x)
                x = self.features[i + 1](x)
                c.append(x)
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
        self.s1_1 = nn.Conv2d(64, 11, 1)
        self.s1_2 = nn.Conv2d(64, 11, 1)
        self.s2_1_top = nn.Conv2d(128, 11, 1)
        self.s2_2_top = nn.Conv2d(128, 11, 1)
        self.s2_1_down = nn.Conv2d(128, 11, 1)
        self.s2_2_down = nn.Conv2d(128, 11, 1)
        self.s3_1_top = nn.Conv2d(256, 11, 1)
        self.s3_2_top = nn.Conv2d(256, 11, 1)
        self.s3_3_top = nn.Conv2d(256, 11, 1)
        self.s3_1_down = nn.Conv2d(256, 11, 1)
        self.s3_2_down = nn.Conv2d(256, 11, 1)
        self.s3_3_down = nn.Conv2d(256, 11, 1)
        self.s4_1_top = nn.Conv2d(512, 11, 1)
        self.s4_2_top = nn.Conv2d(512, 11, 1)
        self.s4_3_top = nn.Conv2d(512, 11, 1)
        self.s4_1_down = nn.Conv2d(512, 11, 1)
        self.s4_2_down = nn.Conv2d(512, 11, 1)
        self.s4_3_down = nn.Conv2d(512, 11, 1)
        self.s5_1 = nn.Conv2d(512, 11, 1)
        self.s5_2 = nn.Conv2d(512, 11, 1)
        self.s5_3 = nn.Conv2d(512, 11, 1)

        self.h1s1_11to11 = nn.Conv2d(11, 11, 1)
        self.h1s2_11to11_top = nn.Conv2d(11, 11, 1)
        self.h1s2_11to11_down = nn.Conv2d(11, 11, 1)
        self.h1s3_11to11_top = nn.Conv2d(11, 11, 1)
        self.h1s3_11to11_down = nn.Conv2d(11, 11, 1)
        self.h1s4_11to11 = nn.Conv2d(11, 11, 1)

        self.h1s1_11to3 = nn.Conv2d(11, 1, 1)
        self.h1s2_11to4 = nn.Conv2d(11, 1, 1)
        self.h1s3_11to5 = nn.Conv2d(11, 1, 1)
        self.h1s4_11to6 = nn.Conv2d(11, 1, 1)
        '''
        self.h2s1_11to3 = nn.Conv2d(11, 3, 1)
        self.h2s2_11to4 = nn.Conv2d(11, 4, 1)
        self.h2s3_11to5 = nn.Conv2d(11, 5, 1)

        self.fuse_h1_1 = nn.Conv2d(4, 1, 1)  # h1s1 h1s2 h1s3 h1s4
        self.fuse_h1_2 = nn.Conv2d(3, 1, 1)  # h1s2 h1s3 h1s4
        self.fuse_h1_3 = nn.Conv2d(2, 1, 1)  # h1s3 h1s4

        self.fuse_h2_1 = nn.Conv2d(3, 1, 1)  # h2s1 h2s2 h2s3
        self.fuse_h2_2 = nn.Conv2d(2, 1, 1)  # h2s2 h2s3

        self.fuse_h1h2 = nn.Conv2d(7, 1, 1)  # fuse all hxsx
        '''
        # to delete
        self.fuse = nn.Conv2d(4, 1, 1)
        self.fuse.weight.data.fill_(0.25)
        self.fuse.bias.data.fill_(0)

        self.upscore2 = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        self.upscore4 = lambda x: F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        self.upscore8 = lambda x: F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        self.upscore16 = lambda x: F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=False)

        self.init_parameters()

        self.VGG16fs = VGGfs_hifi(cfg['D'])
        if pretrained_model is not None:
            print("Loading pretrained weights from %s" % (pretrained_model))
            state_dict = torch.load(pretrained_model)
            self.VGG16fs.load_state_dict({k: v for k, v in state_dict.items() if k in self.VGG16fs.state_dict()})

    # define the computation graph
    def forward(self, *input):
        x = input[0]
        size = x.size()[2:4]
        convs = self.VGG16fs(x)

        ## side output
        s1_1 = self.s1_1(convs[0])
        s1_2 = self.s1_2(convs[1])
        s2_1_top = self.s2_1_top(convs[2])
        s2_2_top = self.s2_2_top(convs[3])
        s2_1_top_up = self.upscore2(s2_1_top)
        s2_2_top_up = self.upscore2(s2_2_top)
        s2_1_top_up_crop = self.crop_1(s2_1_top_up, s1_1.size()[2:4])
        s2_2_top_up_crop = self.crop_1(s2_2_top_up, s1_1.size()[2:4])
        h1s1 = s1_1 + s1_2 + s2_1_top_up_crop + s2_2_top_up_crop

        s2_1_down = self.s2_1_down(convs[2])
        s2_2_down = self.s2_2_down(convs[3])
        s3_1_top = self.s3_1_top(convs[4])
        s3_2_top = self.s3_2_top(convs[5])
        s3_3_top = self.s3_3_top(convs[6])
        s3_1_top_up = self.upscore2(s3_1_top)
        s3_2_top_up = self.upscore2(s3_2_top)
        s3_3_top_up = self.upscore2(s3_3_top)
        s3_1_top_up_crop = self.crop_1(s3_1_top_up, s2_1_down.size()[2:4])
        s3_2_top_up_crop = self.crop_1(s3_2_top_up, s2_1_down.size()[2:4])
        s3_3_top_up_crop = self.crop_1(s3_3_top_up, s2_1_down.size()[2:4])
        h1s2 = s2_1_down + s2_2_down + s3_1_top_up_crop + s3_2_top_up_crop + s3_3_top_up_crop

        s3_1_down = self.s3_1_down(convs[4])
        s3_2_down = self.s3_2_down(convs[5])
        s3_3_down = self.s3_3_down(convs[6])
        s4_1_top = self.s4_1_top(convs[7])
        s4_2_top = self.s4_2_top(convs[8])
        s4_3_top = self.s4_3_top(convs[9])
        s4_1_top_up = self.upscore2(s4_1_top)
        s4_2_top_up = self.upscore2(s4_2_top)
        s4_3_top_up = self.upscore2(s4_3_top)
        s4_1_top_up_crop = self.crop_1(s4_1_top_up, s3_1_down.size()[2:4])
        s4_2_top_up_crop = self.crop_1(s4_2_top_up, s3_1_down.size()[2:4])
        s4_3_top_up_crop = self.crop_1(s4_3_top_up, s3_1_down.size()[2:4])
        h1s3 = s3_1_down + s3_2_down + s3_3_down + s4_1_top_up_crop + s4_2_top_up_crop + s4_3_top_up_crop

        s4_1_down = self.s4_1_down(convs[7])
        s4_2_down = self.s4_2_down(convs[8])
        s4_3_down = self.s4_3_down(convs[9])
        s5_1 = self.s5_1(convs[10])
        s5_2 = self.s5_2(convs[11])
        s5_3 = self.s5_3(convs[12])
        s5_1_up = self.upscore2(s5_1)
        s5_2_up = self.upscore2(s5_2)
        s5_3_up = self.upscore2(s5_3)
        s5_1_up_crop = self.crop_1(s5_1_up, s4_1_down.size()[2:4])
        s5_2_up_crop = self.crop_1(s5_2_up, s4_1_down.size()[2:4])
        s5_3_up_crop = self.crop_1(s5_3_up, s4_1_down.size()[2:4])
        h1s4 = s4_1_down + s4_2_down + s4_3_down + s5_1_up_crop + s5_2_up_crop + s5_3_up_crop

        # loss1  && h1s1_slice
        h1s1_3 = self.h1s1_11to3(h1s1)
        h1s1_3_crop = self.crop_2(h1s1_3, (34, 34) + size)
        h1s1_slice = self.slice(h1s1_3_crop)
        # loss2 && h1s2_slice
        h1s2_4 = self.h1s2_11to4(h1s2)
        h1s2_4_up = self.upscore2(h1s2_4)
        h1s2_4_up_crop = self.crop_2(h1s2_4_up, (34, 34) + size)
        h1s2_slice = self.slice(h1s2_4_up_crop)
        #  loss3 && h1s3_slice
        h1s3_5 = self.h1s3_11to5(h1s3)
        h1s3_5_up = self.upscore4(h1s3_5)
        h1s3_5_up_crop = self.crop_2(h1s3_5_up, (34, 34) + size)
        h1s3_slice = self.slice(h1s3_5_up_crop)
        #  loss4  && h1s4_slice
        h1s4_6 = self.h1s4_11to6(h1s4)
        h1s4_6_up = self.upscore8(h1s4_6)
        h1s4_6_up_crop = self.crop_2(h1s4_6_up, (34, 34) + size)
        h1s4_slice = self.slice(h1s4_6_up_crop)

        '''
        h1s1_11 = self.h1s1_11to11(h1s1)
        h1s2_11_top = self.h1s2_11to11_top(h1s2)
        h1s2_11_top_up = self.upscore2(h1s2_11_top)
        h1s2_11_top_up_crop = self.crop_1(h1s2_11_top_up, h1s1_11.size()[2:4])
        h2s1 = h1s1_11 + h1s2_11_top_up_crop

        h1s2_11_down = self.h1s2_11to11_down(h1s2)
        h1s3_11_top = self.h1s3_11to11_top(h1s3)
        h1s3_11_top_up = self.upscore2(h1s3_11_top)
        h1s3_11_top_up_crop = self.crop_1(h1s3_11_top_up, h1s2_11_down.size()[2:4])
        h2s2 = h1s2_11_down + h1s3_11_top_up_crop

        h1s3_11_down = self.h1s3_11to11_down(h1s3)
        h1s4_11 = self.h1s4_11to11(h1s4)
        h1s4_11_up = self.upscore2(h1s4_11)
        h1s4_11_up_crop = self.crop_1(h1s4_11_up, h1s3_11_down.size()[2:4])
        h2s3 = h1s3_11_down + h1s4_11_up_crop

        h2s1_3 = self.h2s1_11to3(h2s1)
        h2s1_3_crop = self.crop_2(h2s1_3, (34, 34) + size)
        h2s1_slice = self.slice(h2s1_3_crop)

        h2s2_4 = self.h2s2_11to4(h2s2)
        h2s2_4_up = self.upscore2(h2s2_4)
        h2s2_4_up_crop = self.crop_2(h2s2_4_up, (34, 34) + size)
        h2s2_slice = self.slice(h2s2_4_up_crop)

        h2s3_5 = self.h2s3_11to5(h2s3)
        h2s3_5_up = self.upscore4(h2s3_5)
        h2s3_5_up_crop = self.crop_2(h2s3_5_up, (34, 34) + size)
        h2s3_slice = self.slice(h2s3_5_up_crop)
        '''

        d1 = h1s1_slice[0]
        d2 = h1s2_slice[0]
        d3 = h1s3_slice[0]
        d4 = h1s4_slice[0]
        d5 = self.fuse(torch.cat((h1s1_slice[0], h1s2_slice[0], h1s3_slice[0], h1s4_slice[0]), dim=1))

        '''
        d5 = h2s1_slice[0]
        d6 = h2s2_slice[0]
        d7 = h2s3_slice[0]
        fuse_h1_1 = self.fuse_h1_1(torch.cat((h1s1_slice[1], h1s2_slice[1], h1s3_slice[1], h1s4_slice[1]), dim=1))
        fuse_h1_2 = self.fuse_h1_2(torch.cat((h1s2_slice[2], h1s3_slice[2], h1s4_slice[2]), dim=1))
        fuse_h1_3 = self.fuse_h1_3(torch.cat((h1s3_slice[3], h1s4_slice[3]), dim=1))
        d8 = fuse_h1_1 + fuse_h1_2 + fuse_h1_3 + h1s4_slice[4]
        fuse_h2_1 = self.fuse_h2_1(torch.cat((h2s1_slice[1], h2s2_slice[1], h2s3_slice[1]), dim=1))
        fuse_h2_2 = self.fuse_h2_2(torch.cat((h2s2_slice[2], h2s3_slice[2]), dim=1))
        d9 = fuse_h2_1 + fuse_h2_2 + h2s3_slice[3]
        d10 = self.fuse_h1h2(torch.cat((h1s1_slice[2], h1s2_slice[3], h1s3_slice[4], h1s4_slice[5],
                                        h2s1_slice[2], h2s2_slice[3], h2s3_slice[4]), dim=1))
        '''

        if self.training:
            loss1 = bce2d(d1, input[1])
            loss2 = bce2d(d2, input[1])
            loss3 = bce2d(d3, input[1])
            loss4 = bce2d(d4, input[1])
            loss5 = bce2d(d5, input[1])
            # loss6 = bce2d(d6, input[1])
            # loss7 = bce2d(d7, input[1])
            # loss8 = bce2d(d8, input[1])
            # loss9 = bce2d(d9, input[1])
            # loss10 = bce2d(d10, input[1])
            # print('d1', d1.shape)
            # print('d2', d2.shape)
            # print('d3', d3.shape)
            # print('d4', d4.shape)
            # print('d5', d5.shape)
            # print('d6', d6.shape)
            # print('d7', d7.shape)
            # print('d8', d8.shape)
            # print('d9', d9.shape)
            # print('d10', d10.shape)
            # print(loss1)
            # print(loss2)
            # print(loss3)
            # print(loss4)
            # print(loss5)
            # print(loss6)
            # print(loss7)
            # print(loss8)
            # print(loss9)
            # print(loss10)
            # print()
            # return loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10
            return loss1 + loss2 + loss3 + loss4 + loss5

        else:
            d1 = torch.sigmoid(d1)
            d2 = torch.sigmoid(d2)
            d3 = torch.sigmoid(d3)
            d4 = torch.sigmoid(d4)
            d5 = torch.sigmoid(d5)
        return d5

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
        c = []
        for i in range(length):
            c.append(data[:, i, :, :].unsqueeze(0))
        return tuple(c)

    def parameters(net, lr):
        parameters = [
            {'params': net.fuse.weight, 'lr': lr * 0.01},
            {'params': net.fuse.bias, 'lr': lr * 0.02, 'weight_decay': 0},
            {'params': net.s1_1.weight, 'lr': lr * 0.1},
            {'params': net.s1_1.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.s1_2.weight, 'lr': lr * 0.1},
            {'params': net.s1_2.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.s2_1_top.weight, 'lr': lr * 0.1},
            {'params': net.s2_1_top.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.s2_2_top.weight, 'lr': lr * 0.1},
            {'params': net.s2_2_top.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.s2_1_down.weight, 'lr': lr * 0.1},
            {'params': net.s2_1_down.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.s2_2_down.weight, 'lr': lr * 0.1},
            {'params': net.s2_2_down.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.s3_1_top.weight, 'lr': lr * 0.1},
            {'params': net.s3_1_top.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.s3_2_top.weight, 'lr': lr * 0.1},
            {'params': net.s3_2_top.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.s3_3_top.weight, 'lr': lr * 0.1},
            {'params': net.s3_3_top.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.s3_1_down.weight, 'lr': lr * 0.1},
            {'params': net.s3_1_down.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.s3_2_down.weight, 'lr': lr * 0.1},
            {'params': net.s3_2_down.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.s3_3_down.weight, 'lr': lr * 0.1},
            {'params': net.s3_3_down.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.s4_1_top.weight, 'lr': lr * 0.1},
            {'params': net.s4_1_top.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.s4_2_top.weight, 'lr': lr * 0.1},
            {'params': net.s4_2_top.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.s4_3_top.weight, 'lr': lr * 0.1},
            {'params': net.s4_3_top.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.s4_1_down.weight, 'lr': lr * 0.1},
            {'params': net.s4_1_down.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.s4_2_down.weight, 'lr': lr * 0.1},
            {'params': net.s4_2_down.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.s4_3_down.weight, 'lr': lr * 0.1},
            {'params': net.s4_3_down.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.s5_1.weight, 'lr': lr * 0.1},
            {'params': net.s5_1.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.s5_2.weight, 'lr': lr * 0.1},
            {'params': net.s5_2.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.s5_3.weight, 'lr': lr * 0.1},
            {'params': net.s5_3.bias, 'lr': lr * 0.2, 'weight_decay': 0},

            {'params': net.h1s1_11to11.weight, 'lr': lr * 0.1},
            {'params': net.h1s1_11to11.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.h1s2_11to11_top.weight, 'lr': lr * 0.1},
            {'params': net.h1s2_11to11_top.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.h1s2_11to11_down.weight, 'lr': lr * 0.1},
            {'params': net.h1s2_11to11_down.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.h1s3_11to11_top.weight, 'lr': lr * 0.1},
            {'params': net.h1s3_11to11_top.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.h1s3_11to11_down.weight, 'lr': lr * 0.1},
            {'params': net.h1s3_11to11_down.bias, 'lr': lr * 0.2, 'weight_decay': 0},
            {'params': net.h1s4_11to11.weight, 'lr': lr * 0.1},
            {'params': net.h1s4_11to11.bias, 'lr': lr * 0.2, 'weight_decay': 0},

            {'params': net.h1s1_11to3.weight, 'lr': lr * 0.01},
            {'params': net.h1s1_11to3.bias, 'lr': lr * 0.02, 'weight_decay': 0},
            {'params': net.h1s2_11to4.weight, 'lr': lr * 0.01},
            {'params': net.h1s2_11to4.bias, 'lr': lr * 0.02, 'weight_decay': 0},
            {'params': net.h1s3_11to5.weight, 'lr': lr * 0.01},
            {'params': net.h1s3_11to5.bias, 'lr': lr * 0.02, 'weight_decay': 0},
            {'params': net.h1s4_11to6.weight, 'lr': lr * 0.01},
            {'params': net.h1s4_11to6.bias, 'lr': lr * 0.02, 'weight_decay': 0},
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

    def init_parameters(self):
        nn.init.kaiming_normal_(self.s1_1.weight)
        self.s1_1.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.s1_2.weight)
        self.s1_2.bias.data.fill_(0)

        nn.init.kaiming_normal_(self.s2_1_top.weight)
        self.s2_1_top.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.s2_2_top.weight)
        self.s2_2_top.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.s2_1_down.weight)
        self.s2_1_down.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.s2_2_down.weight)
        self.s2_2_down.bias.data.fill_(0)

        nn.init.kaiming_normal_(self.s3_1_top.weight)
        self.s3_1_top.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.s3_2_top.weight)
        self.s3_2_top.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.s3_3_top.weight)
        self.s3_3_top.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.s3_1_down.weight)
        self.s3_1_down.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.s3_2_down.weight)
        self.s3_2_down.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.s3_3_down.weight)
        self.s3_3_down.bias.data.fill_(0)

        nn.init.kaiming_normal_(self.s4_1_top.weight)
        self.s4_1_top.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.s4_2_top.weight)
        self.s4_2_top.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.s4_3_top.weight)
        self.s4_3_top.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.s4_1_down.weight)
        self.s4_1_down.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.s4_2_down.weight)
        self.s4_2_down.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.s4_3_down.weight)
        self.s4_3_down.bias.data.fill_(0)

        nn.init.kaiming_normal_(self.s5_1.weight)
        self.s5_1.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.s5_2.weight)
        self.s5_2.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.s5_3.weight)
        self.s5_3.bias.data.fill_(0)

        nn.init.kaiming_normal_(self.h1s1_11to11.weight)
        self.h1s1_11to11.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.h1s2_11to11_top.weight)
        self.h1s2_11to11_top.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.h1s2_11to11_down.weight)
        self.h1s2_11to11_down.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.h1s3_11to11_top.weight)
        self.h1s3_11to11_top.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.h1s3_11to11_down.weight)
        self.h1s3_11to11_down.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.h1s4_11to11.weight)
        self.h1s4_11to11.bias.data.fill_(0)

        nn.init.normal_(self.h1s1_11to3.weight, std=0.01)
        self.h1s1_11to3.bias.data.fill_(0)
        nn.init.normal_(self.h1s2_11to4.weight, std=0.01)
        self.h1s2_11to4.bias.data.fill_(0)
        nn.init.normal_(self.h1s3_11to5.weight, std=0.01)
        self.h1s3_11to5.bias.data.fill_(0)
        nn.init.normal_(self.h1s4_11to6.weight, std=0.01)
        self.h1s4_11to6.bias.data.fill_(0)

