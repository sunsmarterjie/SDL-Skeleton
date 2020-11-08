from Ada_LSN.operations import *
import torch.nn.functional as F
from modules.binary_cross_entropy import bce2d as b1
from Ada_LSN.create_inception import Inception3 as network
# from Ada_LSN.create_mobilenet import mobilenetv3_large as network
# from Ada_LSN.create_res2net import res2net50_v1b as network
# from Ada_LSN.create_resnest import resnest50 as network
# from Ada_LSN.create_resnet import resnet50 as network
# from Ada_LSN.create_vgg import VGGfs as network


class Cell(nn.Module):
    def __init__(self, genotype, C_prev, C_curr, flag=0):
        super(Cell, self).__init__()
        self.flag = flag
        self.C_prev = C_prev
        self.C_curr = C_curr
        self.up2 = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        op_names, indices = zip(*genotype)
        concat = range(0, len(op_names) + 1)

        self._compile(C_prev, op_names, indices, concat)

    def _compile(self, C_prev, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._steps = len(op_names)
        self._concat = concat

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            op = OPS[name](C_prev, stride=1)
            self._ops += [op]
        self._indices = indices

    def forward(self, s):
        states = [s]
        for i in range(self._steps):
            h = states[self._indices[i]]
            op = self._ops[i]
            h = op(h)
            states += [h]
        s = sum(states[i] for i in self._concat)
        if self.flag == 1:
            s = self.up2(s)
        return s


class Network(nn.Module):
    def __init__(self, C, numu_layers, u_layers, genotype, pretrained_model=None):
        super(Network, self).__init__()
        self.network = network()
        if pretrained_model is not None:
            print("Loading pretrained weights from %s" % (pretrained_model))
            state_dict = torch.load(pretrained_model)
            self.network.load_state_dict({k: v for k, v in state_dict.items() if k in self.network.state_dict()})

        self.genotype = genotype
        self.u_layers = u_layers
        C_prev, C_curr = C, C
        self.ucells = nn.ModuleList()
        for i in range(numu_layers):
            if i in u_layers:
                C_prev, C_curr = C_curr, C_curr
                flag = 1
            else:
                C_prev, C_curr = C_curr, C_curr
                flag = 0
            cell = Cell(genotype[1][i], C_prev, C_curr, flag=flag)  # Unit is Cell
            self.ucells += [cell]
        self.fuse_cell = Cell(genotype[2][6], C_curr, C_curr, flag=1)

        # dsn， change here if backbone changes
        self.dsn1 = nn.Conv2d(64, C, 1)
        self.dsn2 = nn.Conv2d(192, C, 1)
        self.dsn3 = nn.Conv2d(288, C, 1)
        self.dsn4 = nn.Conv2d(768, C, 1)
        self.dsn5 = nn.Conv2d(2048, C, 1)
        # end dsn
        # new add
        self.cat1 = nn.Conv2d(C * 2, C, 1)
        self.cat2 = nn.Conv2d(C * 2, C, 1)
        self.cat3 = nn.Conv2d(C * 2, C, 1)
        self.cat4 = nn.Conv2d(C * 2, C, 1)
        self.cat5 = nn.Conv2d(C * 2, C, 1)
        self.cat6 = nn.Conv2d(C * 2, C, 1)
        self.cat7 = nn.Conv2d(C * 2, C, 1)
        self.cat8 = nn.Conv2d(C * 2, C, 1)
        self.cat9 = nn.Conv2d(C * 2, C, 1)
        self.cat10 = nn.Conv2d(C * 2, C, 1)
        self.cat11 = nn.Conv2d(C * 2, C, 1)
        self.cat12 = nn.Conv2d(C * 2, C, 1)
        self.cat13 = nn.Conv2d(C * 2, C, 1)
        self.cat14 = nn.Conv2d(C * 2, C, 1)
        self.cat15 = nn.Conv2d(C * 2, C, 1)
        # cat end
        # super
        self.super1 = nn.Conv2d(C, 1, 1)
        self.super2 = nn.Conv2d(C, 1, 1)
        self.super3 = nn.Conv2d(C, 1, 1)
        self.super4 = nn.Conv2d(C, 1, 1)
        self.super5 = nn.Conv2d(C, 1, 1)
        # super end
        self.classifier1 = nn.Conv2d(C_curr, 1, 1)
        self.classifier2 = nn.Conv2d(1, 1, kernel_size=15, stride=1, padding=7)

        self.init_parameters()

        self.up2 = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        self.up4 = lambda x: F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        self.up8 = lambda x: F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)

    def forward(self, input0, input1):
        size = input0.size()[2:4]
        conv1, conv2, conv3, conv4, conv5 = self.network(input0)
        dsn1 = self.dsn1(conv1)
        dsn2 = self.dsn2(conv2)
        dsn3 = self.dsn3(conv3)
        dsn4 = self.dsn4(conv4)
        dsn5 = self.dsn5(conv5)

        cell1, cell2, cell3, cell4, cell5 = self.ucells[0], self.ucells[1], self.ucells[2], self.ucells[3], self.ucells[
            4]

        # cell1 & stage5
        if self.genotype[0][0] == 1:
            c1 = cell1(dsn5)
        else:
            c1 = torch.zeros_like(self.up2(dsn5))

        # cell2 & stage4
        if self.genotype[0][1] == 1:
            d4_2 = dsn4
            if self.genotype[2][0][0] == 1:
                d4_2 = F.relu(self.cat1(torch.cat([c1, self.crop(d4_2, (0, 0) + c1.size()[2:4])], dim=1)),
                              inplace=True)
            c2 = cell2(d4_2)
        else:
            c2 = torch.zeros_like(self.up2(dsn4))

        # cell3 & stage3
        if self.genotype[0][2] == 1:
            d3_2 = dsn3
            if self.genotype[2][1][0] == 1:
                d3_2 = F.relu(
                    self.cat2(torch.cat([self.up2(c1), self.crop(d3_2, (0, 0) + self.up2(c1).size()[2:4])], dim=1)),
                    inplace=True)
            if self.genotype[2][1][1] == 1:
                d3_2 = F.relu(self.cat3(torch.cat([c2, self.crop(d3_2, (0, 0) + c2.size()[2:4])], dim=1)),
                              inplace=True)
            c3 = cell3(d3_2)
        else:
            c3 = torch.zeros_like(self.up2(dsn3))

        # cell4 & stage2
        if self.genotype[0][3] == 1:
            d2_2 = dsn2
            if self.genotype[2][2][0] == 1:
                d2_2 = F.relu(
                    self.cat5(torch.cat([self.up2(c2), self.crop(d2_2, (0, 0) + self.up2(c2).size()[2:4])], dim=1)),
                    inplace=True)
            if self.genotype[2][2][1] == 1:
                d2_2 = F.relu(self.cat6(torch.cat([c3, self.crop(d2_2, (0, 0) + c3.size()[2:4])], dim=1)),
                              inplace=True)
            c4 = cell3(d2_2)
        else:
            c4 = torch.zeros_like(self.up2(dsn2))

        # cell5 & stage1
        if self.genotype[0][4] == 1:
            d1_2 = dsn1
            if self.genotype[2][3][0] == 1:
                d1_2 = F.relu(
                    self.cat9(torch.cat([self.up2(c3), self.crop(d1_2, (0, 0) + self.up2(c3).size()[2:4])], dim=1)),
                    inplace=True)
            if self.genotype[2][3][1] == 1:
                d1_2 = F.relu(self.cat10(torch.cat([c4, self.crop(d1_2, (0, 0) + c4.size()[2:4])], dim=1)),
                              inplace=True)
            c5 = cell5(d1_2)
        else:
            c5 = torch.zeros_like(dsn1)

        # fuse cell
        # scale in 1/2 of input, that is, scale of c3
        if self.genotype[0][2] == 1:
            d_fuse = torch.zeros_like(c3)
        elif self.genotype[0][1] == 1:
            d_fuse = torch.zeros_like(self.up2(c2))
        else:
            d_fuse = torch.zeros_like(self.up4(c1))

        if self.genotype[2][4][0] == 1:
            d_fuse = d_fuse + self.crop(self.up4(c1), (0, 0) + d_fuse.size()[2:4])
        if self.genotype[2][4][1] == 1:
            d_fuse = d_fuse + self.crop(self.up2(c2), (0, 0) + d_fuse.size()[2:4])
        if self.genotype[2][4][2] == 1:
            d_fuse = d_fuse + self.crop(c3, (0, 0) + d_fuse.size()[2:4])
        if self.genotype[2][4][3] == 1:
            p1 = F.max_pool2d(c4, 2, 2, ceil_mode=True)
            d_fuse = self.crop(d_fuse, (0, 0) + p1.size()[2:4]) + p1
        if self.genotype[2][4][4] == 1:
            p2 = F.max_pool2d(c5, 2, 2, ceil_mode=True)
            d_fuse = self.crop(d_fuse, (0, 0) + p2.size()[2:4]) + p2
        d_fuse = self.fuse_cell(d_fuse)

        s = self.classifier2(self.classifier1(d_fuse))
        out_fuse = self.crop(s, (34, 34) + size)
        if self.genotype[2][5][0] == 1:
            out1 = self.crop(self.super1(self.up8(c1)), (34, 34) + size)
        if self.genotype[2][5][1] == 1:
            out2 = self.crop(self.super2(self.up4(c2)), (34, 34) + size)
        if self.genotype[2][5][2] == 1:
            out3 = self.crop(self.super3(self.up2(c3)), (34, 34) + size)
        if self.genotype[2][5][3] == 1:
            out4 = self.crop(self.super4(c4), (34, 34) + size)
        if self.genotype[2][5][4] == 1:
            out5 = self.crop(self.super5(c5), (34, 34) + size)

        loss_fuse = b1(out_fuse, input1)
        loss = 0.0
        if self.genotype[2][5][0] == 1 and self.genotype[0][0]:
            loss += b1(out1, input1)
        if self.genotype[2][5][1] == 1 and self.genotype[0][1]:
            loss += b1(out2, input1)
        if self.genotype[2][5][2] == 1 and self.genotype[0][2]:
            loss += b1(out3, input1)
        if self.genotype[2][5][3] == 1 and self.genotype[0][3]:
            loss += b1(out4, input1)
        if self.genotype[2][5][4] == 1 and self.genotype[0][4]:
            loss += b1(out5, input1)
        return loss + loss_fuse, loss_fuse  # for train
        # if input1 is not None:
        #     loss_fuse = b1(out_fuse, input1)
        # else:
        #     loss_fuse = None
        # out = torch.sigmoid(out_fuse)
        # return out, loss_fuse   # for test

    def crop(self, d, region):  # use for crop the keep to input data
        x, y, h, w = region
        d1 = d[:, :, x:x + h, y:y + w]
        return d1

    def center_crop(self, d, size):
        d_h, d_w = d.size()[-2:]
        g_h, g_w = size[0], size[1]
        d1 = d[:, :, int(round((d_h - g_h) / 2)):int(round((d_h - g_h) / 2) + g_h),
             int(round((d_w - g_w) / 2)):int(round((d_w - g_w) / 2) + g_w)]
        return d1

    def init_parameters(self):
        nn.init.xavier_normal_(self.classifier1.weight)
        torch.nn.init.normal_(self.cat1.weight, std=0.01)
        torch.nn.init.normal_(self.cat2.weight, std=0.01)
        torch.nn.init.normal_(self.cat3.weight, std=0.01)
        torch.nn.init.normal_(self.cat4.weight, std=0.01)
        torch.nn.init.normal_(self.cat5.weight, std=0.01)
        torch.nn.init.normal_(self.cat6.weight, std=0.01)
        torch.nn.init.normal_(self.cat7.weight, std=0.01)
        torch.nn.init.normal_(self.cat8.weight, std=0.01)
        torch.nn.init.normal_(self.cat9.weight, std=0.01)
        torch.nn.init.normal_(self.cat10.weight, std=0.01)
        torch.nn.init.normal_(self.cat11.weight, std=0.01)
        torch.nn.init.normal_(self.cat12.weight, std=0.01)
        torch.nn.init.normal_(self.cat13.weight, std=0.01)
        torch.nn.init.normal_(self.cat14.weight, std=0.01)
        torch.nn.init.normal_(self.cat15.weight, std=0.01)
        torch.nn.init.normal_(self.dsn1.weight, std=0.0001)
        torch.nn.init.normal_(self.dsn2.weight, std=0.0001)
        torch.nn.init.normal_(self.dsn3.weight, std=0.0001)
        torch.nn.init.normal_(self.dsn4.weight, std=0.001)
        torch.nn.init.normal_(self.dsn5.weight, std=0.01)
        self.classifier2.weight.data.fill_(0)

        self.super1.weight.data.fill_(0)
        self.super2.weight.data.fill_(0)
        self.super3.weight.data.fill_(0)
        self.super4.weight.data.fill_(0)
        self.super5.weight.data.fill_(0)

        self.cat1.bias.data.fill_(0)
        self.cat2.bias.data.fill_(0)
        self.cat3.bias.data.fill_(0)
        self.cat4.bias.data.fill_(0)
        self.cat5.bias.data.fill_(0)
        self.cat6.bias.data.fill_(0)
        self.cat7.bias.data.fill_(0)
        self.cat8.bias.data.fill_(0)
        self.cat9.bias.data.fill_(0)
        self.cat10.bias.data.fill_(0)
        self.cat11.bias.data.fill_(0)
        self.cat12.bias.data.fill_(0)
        self.cat13.bias.data.fill_(0)
        self.cat14.bias.data.fill_(0)
        self.cat15.bias.data.fill_(0)
        self.dsn1.bias.data.fill_(0)
        self.dsn2.bias.data.fill_(0)
        self.dsn3.bias.data.fill_(0)
        self.dsn4.bias.data.fill_(0)
        self.dsn5.bias.data.fill_(0)
        self.super1.bias.data.fill_(0)
        self.super2.bias.data.fill_(0)
        self.super3.bias.data.fill_(0)
        self.super4.bias.data.fill_(0)
        self.super5.bias.data.fill_(0)
        self.classifier1.bias.data.fill_(0)
        self.classifier2.bias.data.fill_(0)

    def parameter(net, lr):
        lr_c = lr * 100
        lr_d = lr * 20
        parameters = [
            # super begin
            {'params': net.super1.weight, 'lr': lr_d},
            {'params': net.super1.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.super2.weight, 'lr': lr_d},
            {'params': net.super2.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.super3.weight, 'lr': lr_d},
            {'params': net.super3.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.super4.weight, 'lr': lr_d},
            {'params': net.super4.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.super5.weight, 'lr': lr_d},
            {'params': net.super5.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            # super end
            # cat begin
            {'params': net.cat1.weight, 'lr': lr_d},
            {'params': net.cat1.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.cat2.weight, 'lr': lr_d},
            {'params': net.cat2.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.cat3.weight, 'lr': lr_d},
            {'params': net.cat3.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.cat4.weight, 'lr': lr_d},
            {'params': net.cat4.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.cat5.weight, 'lr': lr_d},
            {'params': net.cat5.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.cat6.weight, 'lr': lr_d},
            {'params': net.cat6.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.cat7.weight, 'lr': lr_d},
            {'params': net.cat7.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.cat8.weight, 'lr': lr_d},
            {'params': net.cat8.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.cat9.weight, 'lr': lr_d},
            {'params': net.cat9.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.cat10.weight, 'lr': lr_d},
            {'params': net.cat10.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.cat11.weight, 'lr': lr_d},
            {'params': net.cat11.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.cat12.weight, 'lr': lr_d},
            {'params': net.cat12.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.cat13.weight, 'lr': lr_d},
            {'params': net.cat13.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.cat14.weight, 'lr': lr_d},
            {'params': net.cat14.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.cat15.weight, 'lr': lr_d},
            {'params': net.cat15.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            # cat end
            # dsn begin
            {'params': net.dsn1.weight, 'lr': lr_d},
            {'params': net.dsn1.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.dsn2.weight, 'lr': lr_d},
            {'params': net.dsn2.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.dsn3.weight, 'lr': lr_d},
            {'params': net.dsn3.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.dsn4.weight, 'lr': lr_d},
            {'params': net.dsn4.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.dsn5.weight, 'lr': lr_d},
            {'params': net.dsn5.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            # dsn end
            {'params': net.classifier1.weight, 'lr': lr_d},
            {'params': net.classifier1.bias, 'lr': lr_d * 2, 'weight_decay': 0},
            {'params': net.classifier2.weight, 'lr': lr_d},
            {'params': net.classifier2.bias, 'lr': lr_d * 2, 'weight_decay': 0},
        ]
        for i, layer in enumerate(net.network.modules()):
            if isinstance(layer, nn.Conv2d):
                parameters.extend([{'params': layer.weight, 'lr': lr_c}])
                if layer.bias is not None:
                    parameters.extend([{'params': layer.bias, 'lr': lr_c * 2, 'weight_decay': 0}])
            if isinstance(layer, nn.BatchNorm2d):
                parameters.extend([{'params': layer.running_mean, 'lr': 0},
                                   {'params': layer.running_var, 'lr': 0}])
        for i, layer in enumerate(net.ucells.modules()):
            if isinstance(layer, nn.Conv2d):
                parameters.extend(([{'params': layer.weight, 'lr': lr_c},
                                    {'params': layer.bias, 'lr': lr_c * 2, 'weight_decay': 0}]))
        for i, layer in enumerate(net.fuse_cell.modules()):
            if isinstance(layer, nn.Conv2d):
                parameters.extend(([{'params': layer.weight, 'lr': lr_c},
                                    {'params': layer.bias, 'lr': lr_c * 2, 'weight_decay': 0}]))
        return parameters
