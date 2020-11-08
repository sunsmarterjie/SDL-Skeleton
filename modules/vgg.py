from torch import nn

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'H': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
}


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
