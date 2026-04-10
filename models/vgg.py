from torch import nn


class VGG(nn.Module):
    def __init__(self, layers, init_weights=True, **kwargs):
        super(VGG, self).__init__()
        self.layer1 = layers[0]
        self.layer2 = layers[1]
        self.layer3 = layers[2]
        self.layer4 = layers[3]
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1, x2, x3, x4

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# ------------------------------------------------------------------------------
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    layer = []
    for v in cfg:
        if v == 'M':
            layers.append(nn.Sequential(*layer))
            layer = [nn.MaxPool3d(kernel_size=2, stride=2)]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layer.extend([conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)])
            else:
                layer.extend([conv3d, nn.ReLU(inplace=True)])

            in_channels = v

    return layers


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


# ------------------------------------------------------------------------------

def _vgg(arch, cfg, batch_norm, **kwargs):
    return VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)


# ------------------------------------------------------------------------------

def vgg11(**kwargs):
    return _vgg('vgg11', 'A', False, **kwargs)


def vgg11_bn(**kwargs):
    return _vgg('vgg11_bn', 'A', True, **kwargs)


def vgg13(**kwargs):
    return _vgg('vgg13', 'B', False, **kwargs)


def vgg13_bn(**kwargs):
    return _vgg('vgg13_bn', 'B', True, **kwargs)


def vgg16(**kwargs):
    return _vgg('vgg16', 'D', False, **kwargs)


class vgg16_bn(nn.Module):
    def __init__(self, **kwargs):
        super(vgg16_bn, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, dilation=1),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, dilation=1),
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, dilation=1),
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]


def vgg19(**kwargs):
    return _vgg('vgg19', 'E', False, **kwargs)


def vgg19_bn(**kwargs):
    return _vgg('vgg19_bn', 'E', True, **kwargs)


if __name__ == "__main__":
    model = vgg16_bn()
    print(model)