import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, output_dim=72, mode=0, **kwargs):

        super(DenseNet, self).__init__()
        self.mode = mode
        # First convolution

        self.f1 = nn.Conv3d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.f2 = nn.BatchNorm3d(num_init_features)
        self.f3 = nn.ReLU(inplace=True)
        self.f4 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        # Each denseblock
        num_features = num_init_features

        self.l1 = nn.Sequential()
        block = _DenseBlock(num_layers=block_config[0], num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.l1.add_module('denseblock%d' % (1), block)
        num_features = num_features + block_config[0] * growth_rate
        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        self.l1.add_module('transition%d' % (1), trans)
        num_features = num_features // 2

        self.l2 = nn.Sequential()
        block = _DenseBlock(num_layers=block_config[1], num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.l2.add_module('denseblock%d' % (2), block)
        num_features = num_features + block_config[1] * growth_rate
        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        self.l2.add_module('transition%d' % (2), trans)
        num_features = num_features // 2

        self.l3 = nn.Sequential()
        block = _DenseBlock(num_layers=block_config[2], num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.l3.add_module('denseblock%d' % (3), block)
        num_features = num_features + block_config[2] * growth_rate
        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        self.l3.add_module('transition%d' % (3), trans)
        num_features = num_features // 2

        self.l4 = nn.Sequential()
        block = _DenseBlock(num_layers=block_config[3], num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.l4.add_module('denseblock%d' % (4), block)
        num_features = num_features + block_config[3] * growth_rate

        # Final batch norm
        self.l4.add_module('norm5', nn.BatchNorm3d(num_features))

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, tem=1):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)

        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)

        return [x1, x2, x3, x4]


def densenet121(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    return model


def densenet169(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
    return model


def densenet201(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    return model


def densenet161(**kwargs):
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), **kwargs)
    return model
