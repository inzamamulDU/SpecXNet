import torch
import torch.nn as nn
from .ffc import FFCSE_block, S2NN_FFC_BN_ACT, FFC_BN_ACT

__all__ = ['FFCResNet', 'ffc_resnet18', 'ffc_resnet34',
           'ffc_resnet26', 'ffc_resnet50', 'ffc_resnet101',
           'ffc_resnet152', 'ffc_resnet200', 'ffc_resnext50_32x4d',
           'ffc_resnext101_32x8d']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, ratio_gin=0.5, ratio_gout=0.5,
                 lfu=True, use_se=False, norm_layer=None, use_s2nn=False,
                 lif_tau=2.0, lif_threshold=1.0, lif_decay=0.99):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        width = int(planes * (base_width / 64.)) * groups
        # Use the spiking block if use_s2nn is True, otherwise use the standard one.
        conv_block = S2NN_FFC_BN_ACT if use_s2nn else FFC_BN_ACT
        self.conv1 = conv_block(inplanes, width, kernel_size=3, padding=1, stride=stride,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout,
                                norm_layer=norm_layer, enable_lfu=lfu,
                                lif_tau=lif_tau, lif_threshold=lif_threshold, lif_decay=lif_decay)
        self.conv2 = conv_block(width, planes * self.expansion, kernel_size=3, padding=1,
                                ratio_gin=ratio_gout, ratio_gout=ratio_gout,
                                norm_layer=norm_layer, enable_lfu=lfu,
                                lif_tau=lif_tau, lif_threshold=lif_threshold, lif_decay=lif_decay)
        self.se_block = FFCSE_block(planes * self.expansion, ratio_gout) if use_se else nn.Identity()
        self.relu_l = nn.Identity() if ratio_gout == 1 else nn.ReLU(inplace=True)
        self.relu_g = nn.Identity() if ratio_gout == 0 else nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x if self.downsample is None else self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x_l, x_g = self.se_block(x)
        x_l = self.relu_l(x_l + id_l)
        x_g = self.relu_g(x_g + id_g)
        return x_l, x_g

# Similarly, update Bottleneck to accept lif_tau, lif_threshold, lif_decay:
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, ratio_gin=0.5, ratio_gout=0.5,
                 lfu=True, use_se=False, use_s2nn=False,
                 lif_tau=2.0, lif_threshold=1.0, lif_decay=0.99):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        conv_block = S2NN_FFC_BN_ACT if use_s2nn else FFC_BN_ACT
        self.conv1 = conv_block(inplanes, width, kernel_size=1,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout,
                                enable_lfu=lfu,
                                lif_tau=lif_tau, lif_threshold=lif_threshold, lif_decay=lif_decay)
        self.conv2 = conv_block(width, width, kernel_size=3,
                                ratio_gin=ratio_gout, ratio_gout=ratio_gout,
                                stride=stride, padding=1, groups=groups,
                                enable_lfu=lfu,
                                lif_tau=lif_tau, lif_threshold=lif_threshold, lif_decay=lif_decay)
        self.conv3 = conv_block(width, planes * self.expansion, kernel_size=1,
                                ratio_gin=ratio_gout, ratio_gout=ratio_gout,
                                enable_lfu=lfu,
                                lif_tau=lif_tau, lif_threshold=lif_threshold, lif_decay=lif_decay)
        self.se_block = FFCSE_block(planes * self.expansion, ratio_gout) if use_se else nn.Identity()
        self.relu_l = nn.Identity() if ratio_gout == 1 else nn.ReLU(inplace=True)
        self.relu_g = nn.Identity() if ratio_gout == 0 else nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x if self.downsample is None else self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_l, x_g = self.se_block(x)
        x_l = self.relu_l(x_l + id_l)
        x_g = self.relu_g(x_g + id_g)
        return x_l, x_g


class FFCResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None, ratio=0.5, lfu=True,
                 use_se=False, use_s2nn=False, 
                 lif_tau=2.0, lif_threshold=1.0, lif_decay=0.99):
        super(FFCResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        inplanes = 64
        self.inplanes = inplanes
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.lfu = lfu
        self.use_se = use_se
        self.use_s2nn = use_s2nn
        # Save the spiking hyperparameters
        self.lif_tau = lif_tau
        self.lif_threshold = lif_threshold
        self.lif_decay = lif_decay

        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, inplanes * 1, layers[0],
                                        stride=1, ratio_gin=0, ratio_gout=ratio,
                                        use_s2nn=self.use_s2nn,
                                        lif_tau=self.lif_tau, lif_threshold=self.lif_threshold, lif_decay=self.lif_decay)
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1],
                                        stride=2, ratio_gin=ratio, ratio_gout=ratio,
                                        use_s2nn=self.use_s2nn,
                                        lif_tau=self.lif_tau, lif_threshold=self.lif_threshold, lif_decay=self.lif_decay)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2],
                                        stride=2, ratio_gin=ratio, ratio_gout=ratio,
                                        use_s2nn=self.use_s2nn,
                                        lif_tau=self.lif_tau, lif_threshold=self.lif_threshold, lif_decay=self.lif_decay)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3],
                                        stride=2, ratio_gin=ratio, ratio_gout=0,
                                        use_s2nn=self.use_s2nn,
                                        lif_tau=self.lif_tau, lif_threshold=self.lif_threshold, lif_decay=self.lif_decay)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inplanes * 8 * block.expansion, num_classes)

        # Weight initialization omitted for brevity…

    def _make_layer(self, block, planes, blocks, stride=1, ratio_gin=0.5, ratio_gout=0.5,
                    use_s2nn=False, lif_tau=2.0, lif_threshold=1.0, lif_decay=0.99):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or ratio_gin == 0:
            downsample = FFC_BN_ACT(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,
                                    ratio_gin=ratio_gin, ratio_gout=ratio_gout, enable_lfu=self.lfu)
        layers = []
        if block.__name__ == "BasicBlock":
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                                self.dilation, ratio_gin, ratio_gout, lfu=self.lfu, use_se=self.use_se,
                                norm_layer=norm_layer, use_s2nn=use_s2nn,
                                lif_tau=lif_tau, lif_threshold=lif_threshold, lif_decay=lif_decay))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                                self.dilation, ratio_gin, ratio_gout, lfu=self.lfu, use_se=self.use_se,
                                use_s2nn=use_s2nn,
                                lif_tau=lif_tau, lif_threshold=lif_threshold, lif_decay=lif_decay))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if block.__name__ == "BasicBlock":
                layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation,
                                    ratio_gin=ratio_gout, ratio_gout=ratio_gout, lfu=self.lfu, use_se=self.use_se,
                                    norm_layer=norm_layer, use_s2nn=use_s2nn,
                                    lif_tau=lif_tau, lif_threshold=lif_threshold, lif_decay=lif_decay))
            else:
                layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation,
                                    ratio_gin=ratio_gout, ratio_gout=ratio_gout, lfu=self.lfu, use_se=self.use_se,
                                    use_s2nn=use_s2nn,
                                    lif_tau=lif_tau, lif_threshold=lif_threshold, lif_decay=lif_decay))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # For simplicity, assume the local branch output is used for classification.
        x = self.avgpool(x[0])
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ffc_resnet18(pretrained=False, **kwargs):
    model = FFCResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def ffc_resnet34(pretrained=False, **kwargs):
    model = FFCResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def ffc_resnet26(pretrained=False, **kwargs):
    model = FFCResNet(Bottleneck, [2, 2, 2, 2], **kwargs)
    return model

def ffc_resnet50(pretrained=False, **kwargs):
    model = FFCResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def ffc_resnet101(pretrained=False, **kwargs):
    model = FFCResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def ffc_resnet152(pretrained=False, **kwargs):
    model = FFCResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def ffc_resnet200(pretrained=False, **kwargs):
    model = FFCResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

def ffc_resnext50_32x4d(pretrained=False, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    model = FFCResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def ffc_resnext101_32x8d(pretrained=False, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    model = FFCResNet(Bottleneck, [3, 4, 32, 3], **kwargs)
    return model
