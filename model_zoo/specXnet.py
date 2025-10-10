import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from .spectral import FFC_BN_ACT, FFCSE_block, match_size

__all__ = ['ffc_xception']

model_urls = {
    'xception': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth'
}

# --- Helpers for tuple operations ---

class TupleReLU(nn.Module):
    def __init__(self, inplace=False):
        super(TupleReLU, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)
    def forward(self, x):
        if isinstance(x, tuple):
            return (self.relu(x[0]), self.relu(x[1]))
        else:
            return self.relu(x)

class TupleMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(TupleMaxPool2d, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride, padding)
    def forward(self, x):
        if isinstance(x, tuple):
            return (self.pool(x[0]), self.pool(x[1]))
        else:
            return self.pool(x)

class FFCSeparableConv2d(nn.Module):
    """
    An FFC-enabled depthwise separable convolution.
    It applies an FFC-based depthwise conv followed by an FFC-based pointwise conv.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                 ratio_gin=0.5, ratio_gout=0.5, lfu=True):
        super(FFCSeparableConv2d, self).__init__()
        # Compute number of local channels (used as groups for depthwise conv).
        local_in_channels = int(in_channels * (1 - ratio_gin))
        if local_in_channels < 1:
            local_in_channels = in_channels
        self.dw = FFC_BN_ACT(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                             padding=padding, groups=local_in_channels,
                             ratio_gin=ratio_gin, ratio_gout=ratio_gout, enable_lfu=lfu)
        self.pw = FFC_BN_ACT(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                             ratio_gin=ratio_gout, ratio_gout=ratio_gout, enable_lfu=lfu)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x

class FFCBlock(nn.Module):
    """
    An FFC-enabled version of the Xception block.
    Replaces SeparableConv2d with FFCSeparableConv2d and uses an FFC skip connection.
    All layers propagate a tuple (local, global); if a global branch is missing, zeros are used.
    """
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True,
                 grow_first=True, ratio_gin=0.5, ratio_gout=0.5, lfu=True):
        super(FFCBlock, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = FFC_BN_ACT(in_filters, out_filters, kernel_size=1, stride=strides,
                                   padding=0, ratio_gin=ratio_gin, ratio_gout=ratio_gout, enable_lfu=lfu)
        else:
            self.skip = None

        layers = []
        if grow_first:
            layers.append(TupleReLU(inplace=False))
            layers.append(FFCSeparableConv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1,
                                             bias=False, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu))
            filters = out_filters
        else:
            filters = in_filters

        for i in range(reps - 1):
            layers.append(TupleReLU(inplace=False))
            layers.append(FFCSeparableConv2d(filters, filters, kernel_size=3, stride=1, padding=1,
                                             bias=False, ratio_gin=ratio_gout, ratio_gout=ratio_gout, lfu=lfu))

        if not grow_first:
            layers.append(TupleReLU(inplace=False))
            layers.append(FFCSeparableConv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1,
                                             bias=False, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu))

        if strides != 1:
            layers.append(TupleMaxPool2d(3, stride=strides, padding=1))
        self.rep = nn.Sequential(*layers)
    def forward(self, inp):
        if not isinstance(inp, tuple):
            inp = (inp, torch.zeros_like(inp))
        out = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
        else:
            skip = inp
        # Crop outputs and skip connections to matching spatial sizes before addition.
        out_local, skip_local = match_size(out[0], skip[0])
        out_global, skip_global = match_size(out[1], skip[1])
        out_local = F.relu(out_local + skip_local, inplace=False)
        out_global = F.relu(out_global + skip_global, inplace=False)
        return (out_local, out_global)

class Xception(nn.Module):
    """
    FFC Xception optimized for ImageNet.
    This network uses FFC-based separable convolutions and blocks.
    """
    def __init__(self, num_classes=2, ratio_gin=0.5, ratio_gout=0.5, lfu=True):
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.ratio = ratio_gin

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = FFCBlock(64, 128, reps=2, strides=2, start_with_relu=False, grow_first=True,
                               ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)
        self.block2 = FFCBlock(128, 256, reps=2, strides=2, start_with_relu=True, grow_first=True,
                               ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)
        self.block3 = FFCBlock(256, 728, reps=2, strides=2, start_with_relu=True, grow_first=True,
                               ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)

        self.block4 = FFCBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True,
                               ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)
        self.block5 = FFCBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True,
                               ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)
        self.block6 = FFCBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True,
                               ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)
        self.block7 = FFCBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True,
                               ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)

        self.block8 = FFCBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True,
                               ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)
        self.block9 = FFCBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True,
                               ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)
        self.block10 = FFCBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)
        self.block11 = FFCBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)

        self.block12 = FFCBlock(728, 1024, reps=2, strides=2, start_with_relu=True, grow_first=False,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)

        # With ratio=0.5, conv3 will output local branch with 1536*(1-0.5)=768 channels.
        self.conv3 = FFCSeparableConv2d(1024, 1536, kernel_size=3, stride=1, padding=1,
                                        bias=False, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)
        self.bn3 = nn.BatchNorm2d(768)  # update to 768 for local branch

        # With ratio=0.5, conv4 will output local branch with 2048*(1-0.5)=1024 channels.
        self.conv4 = FFCSeparableConv2d(1536, 2048, kernel_size=3, stride=1, padding=1,
                                        bias=False, ratio_gin=ratio_gin, ratio_gout=ratio_gout, lfu=lfu)
        self.bn4 = nn.BatchNorm2d(1024)  # update to 1024 for local branch

        self.fc = nn.Linear(1024, num_classes)  # update to 1024

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        _, _, H, W = x.shape
        if H % 2 == 1 or W % 2 == 1:
            new_H = H - (H % 2)
            new_W = W - (W % 2)
            x = x[:, :, :new_H, :new_W]

        local_channels = int(x.size(1) * (1 - self.ratio))
        if local_channels == 0 or local_channels == x.size(1):
            x = (x, torch.zeros_like(x))
        else:
            x = (x[:, :local_channels, ...], x[:, local_channels:, ...])

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x_local = x[0]
        # conv3 returns a tuple; take the local branch.
        x_local = self.conv3((x_local, torch.zeros_like(x_local)))[0]
        x_local = self.bn3(x_local)
        x_local = self.relu(x_local)

        x_local = self.conv4((x_local, torch.zeros_like(x_local)))[0]
        x_local = self.bn4(x_local)
        x_local = self.relu(x_local)

        x_local = F.adaptive_avg_pool2d(x_local, (1, 1))
        x_local = x_local.view(x_local.size(0), -1)
        x_local = self.fc(x_local)
        return x_local

def ffc_xception(pretrained=False, **kwargs):
    ratio = kwargs.pop('ratio', 0.5)
    kwargs.pop('use_se', None)
    model = Xception(ratio_gin=ratio, ratio_gout=ratio, **kwargs)
    if pretrained:
        # Optionally load pretrained weights.
        pass
    return model
