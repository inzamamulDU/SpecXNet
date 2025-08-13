import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.base import MemoryModule

##########################################
# Original FFC Modules (unchanged)
##########################################

class FFCSE_block(nn.Module):
    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * self.sigmoid(self.conv_a2g(x))
        return x_l, x_g

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                    kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        # Compute real-valued FFT (using rfft along height dimension)
        ffted = torch.view_as_real(torch.fft.rfft(x, dim=2, norm="ortho"))
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()

        output = torch.fft.irfft(torch.view_as_complex(ffted), n=r_size[2], dim=2, norm="ortho")
        return output

class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)
        return output

class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)
        return out_xl, out_xg

class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 enable_lfu=True):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l = lnorm(int(out_channels * (1 - ratio_gout)))
        self.bn_g = gnorm(int(out_channels * ratio_gout))
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g

##########################################
# New Spiking Modules for S²NN
##########################################

class SpikeActivation(nn.Module):
    """
    A simple spike activation that outputs binary spikes.
    """
    def __init__(self, threshold=0.5):
        super(SpikeActivation, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        return (x > self.threshold).float()

class DSFA(nn.Module):
    """
    Dual Spike Fourier Attention: computes attention maps for local and global branches.
    """
    def __init__(self, in_channels):
        super(DSFA, self).__init__()
        self.attn_local = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.attn_global = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x_local, x_global):
        A_l = self.attn_local(x_local)
        A_g = self.attn_global(x_global)
        return A_l, A_g

# class S2NN_FFC_BN_ACT(nn.Module):
#     """
#     Spiking version of FFC_BN_ACT that integrates spike activation and DSFA-based fusion.
#     """
#     def __init__(self, in_channels, out_channels,
#                  kernel_size, ratio_gin, ratio_gout,
#                  stride=1, padding=0, dilation=1, groups=1, bias=False,
#                  norm_layer=nn.BatchNorm2d, activation_layer=SpikeActivation,
#                  enable_lfu=True):
#         super(S2NN_FFC_BN_ACT, self).__init__()
#         self.ffc = FFC(in_channels, out_channels, kernel_size,
#                        ratio_gin, ratio_gout, stride, padding, dilation,
#                        groups, bias, enable_lfu)
#         # BatchNorm for each branch
#         self.bn_l = nn.Identity() if ratio_gout == 1 else norm_layer(int(out_channels * (1 - ratio_gout)))
#         self.bn_g = nn.Identity() if ratio_gout == 0 else norm_layer(int(out_channels * ratio_gout))
#         # Use spiking activation instead of ReLU
#         self.spike_act = activation_layer()  
#         # DSFA to compute attention maps for local and global branches
#         self.dsfa = DSFA(out_channels)
#         # To fuse branches into a common channel space, we use 1x1 convolutions.
#         local_channels = int(out_channels * (1 - ratio_gout))
#         global_channels = int(out_channels * ratio_gout)
#         self.conv_local_fuse = nn.Conv2d(local_channels, out_channels, kernel_size=1, bias=False) if local_channels > 0 else nn.Identity()
#         self.conv_global_fuse = nn.Conv2d(global_channels, out_channels, kernel_size=1, bias=False) if global_channels > 0 else nn.Identity()

#     def forward(self, x):
#         # Obtain local and global branches from FFC
#         x_l, x_g = self.ffc(x)
#         # Apply BN and spike activation
#         x_l = self.spike_act(self.bn_l(x_l))
#         x_g = self.spike_act(self.bn_g(x_g)) if isinstance(x_g, torch.Tensor) else 0
#         # Compute DSFA attention weights
#         A_l, A_g = self.dsfa(x_l, x_g)
#         # Weight the branches
#         x_l = A_l * x_l
#         x_g = A_g * x_g
#         # Map branches to same channel dimension and fuse
#         x_l_f = self.conv_local_fuse(x_l) if isinstance(x_l, torch.Tensor) else 0
#         x_g_f = self.conv_global_fuse(x_g) if isinstance(x_g, torch.Tensor) else 0
#         out = x_l_f + x_g_f
#         return out

class AdaptiveLIFNode(MemoryModule):
    def __init__(self, tau=2.0, init_threshold=1.0, threshold_decay=0.99):
        """
        Adaptive LIF neuron integrated with SpikingJelly's MemoryModule.
        tau: membrane time constant.
        init_threshold: initial firing threshold.
        threshold_decay: multiplicative decay factor applied to the threshold each forward pass.
        """
        super().__init__()
        self.tau = tau
        self.threshold = nn.Parameter(torch.tensor(init_threshold))
        self.threshold_decay = threshold_decay
        self.v = None  # membrane potential

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initialize membrane potential if necessary
        if self.v is None or self.v.size() != x.size():
            self.v = torch.zeros_like(x)
        
        # Update membrane potential using simple Euler integration
        self.v = self.v + (x - self.v) / self.tau
        
        # Apply adaptive threshold decay
        current_threshold = self.threshold * self.threshold_decay
        
        # Generate spike: fire when membrane potential exceeds threshold
        spike = (self.v >= current_threshold).float()
        
        # Reset membrane potential for neurons that fired
        self.v = self.v * (1 - spike)
        
        # Update threshold parameter (here, simply decaying it)
        self.threshold.data = current_threshold
        
        return spike

    def reset(self):
        # Reset internal state (membrane potential)
        self.v = None
    
class S2NN_FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, enable_lfu=True,
                 lif_tau=2.0, lif_threshold=1.0, lif_decay=0.99):
        super(S2NN_FFC_BN_ACT, self).__init__()
        # Compute split parameters for local (cl) and global (cg) branches.
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        self.out_cl = out_cl
        self.out_cg = out_cg

        # --- Create Local Branch Layers ---
        if out_cl > 0:
            self.convl2l = nn.Conv2d(in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias)
            self.convg2l = nn.Conv2d(in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias) if in_cg > 0 else None
        else:
            self.convl2l = None
            self.convg2l = None

        # --- Create Global Branch Layers ---
        if out_cg > 0:
            self.convl2g = nn.Conv2d(in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias)
            from .ffc import SpectralTransform  # ensure correct import
            self.convg2g = SpectralTransform(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu) if in_cg > 0 else None
        else:
            self.convl2g = None
            self.convg2g = None

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        # Setup BatchNorm layers
        self.bn_l = nn.Identity() if out_cl == 0 else norm_layer(out_cl)
        self.bn_g = nn.Identity() if out_cg == 0 else norm_layer(out_cg)

        # Use adaptive LIF neurons for spiking activation.
        self.act_l = AdaptiveLIFNode(tau=lif_tau, init_threshold=lif_threshold, threshold_decay=lif_decay)
        self.act_g = AdaptiveLIFNode(tau=lif_tau, init_threshold=lif_threshold, threshold_decay=lif_decay)

    def forward(self, x):
        if isinstance(x, tuple):
            x_l, x_g = x
        else:
            x_l, x_g = x, None

        # Local branch
        if self.convl2l is not None:
            out_l = self.convl2l(x_l)
        else:
            out_l = 0
        if self.convg2l is not None and (x_g is not None):
            out_g = self.convg2l(x_g)
        else:
            out_g = 0
        if isinstance(out_l, torch.Tensor) and isinstance(out_g, torch.Tensor) and out_l.shape[-2:] != out_g.shape[-2:]:
            target_size = (max(out_l.shape[-2], out_g.shape[-2]),
                           max(out_l.shape[-1], out_g.shape[-1]))
            out_l = F.interpolate(out_l, size=target_size, mode='bilinear', align_corners=False)
            out_g = F.interpolate(out_g, size=target_size, mode='bilinear', align_corners=False)
        local_out = (out_l if isinstance(out_l, torch.Tensor) else 0) + (out_g if isinstance(out_g, torch.Tensor) else 0)

        # Global branch
        if self.convl2g is not None:
            out_l2 = self.convl2g(x_l)
        else:
            out_l2 = 0
        if self.convg2g is not None and (x_g is not None):
            out_g2 = self.convg2g(x_g)
        else:
            out_g2 = 0
        if isinstance(out_l2, torch.Tensor) and isinstance(out_g2, torch.Tensor) and out_l2.shape[-2:] != out_g2.shape[-2:]:
            target_size = (max(out_l2.shape[-2], out_g2.shape[-2]),
                           max(out_l2.shape[-1], out_g2.shape[-1]))
            out_l2 = F.interpolate(out_l2, size=target_size, mode='bilinear', align_corners=False)
            out_g2 = F.interpolate(out_g2, size=target_size, mode='bilinear', align_corners=False)
        global_out = (out_l2 if isinstance(out_l2, torch.Tensor) else 0) + (out_g2 if isinstance(out_g2, torch.Tensor) else 0)

        # Apply BatchNorm and spiking activation
        if isinstance(local_out, torch.Tensor):
            local_out = self.act_l(self.bn_l(local_out))
        else:
            batch, _, h, w = x_l.size()
            local_out = torch.zeros(batch, self.out_cl, h, w, device=x_l.device)
        if isinstance(global_out, torch.Tensor):
            global_out = self.act_g(self.bn_g(global_out))
        else:
            batch, _, h, w = x_l.size()
            global_out = torch.zeros(batch, self.out_cg, h, w, device=x_l.device)

        return local_out, global_out
