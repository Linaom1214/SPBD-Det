import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .encoder.models import ViG, ViG2
from .decoder.models import MultiQueryMaskDecoder

def autopad(kernel_size):
    return (kernel_size-1)//2

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, autopad(kernel_size), bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, autopad(dw_size), groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class Decoder(nn.Module):
    def __init__(self, hidden, oup, kernel_size=3):
        super().__init__()
        self.ghost = GhostModule(hidden, oup, kernel_size)
    
    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode="bilinear", align_corners=True)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.ghost(x1)
        return x1

class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, drop=0.1):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)

class SimpleDecoder(nn.Module):
    def __init__(self, ratio=1, n_class=2):
        super().__init__()
        self.decode2 = Decoder(int(ratio*256) + int(ratio*128), int(ratio*128))
        self.decode1 = Decoder(int(ratio*128) + int(ratio*64), int(ratio*64))
        self.decode0 = Decoder(int(ratio*64) + int(ratio*32), int(ratio*32))
        self.head = _FCNHead(int(ratio*32), n_class)
        self.apply(self.__init_weights)

    def __init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.2)
    
    def forward(self, x, img):
        e1, e2, e3, e4 = x
        d2 = self.decode2(e4, e3)
        d1 = self.decode1(d2, e2)
        d0 = self.decode0(d1, e1)
        out = self.head(d0)
        return out, None


class SPD(nn.Module):
    def __init__(
        self,
        encoder_out_features=(64, 128, 256),
        mask_depths=(1, 2, 4),
        mask_num_head=8,
        mask_mlp_dim=2048,
        step_size=0.1,
        t_total=1.0,
        alpha=0.5,
        beta=0.5,
        sobel=True,
        without_spd=False,
        without_rep=False,
        **kwargs,
    ):
        super().__init__()
        self.image_encoder = ViG() if without_rep else ViG2()
        self.mask_decoder = MultiQueryMaskDecoder(
            encoder_out_features=list(encoder_out_features),
            mask_depths=list(mask_depths),
            mask_num_head=mask_num_head,
            mask_mlp_dim=mask_mlp_dim,
            step_size=step_size,
            t_total=t_total,
            alpha=alpha,
            beta=beta,
            sobel=sobel,
        )
        if without_spd:
            self.mask_decoder = SimpleDecoder()

    def forward(self, inputs):
        image_features = self.image_encoder(inputs)
        low_res_masks, _ = self.mask_decoder(image_features, inputs)
        low_res_masks = F.interpolate(
            low_res_masks,
            size=(inputs.shape[2], inputs.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return low_res_masks

if __name__ == "__main__":
    model = SPD()
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print(y.shape)