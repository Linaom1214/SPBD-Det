import torch
from torch import nn
from timm.models.layers import DropPath
import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation).
    Ref: https://github.com/ultralytics/ultralytics/blob/70d4a3752eda44580209ac821ff2c70df17a41bb/ultralytics/nn/modules/conv.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer.
    Ref: https://github.com/ultralytics/ultralytics/blob/70d4a3752eda44580209ac821ff2c70df17a41bb/ultralytics/nn/modules/conv.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension.
    Ref: https://github.com/ultralytics/ultralytics/blob/70d4a3752eda44580209ac821ff2c70df17a41bb/ultralytics/nn/modules/conv.py
    """

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class Bottleneck(nn.Module):
    """Standard bottleneck.
    Ref: https://github.com/ultralytics/ultralytics/blob/70d4a3752eda44580209ac821ff2c70df17a41bb/ultralytics/nn/modules/block.py
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher.
    Ref: https://github.com/ultralytics/ultralytics/blob/70d4a3752eda44580209ac821ff2c70df17a41bb/ultralytics/nn/modules/block.py
    """

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions.
    Ref: https://github.com/ultralytics/ultralytics/blob/70d4a3752eda44580209ac821ff2c70df17a41bb/ultralytics/nn/modules/block.py
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

def autopad2(kernel_size):
    return (kernel_size-1)//2

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, autopad2(kernel_size), bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, autopad2(dw_size), groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class Encoder(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=2):
        super().__init__()
        self.ghost1 = GhostModule(inp, int(inp*2), kernel_size)
        self.convdw = nn.Conv2d(in_channels = int(inp*2), 
                                out_channels = int(inp*2), 
                                kernel_size = kernel_size,
                                stride = stride,
                                padding= autopad(kernel_size), 
                                groups=int(inp*2))
        self.bn = nn.BatchNorm2d(int(inp*2))
        self.ghost2 = GhostModule(int(inp*2), oup, kernel_size, stride=1)
        self.shortcut = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size, stride,
                          autopad(kernel_size), groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            )
    def forward(self,x):
        residual = x
        x = self.ghost1(x)
        x = self.convdw(x)
        x = self.bn(x)
        x = self.ghost2(x)
        x = x + self.shortcut(residual)
        return x 

class Encoder2(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=2):
        super().__init__()
        self.ghost1 = GhostModule(inp, int(inp*2), kernel_size)
        self.convdw = nn.Conv2d(in_channels=int(inp*2),
                                out_channels=int(inp*2),
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=autopad(kernel_size),
                                groups=int(inp*2), bias=False)
        self.bn = nn.BatchNorm2d(int(inp*2))
        self.ghost2 = GhostModule(int(inp*2), oup, kernel_size, stride=1)

        # 主分支的 shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size, stride,
                      autopad(kernel_size), groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.Conv2d(inp, oup, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
        )

        # 重参数化分支
        self.reparam_branch = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, stride, autopad(kernel_size), bias=False),
            nn.BatchNorm2d(oup)
        )
        self.dropout = nn.Dropout2d(0.3)  # 添加 Dropout
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if hasattr(self, 'fused'):
            return self.dropout(self.forward_fused(x))
        else:
            residual = x
            x = self.ghost1(x)
            x = self.convdw(x)
            x = self.bn(x)
            x = self.ghost2(x)
            x = x + self.shortcut(residual) + self.alpha * self.reparam_branch(residual)
            return self.dropout(x)

    def forward_fused(self, x):
        residual = x
        x = self.ghost1(x)
        x = self.convdw(x)
        x = self.ghost2(x)
        x = x + self.shortcut(residual)
        return x

    def fuse_conv_bn(self, conv, bn):
        """融合卷积层和批归一化层"""
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        """将每个 conv-bn 组合转换成一个卷积层"""
        # ghost1
        ghost1_conv1_weight, ghost1_conv1_bias = self.fuse_conv_bn(self.ghost1.primary_conv[0], self.ghost1.primary_conv[1])
        ghost1_conv2_weight, ghost1_conv2_bias = self.fuse_conv_bn(self.ghost1.cheap_operation[0], self.ghost1.cheap_operation[1])

        # convdw
        convdw_weight, convdw_bias = self.fuse_conv_bn(self.convdw, self.bn)

        # ghost2
        ghost2_conv1_weight, ghost2_conv1_bias = self.fuse_conv_bn(self.ghost2.primary_conv[0], self.ghost2.primary_conv[1])
        ghost2_conv2_weight, ghost2_conv2_bias = self.fuse_conv_bn(self.ghost2.cheap_operation[0], self.ghost2.cheap_operation[1])

        # shortcut
        shortcut_conv1_weight, shortcut_conv1_bias = self.fuse_conv_bn(self.shortcut[0], self.shortcut[1])
        shortcut_conv2_weight, shortcut_conv2_bias = self.fuse_conv_bn(self.shortcut[2], self.shortcut[3])

        # reparam_branch
        reparam_branch_conv_weight, reparam_branch_conv_bias = self.fuse_conv_bn(self.reparam_branch[0], self.reparam_branch[1])

        return [
            (ghost1_conv1_weight, ghost1_conv1_bias),
            (ghost1_conv2_weight, ghost1_conv2_bias),
            (convdw_weight, convdw_bias),
            (ghost2_conv1_weight, ghost2_conv1_bias),
            (ghost2_conv2_weight, ghost2_conv2_bias),
            (shortcut_conv1_weight, shortcut_conv1_bias),
            (shortcut_conv2_weight, shortcut_conv2_bias),
            (reparam_branch_conv_weight, reparam_branch_conv_bias)  # 新增
        ]

    def merge_consecutive_convs(self, conv1_weight, conv1_bias, conv2_weight, conv2_bias):
        """
        融合连续两层卷积。这里假设：
          - conv1 权重 shape 为 [in_ch, 1, k, k]（深度卷积）
          - conv2 权重 shape 为 [oup, in_ch, 1, 1]（逐点卷积）
        最终融合得到的权重 shape 为 [oup, in_ch, k, k]。
        融合公式：
          fused_weight[i,j,:,:] = conv2_weight[i,j,0,0] * conv1_weight[j,0,:,:]
          fused_bias[i] = conv2_bias[i] + sum_j(conv2_weight[i,j,0,0] * conv1_bias[j])
        """
        k = conv1_weight.size(2)  # 卷积核尺寸
        # 将 conv1_weight 去掉通道单一因子，shape 从 [in_ch,1,k,k] 调整为 [in_ch,k,k]
        conv1_weight_ = conv1_weight.squeeze(1)  # shape: [in_ch, k, k]
        # 提取 conv2_weight 的数值因子，shape 为 [oup, in_ch]
        conv2_factor = conv2_weight[:, :, 0, 0]    # shape: [oup, in_ch]
        # 将 conv2_factor 扩展以便做元素乘法，shape: [oup, in_ch, 1, 1]
        conv2_factor = conv2_factor.unsqueeze(-1).unsqueeze(-1)
        # 计算融合后的卷积核权重： [oup, in_ch, k, k]
        fused_weight = conv2_factor * conv1_weight_.unsqueeze(0)
        
        # 计算融合后的 bias：
        if conv1_bias is None:
            fused_bias = conv2_bias
        else:
            # 对于每个输出通道 i，
            # fused_bias[i] = conv2_bias[i] + sum_j(conv2_weight[i,j,0,0] * conv1_bias[j])
            fused_bias = conv2_bias + (conv2_weight[:, :, 0, 0] @ conv1_bias)
        return fused_weight, fused_bias

    def switch_to_deploy(self):
        """将模型转换为推理模式 (融合 conv-bn，并融合 shortcut 与 reparam_branch 分支)"""
        if hasattr(self, 'fused'):
            print("Already in deploy mode.")
            return

        # 获取融合后的各个 conv-bn 参数
        (ghost1_conv1_weight, ghost1_conv1_bias), \
        (ghost1_conv2_weight, ghost1_conv2_bias), \
        (convdw_weight, convdw_bias), \
        (ghost2_conv1_weight, ghost2_conv1_bias), \
        (ghost2_conv2_weight, ghost2_conv2_bias), \
        (shortcut_conv1_weight, shortcut_conv1_bias), \
        (shortcut_conv2_weight, shortcut_conv2_bias), \
        (reparam_branch_conv_weight, reparam_branch_conv_bias) = self.get_equivalent_kernel_bias()

        # --- GhostModule1 ---
        # 为了保留激活，采用 nn.Sequential 包裹 conv 和激活函数
        self.ghost1.primary_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.ghost1.primary_conv[0].in_channels,
                out_channels=self.ghost1.primary_conv[0].out_channels,
                kernel_size=self.ghost1.primary_conv[0].kernel_size,
                stride=self.ghost1.primary_conv[0].stride,
                padding=self.ghost1.primary_conv[0].padding,
                bias=True
            ),
            nn.ReLU(inplace=True)
        )
        self.ghost1.primary_conv[0].weight.data = ghost1_conv1_weight
        self.ghost1.primary_conv[0].bias.data = ghost1_conv1_bias

        self.ghost1.cheap_operation = nn.Sequential(
            nn.Conv2d(
                in_channels=self.ghost1.cheap_operation[0].in_channels,
                out_channels=self.ghost1.cheap_operation[0].out_channels,
                kernel_size=self.ghost1.cheap_operation[0].kernel_size,
                stride=self.ghost1.cheap_operation[0].stride,
                padding=self.ghost1.cheap_operation[0].padding,
                groups=self.ghost1.cheap_operation[0].groups,
                bias=True
            ),
            nn.ReLU(inplace=True)
        )
        self.ghost1.cheap_operation[0].weight.data = ghost1_conv2_weight
        self.ghost1.cheap_operation[0].bias.data = ghost1_conv2_bias

        # --- ConvDW ---
        self.convdw = nn.Conv2d(
            in_channels=self.convdw.in_channels,
            out_channels=self.convdw.out_channels,
            kernel_size=self.convdw.kernel_size,
            stride=self.convdw.stride,
            padding=self.convdw.padding,
            groups=self.convdw.groups,
            bias=True
        )
        self.convdw.weight.data = convdw_weight
        self.convdw.bias.data = convdw_bias
        delattr(self, 'bn')

        # --- GhostModule2 ---
        self.ghost2.primary_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.ghost2.primary_conv[0].in_channels,
                out_channels=self.ghost2.primary_conv[0].out_channels,
                kernel_size=self.ghost2.primary_conv[0].kernel_size,
                stride=self.ghost2.primary_conv[0].stride,
                padding=self.ghost2.primary_conv[0].padding,
                bias=True
            ),
            nn.ReLU(inplace=True)
        )
        self.ghost2.primary_conv[0].weight.data = ghost2_conv1_weight
        self.ghost2.primary_conv[0].bias.data = ghost2_conv1_bias

        self.ghost2.cheap_operation = nn.Sequential(
            nn.Conv2d(
                in_channels=self.ghost2.cheap_operation[0].in_channels,
                out_channels=self.ghost2.cheap_operation[0].out_channels,
                kernel_size=self.ghost2.cheap_operation[0].kernel_size,
                stride=self.ghost2.cheap_operation[0].stride,
                padding=self.ghost2.cheap_operation[0].padding,
                groups=self.ghost2.cheap_operation[0].groups,
                bias=True
            ),
            nn.ReLU(inplace=True)
        )
        self.ghost2.cheap_operation[0].weight.data = ghost2_conv2_weight
        self.ghost2.cheap_operation[0].bias.data = ghost2_conv2_bias

        # --- Shortcut & Reparam Branch 融合 ---
        # 1. 首先融合 shortcut 分支（原来为两层 conv 的 sequential）
        merged_shortcut_k, merged_shortcut_b = self.merge_consecutive_convs(
            shortcut_conv1_weight, shortcut_conv1_bias,
            shortcut_conv2_weight, shortcut_conv2_bias
        )
        # 2. 利用 alpha 融合 reparam_branch 分支
        # 注意：此处 alpha 使用 detach 保证不参与梯度计算
        merged_branch_k = merged_shortcut_k + self.alpha.detach() * reparam_branch_conv_weight
        merged_branch_b = merged_shortcut_b + self.alpha.detach() * reparam_branch_conv_bias

        # 用融合后的参数创建新的分支，命名为 shortcut（原来的 reparam_branch 将不再存在）
        in_ch = self.shortcut[0].in_channels  # 原始 shortcut 第一层的 in_channels
        out_ch = merged_branch_k.size(0)       # 融合后的输出通道数
        new_kernel_size = merged_branch_k.shape[2:]
        new_stride = self.shortcut[0].stride
        new_padding = autopad(new_kernel_size[0])
        self.shortcut = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=new_kernel_size,
            stride=new_stride,
            padding=new_padding,
            bias=True
        )
        self.shortcut.weight.data = merged_branch_k
        self.shortcut.bias.data = merged_branch_b

        # 删除原来的 reparam_branch 分支
        if hasattr(self, 'reparam_branch'):
            del self.reparam_branch

        self.fused = True
        print("Switch to deploy mode (fused conv-bn with activations & merged shortcut-reparam branch).")

if __name__ == '__main__':
    # 创建模型
    model = Encoder2(3, 16)
    model.eval()

    # 测试精度是否对齐
    x = torch.randn(1, 3, 256, 256)
    y1 = model(x)
    model.switch_to_deploy()
    y2 = model(x)
    print(torch.allclose(y1, y2, atol=1e-6))