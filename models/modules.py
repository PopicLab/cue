import torch.nn as nn
import torch.nn.modules.utils as torch_utils
from collections import namedtuple

ConvLayerConfig = namedtuple('LayerConfig', 'in_channels out_channels kernel_size padding pool')

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, pool=True, relu=True, bn=False):
        super(Conv, self).__init__()
        layers = [nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x if self.skip is None else self.skip(x)
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, depth, nc, expansion):
        super(Hourglass, self).__init__()
        self.depth = depth
        nc_expanded = nc + expansion
        self.up1 = Residual(nc, nc)
        self.pool = nn.MaxPool2d(2, 2)
        self.low1 = Residual(nc, nc_expanded)
        if self.depth > 1:
            self.low2 = Hourglass(self.depth - 1, nc_expanded, expansion)
        else:
            self.low2 = Residual(nc_expanded, nc_expanded)
        self.low3 = Residual(nc_expanded, nc)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        pool = self.pool(x)
        low1 = self.low1(pool)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up1 = self.up1(x)
        up2 = self.up2(low3)
        return up1 + up2


class HourglassBackbone(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HourglassBackbone, self).__init__()
        self.layers = nn.Sequential(Conv(in_channels, 64, kernel_size=7, stride=2, pool=False,
                                         padding=3, relu=True, bn=True),
                                    Residual(64, 128),
                                    nn.MaxPool2d(2, 2),
                                    Residual(128, 128),
                                    Residual(128, out_channels))

    def forward(self, x):
        return self.layers(x)

class RefineBackbone(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RefineBackbone, self).__init__()
        self.layers = nn.Sequential(Conv(in_channels=in_channels, out_channels=6, kernel_size=5, padding=0, pool=True),
                                    Conv(in_channels=6, out_channels=out_channels, kernel_size=5, padding=0, pool=True))

    def forward(self, x):
        return self.layers(x)


class RefineBackboneKP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RefineBackboneKP, self).__init__()
        self.layers = nn.Sequential(Conv(in_channels=in_channels, out_channels=6, kernel_size=5, padding=0, pool=True),
                                    Conv(in_channels=6, out_channels=16, kernel_size=5, padding=0, pool=True),
                                    Conv(in_channels=16, out_channels=out_channels, kernel_size=5, padding=0, pool=True))

    def forward(self, x):
        return self.layers(x)


def conv_module(layer_configs):
    layers = []
    for layer_config in layer_configs:
        layers.append(Conv(in_channels=layer_config.in_channels,
                           out_channels=layer_config.out_channels,
                           kernel_size=layer_config.kernel_size,
                           padding=layer_config.padding,
                           pool=layer_config.pool))
    return nn.Sequential(*layers)


def staged_conv_module(staged_layer_configs):
    stage2layers = []
    for layer_configs, count in staged_layer_configs:
        for _ in range(count):
            stage2layers.append(conv_module(layer_configs))
    return nn.ModuleList(stage2layers)


def fc_module(init_in_features, final_out_features, inner_layer_dims, relu=True):
    layers = []
    for i in range(len(inner_layer_dims)):
        in_features = init_in_features if i == 0 else inner_layer_dims[i - 1]
        out_features = final_out_features if i == (len(inner_layer_dims) - 1) else inner_layer_dims[i]
        layers.append(nn.Linear(in_features=in_features, out_features=out_features))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        # dropout
    return nn.Sequential(*layers)


# noinspection PyProtectedMember
def get_conv2d_layer_output_shape(in_dim, kernel_size, stride, padding, dilation=1):
    in_dim = torch_utils._pair(in_dim)
    kernel_size = torch_utils._pair(kernel_size)
    stride = torch_utils._pair(stride)
    padding = torch_utils._pair(padding)
    dilation = torch_utils._pair(dilation)
    out_dim_0 = (in_dim[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    out_dim_1 = (in_dim[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    return out_dim_0, out_dim_1


def get_conv_module_output_shape(input_dim, module):
    dim = input_dim
    for module_layer in module:
        if isinstance(module_layer, Conv):
            for layer in module_layer.layers:
                if isinstance(layer, nn.Conv2d):
                    dim = get_conv2d_layer_output_shape(dim, layer.kernel_size, layer.stride, layer.padding)
                elif isinstance(layer, nn.MaxPool2d):
                    dim = (int(dim[0] / 2),
                           int(dim[1] / 2))
    return dim

