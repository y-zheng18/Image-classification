import torch.nn as nn
import torch
from torchvision.models import resnet


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None, norm_layer=None, dropout_rate=0):
        super(ResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = norm_layer(out_planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm_layer(out_planes)
        self.downsample = downsample
        self.dropout = None
        if dropout_rate:
            self.dropout = nn.Dropout(p=dropout_rate, inplace=False)

    def forward(self, x):
        res = x
        # print(x.shape)
        out = self.relu(self.bn1(self.conv1(x)))
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            res = self.downsample(x)

        out += res
        out = self.relu(out)

        return out


class ResLayer(nn.Module):
    def __init__(self, in_planes, out_planes, layer_num=2, stride=1, norm_layer=None, dropout_rate=0):
        super(ResLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        downsample_layer = None
        if stride != 1 or in_planes != out_planes:
            downsample_layer = nn.Sequential(
                conv1x1(in_planes, out_planes, stride),
                norm_layer(out_planes),
            )
        block1 = ResBlock(in_planes, out_planes, stride, downsample_layer, norm_layer, dropout_rate)
        self.layer = []
        self.layer.append(block1)
        for i in range(layer_num - 1):
            self.layer.append(ResBlock(in_planes=out_planes, out_planes=out_planes, stride=1, norm_layer=norm_layer, dropout_rate=dropout_rate))
        self.layer = nn.Sequential(*self.layer)

    def forward(self, x):
        return self.layer(x)


class WideResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None, norm_layer=None, dropout_rate=0):
        super(WideResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_planes, out_planes)
        self.bn1 = norm_layer(out_planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_planes, out_planes, stride)
        self.bn2 = norm_layer(out_planes)
        self.downsample = downsample
        self.dropout = None
        if dropout_rate:
            self.dropout = nn.Dropout(p=dropout_rate, inplace=False)

    def forward(self, x):
        res = x
        # print(x.shape)
        out = self.relu(self.bn1(self.conv1(x)))
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            res = self.downsample(x)

        out += res
        out = self.relu(out)

        return out


class WideResLayer(nn.Module):
    def __init__(self, in_planes, out_planes, layer_num=2, stride=1, norm_layer=None, dropout_rate=0):
        super(WideResLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        downsample_layer = None
        if stride != 1 or in_planes != out_planes:
            downsample_layer = nn.Sequential(
                conv1x1(in_planes, out_planes, stride),
                norm_layer(out_planes),
            )
        block1 = WideResBlock(in_planes, out_planes, stride, downsample_layer, norm_layer, dropout_rate)
        self.layer = []
        self.layer.append(block1)
        for i in range(layer_num - 1):
            self.layer.append(WideResBlock(in_planes=out_planes, out_planes=out_planes, stride=1, norm_layer=norm_layer, dropout_rate=dropout_rate))
        self.layer = nn.Sequential(*self.layer)

    def forward(self, x):
        return self.layer(x)
