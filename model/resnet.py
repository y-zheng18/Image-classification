from model.modules import *
import torch.nn as nn
import torch

class ResNet(nn.Module):
    def __init__(self, layers=(2, 2, 2, 2), num_classes=20, dropout_rate=0, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = ResLayer(64, 64, layer_num=layers[0], stride=1, norm_layer=norm_layer, dropout_rate=dropout_rate)
        self.layer2 = ResLayer(64, 128, layer_num=layers[1], stride=2, norm_layer=norm_layer, dropout_rate=dropout_rate)  # 16 * 16
        self.layer3 = ResLayer(128, 256, layer_num=layers[2], stride=2, norm_layer=norm_layer, dropout_rate=dropout_rate)  # 8 * 8
        self.layer4 = ResLayer(256, 512, layer_num=layers[3], stride=2, norm_layer=norm_layer, dropout_rate=dropout_rate)  # 4 * 4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.embedding_size = 512
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.data_dropout = nn.Dropout(p=0.1)

    def forward(self, x, return_feats=False):
        x = self.data_dropout(x)
        x = self.relu(self.bn1(self.conv1(x)))
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        embedding = torch.flatten(x, 1)
        outputs = self.fc(embedding)
        if return_feats:
            return outputs, embedding
        return outputs


class WideResNet(nn.Module):
    def __init__(self, layers=(4, 4, 4), factor=10, num_classes=20, dropout_rate=0.3, norm_layer=None):
        super(WideResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(16)
        self.relu1 = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = WideResLayer(16, 16 * factor, layer_num=layers[0], stride=1, norm_layer=norm_layer, dropout_rate=dropout_rate)
        self.layer2 = WideResLayer(16 * factor, 32 * factor, layer_num=layers[1], stride=2, norm_layer=norm_layer,
                               dropout_rate=dropout_rate)  # 16 * 16
        self.layer3 = WideResLayer(32 * factor, 64 * factor, layer_num=layers[2], stride=2, norm_layer=norm_layer,
                               dropout_rate=dropout_rate)  # 8 * 8
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * factor, num_classes)
        self.bn2 = norm_layer(64 * factor)
        self.relu2 = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.embedding_size = 64 * factor

    def forward(self, x, return_feats=False):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu2(self.bn2(x))
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        embedding = torch.flatten(x, 1)
        outputs = self.fc(embedding)
        if return_feats:
            return outputs, embedding
        return outputs


class WideResNext(nn.Module):
    def __init__(self, layers=(4, 4, 4), factor=10, groups=32, num_classes=20, dropout_rate=0.3, norm_layer=None):
        super(WideResNext, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(16)
        self.relu1 = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = WideResnextLayer(16, 16 * factor, layer_num=layers[0], groups=groups, base_width=2,
                                       stride=1, norm_layer=norm_layer, dropout_rate=dropout_rate)
        self.layer2 = WideResnextLayer(16 * factor, 32 * factor, layer_num=layers[1], groups=groups, base_width=2,
                                       stride=2, norm_layer=norm_layer, dropout_rate=dropout_rate)
        self.layer3 = WideResnextLayer(32 * factor, 64 * factor, layer_num=layers[2], groups=groups, base_width=2,
                                       stride=2, norm_layer=norm_layer, dropout_rate=dropout_rate)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * factor, num_classes)
        self.bn2 = norm_layer(64 * factor)
        self.relu2 = nn.ReLU()
        # self.encoder = nn.Sequential(
        #     nn.Linear(640, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.PReLU(),
        #     nn.Linear(1024, 1024),
        # )
        self.fc = nn.Sequential(
            # nn.BatchNorm1d(1024),
            nn.Linear(64 * factor, num_classes)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_feats=False):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu2(self.bn2(x))
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        embedding = torch.flatten(x, 1)
        # embedding = self.encoder(embedding)
        outputs = self.fc(embedding)
        if return_feats:
            return outputs, embedding
        return outputs

class MultiResNet(nn.Module):
    def __init__(self, layers=(2, 2, 2, 2), num_classes=20, dropout_rate=0, norm_layer=None):
        super(MultiResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = ResLayer(64, 64, layer_num=layers[0], stride=1, norm_layer=norm_layer, dropout_rate=dropout_rate)
        self.layer2 = ResLayer(64, 128, layer_num=layers[1], stride=2, norm_layer=norm_layer, dropout_rate=dropout_rate)  # 16 * 16
        self.layer3 = ResLayer(128, 256, layer_num=layers[2], stride=2, norm_layer=norm_layer, dropout_rate=dropout_rate)  # 8 * 8
        self.layer4 = ResLayer(256, 512, layer_num=layers[3], stride=2, norm_layer=norm_layer, dropout_rate=dropout_rate)  # 4 * 4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            # nn.BatchNorm1d(1024),
            nn.Linear(960, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_feats=False):
        x = self.relu(self.bn1(self.conv1(x)))
        # x = self.maxpool(x)

        x = self.layer1(x)
        feats1 = self.avgpool(x)
        x = self.layer2(x)
        feats2 = self.avgpool(x)
        x = self.layer3(x)
        feats3 = self.avgpool(x)
        x = self.layer4(x)
        feats4 = self.avgpool(x)
        # x = torch.flatten(x, 1)
        embedding = torch.cat((feats1, feats2, feats3, feats4), dim=1)
        embedding = torch.flatten(embedding, 1)
        outputs = self.fc(embedding)
        return outputs

class MultiWideResNet(nn.Module):
    def __init__(self, layers=(4, 4, 4), factor=10, num_classes=20, dropout_rate=0.3, norm_layer=None):
        super(MultiWideResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(16)
        self.relu1 = nn.ReLU()
        self.layer1 = WideResLayer(16, 16 * factor, layer_num=layers[0], stride=1, norm_layer=norm_layer, dropout_rate=dropout_rate)
        self.layer2 = WideResLayer(16 * factor, 32 * factor, layer_num=layers[1], stride=2, norm_layer=norm_layer,
                               dropout_rate=dropout_rate)  # 16 * 16
        self.layer3 = WideResLayer(32 * factor, 64 * factor, layer_num=layers[2], stride=2, norm_layer=norm_layer,
                               dropout_rate=dropout_rate)  # 8 * 8
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * factor, num_classes)
        self.bn2 = norm_layer(64 * factor)
        self.relu2 = nn.ReLU()
        self.fc = nn.Sequential(
            # nn.BatchNorm1d(1024),
            nn.Linear((64 + 32 + 16) * factor, num_classes)
        )
        self.feats_encoder1 = nn.Sequential(
            # nn.BatchNorm1d(1024),
            nn.Linear(16 * factor, 16 * factor),
            # norm_layer(16 * factor),
            nn.BatchNorm1d(16 * factor),
            nn.ReLU()
        )
        self.feats_encoder2 = nn.Sequential(
            # nn.BatchNorm1d(1024),
            nn.Linear(32 * factor, 32 * factor),
            # norm_layer(16 * factor),
            nn.BatchNorm1d(32 * factor),
            nn.ReLU()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_feats=False):
        x = self.relu1(self.bn1(self.conv1(x)))
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out3 = self.relu2(self.bn2(out3))
        out3 = self.avgpool(out3)
        # x = torch.flatten(x, 1)
        embedding = torch.flatten(out3, 1)
        feats1 = self.feats_encoder1(torch.flatten(self.avgpool(out1), 1))
        feats2 = self.feats_encoder2(torch.flatten(self.avgpool(out2), 1))
        multi_feats = torch.cat((feats1, feats2, embedding), dim=1)
        outputs = self.fc(multi_feats)
        if return_feats:
            return outputs, multi_feats
        return outputs