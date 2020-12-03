from model.modules import ResLayer
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class WideResNet(nn.Module):
    def __init__(self, layers=(4, 4, 4), num_classes=20, dropout_rate=0.3, norm_layer=None):
        super(WideResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(16)
        self.relu1 = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = ResLayer(16, 160, layer_num=layers[0], stride=1, norm_layer=norm_layer, dropout_rate=dropout_rate)
        self.layer2 = ResLayer(160, 320, layer_num=layers[1], stride=2, norm_layer=norm_layer,
                               dropout_rate=dropout_rate)  # 16 * 16
        self.layer3 = ResLayer(320, 640, layer_num=layers[2], stride=2, norm_layer=norm_layer,
                               dropout_rate=dropout_rate)  # 8 * 8
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(640, num_classes)
        self.bn2 = norm_layer(640)
        self.relu2 = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu2(self.bn2(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    net = WideResNet()
    print(net)
    from torchvision.models import resnet, resnext101_32x8d, ResNet
    import torchvision.models as models
    resnet18 = resnet.resnet18(num_classes=20)
    # print(resnet18, models.resnext50_32x4d(num_classes=20))

    torch.save(net.state_dict(), 'test_resnet.pth')
    # torch.save(resnet18.state_dict(), 'official_resnet.pth')
    for i in range(1000):
        x = net(torch.randn((32, 3, 32, 32)))
        v, l = torch.max(x, dim=1)
        print(l)