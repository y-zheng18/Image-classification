from model.modules import *
from model.resnet import *
import torch.nn as nn
import torch

class AutoEnocder(nn.Module):
    def __init__(self, backbone, embedding_size=2048, num_classes=20, fix_backbone=True):
        super(AutoEnocder, self).__init__()
        self.backbone = backbone
        self.fix_backbone = fix_backbone
        if self.fix_backbone:
            self.backbone.requires_grad_(requires_grad=False)
        in_planes = self.backbone.embedding_size
        self.encoder = nn.Sequential(
            nn.Linear(in_planes, embedding_size * 2),
            nn.PReLU(),
            nn.Linear(embedding_size * 2, embedding_size * 2),
            nn.PReLU(),
            nn.Linear(embedding_size * 2, embedding_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 2),
            nn.PReLU(),
            nn.Linear(embedding_size * 2, embedding_size * 2),
            nn.PReLU(),
            nn.Linear(embedding_size * 2, in_planes)
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.PReLU(),
            nn.Linear(embedding_size, num_classes),
        )

    def forward(self, images):
        if self.fix_backbone:
            with torch.no_grad():
                _, feats = self.backbone(images, return_feats=True)
            embedding = self.encoder(feats)
            decoded_feats = self.decoder(embedding)
            cls_outputs = self.classifier(embedding)
        else:
            cls_outputs, feats = self.backbone(images, return_feats=True)
            embedding = self.encoder(feats)
            decoded_feats = self.decoder(embedding)
            # cls_outputs = self.classifier(embedding)

        return embedding, feats, decoded_feats, cls_outputs


if __name__ == '__main__':
    back_bone = WideResNet()
    net = AutoEnocder(back_bone, embedding_size=1024, num_classes=100)
    # torch.save(net.state_dict(), 'test_resnet.pth')
    x = net(torch.randn((32, 3, 32, 32)))
    print(x[0].shape)