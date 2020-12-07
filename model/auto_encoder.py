from model.modules import *
from model.resnet import *
import torch.nn as nn
import torch

class AutoEnocder(nn.Module):
    def __init__(self, in_planes, embedding_size=2048, num_classes=20):
        super(AutoEnocder, self).__init__()
        in_planes = self.back_bone.embedding_size
        self.encoder = nn.Sequential(
            nn.Linear(in_planes, embedding_size // 2),
            nn.PReLU(),
            nn.Linear(embedding_size // 2, embedding_size // 2),
            nn.PReLU(),
            nn.Linear(embedding_size // 2, embedding_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.PReLU(),
            nn.Linear(embedding_size // 2, embedding_size // 2),
            nn.PReLU(),
            nn.Linear(embedding_size // 2, in_planes)
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(embedding_size),
            nn.PReLU(),
            nn.Linear(embedding_size, num_classes),
        )

    def forward(self, back_bone, images):
        with torch.no_grad():
            _, feats = back_bone(images, return_feats=True)
        embedding = self.encoder(feats)
        decoded_feats = self.decoder(embedding)
        cls_outputs = self.classifier(embedding)
        return embedding, feats, decoded_feats, cls_outputs


if __name__ == '__main__':
    back_bone = WideResNet()
    net = AutoEnocder(in_planes=back_bone.embedding_size, embedding_size=1024, num_classes=100)
    # torch.save(net.state_dict(), 'test_resnet.pth')
    x = net(back_bone, torch.randn((32, 3, 32, 32)))
    print(x[0].shape)