import torch
import torch.nn as nn
from torchvision.models import resnet18

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(pretrained=True)
        self.layer1 = nn.Sequential(backbone.conv1,backbone.bn1,backbone.relu,backbone.maxpool,backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        for p in self.parameters():
            p.requires_grad = False
        return
    def forward(self,x):
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        return f2 , f3