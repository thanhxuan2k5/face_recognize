
import torch
from torch import nn
import torch.nn.functional as F

class DepthwiseConv(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride, padding, groups):
        super(DepthwiseConv, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_chs)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out

class MiniFASNet(nn.Module):
    def __init__(self, keep, embedding_size, conv6_kernel=(7, 7)):
        super(MiniFASNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        self.conv2 = nn.Sequential(
            DepthwiseConv(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.PReLU(64)
        )
        self.conv3 = nn.Sequential(
            DepthwiseConv(64, 128, kernel_size=3, stride=2, padding=1, groups=64),
            nn.PReLU(128),
            DepthwiseConv(128, 128, kernel_size=3, stride=1, padding=1, groups=128),
            nn.PReLU(128)
        )
        self.conv4 = nn.Sequential(
            DepthwiseConv(128, 256, kernel_size=3, stride=2, padding=1, groups=128),
            nn.PReLU(256),
            DepthwiseConv(256, 256, kernel_size=3, stride=1, padding=1, groups=256),
            nn.PReLU(256)
        )
        self.conv5 = nn.Sequential(
            DepthwiseConv(256, 512, kernel_size=3, stride=2, padding=1, groups=256),
            nn.PReLU(512),
            DepthwiseConv(512, 512, kernel_size=3, stride=1, padding=1, groups=512),
            nn.PReLU(512)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=conv6_kernel, stride=1, padding=0, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.PReLU(512)
        )
        self.fc = nn.Linear(512, embedding_size)
        self.prob = nn.Linear(embedding_size, 3) # 3 classes: Real, Spoof1, Spoof2

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.prob(out)
        return out

def MiniFASNetV2():
    return MiniFASNet(keep=None, embedding_size=128, conv6_kernel=(5, 5))
