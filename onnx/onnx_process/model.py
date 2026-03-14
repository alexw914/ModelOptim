import torch
from torch import nn
import torch.nn.functional as F


def block_layers(block, in_channel, t, out_channel, kernel_size, stride, n):
    strides = [stride] + [1] * (n - 1)
    layers = []
    for stride in strides:
        layers.append(block(in_channel, t * in_channel, out_channel, kernel_size, stride))
        in_channel = out_channel
    return nn.Sequential(*layers)


class MobileNetV2Block(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, ksize, stride, shortcut=False, padding=1):
        super(MobileNetV2Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, 1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU6(True),
            nn.Conv2d(mid_channel, mid_channel, ksize, stride, padding, bias=False, groups=mid_channel),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU6(True),
            nn.Conv2d(mid_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        if in_channel == out_channel:
            shortcut = True

        self.shortcut = shortcut

    def forward(self, x):
        x1 = self.block(x)
        if self.shortcut:
            x1 += x
        x1 = F.relu(x1)
        return x1


class MobileNetV2Output(nn.Module):
    def __init__(self, num_class=2, img_size=(448, 448)):
        super(MobileNetV2Output, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.t = [1, 6, 6, 6, 6, 6, 6]
        self.c = [32, 16, 24, 32, 64, 96, 160, 320]
        self.n = [1, 2, 3, 4, 3, 3, 1]
        self.s = [1, 2, 2, 2, 1, 2, 1]
        layer = []
        for i in range(len(self.t)):
            layer.append(block_layers(MobileNetV2Block, self.c[i], self.t[i], self.c[i + 1], 3, self.s[i], self.n[i]))
        self.layers = nn.Sequential(*layer)

        self.conv2 = nn.Sequential(
                nn.Conv2d(self.c[-1], 1280, 1, bias=False),
                nn.BatchNorm2d(1280),
                nn.ReLU(False),
                nn.AvgPool2d(kernel_size=(int(img_size[0] / 32), int(img_size[1] / 32)), stride=1),
                nn.Conv2d(1280, num_class, 1, bias=False),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.conv2(x)
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        return x


if __name__ == '__main__':
    model = MobileNetV2Output()
    x = torch.randn(1, 3, 448, 448)
    y = model(x)
    print(y.shape)
