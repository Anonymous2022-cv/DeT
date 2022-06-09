import torch
from torch import nn as nn
from torch.nn import functional as F

class Fu_net(nn.Module):
    def __init__(self, channels):
        super(Fu_net, self).__init__()
        ##1
        self.conv11_k3_s1 = nn.Conv2d(channels, channels, 3, 1, 1)  # k3 s1
        self.conv12_k3_s1 = nn.Conv2d(channels, channels, 3, 1, 1)  # k3 s1
        self.conv13_k3_s1 = nn.Conv2d(channels, channels, 3, 1, 1)  # k3 s2

        ##2
        self.conv21_k3_s1 = nn.Conv2d(channels, channels, 3, 1, 1)  # k3 s1
        self.conv22_k3_s1 = nn.Conv2d(channels, channels, 3, 1, 1)  # k3 s2

        ##3
        self.conv31_k3_s1 = nn.Conv2d(channels, channels, 3, 1, 1)  # k3 s1
        self.conv32_k3_s1 = nn.Conv2d(channels, channels, 3, 1, 1)  # k3 s2

        ##4
        self.conv41_k3_s1 = nn.Conv2d(channels*2, channels, 3, 1, 1)  # k3 s1
        self.conv42_k3_s1 = nn.Conv2d(channels, channels, 3, 1, 1)  # k3 s1

        ##5
        self.conv51_k3_s1 = nn.Conv2d(channels*2, channels, 3, 1, 1)  # k3 s1
        self.conv52_k3_s1 = nn.Conv2d(channels, channels, 3, 1, 1)  # k3 s1

        ##6
        self.conv61_k3_s1 = nn.Conv2d(channels, channels, 3, 1, 1)  # k3 s1
        self.conv62_k3_s1 = nn.Conv2d(channels, channels, 3, 1, 1)  # k3 s1
        self.conv63_k3_s1 = nn.Conv2d(channels, 64, 3, 1, 1)  # k3 s1

        self.pool = nn.AvgPool2d(kernel_size=2)
        self.LRelu = nn.LeakyReLU(0.1, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, feature):
        ##1
        out = self.LRelu(self.conv12_k3_s1(self.conv11_k3_s1(feature)))
        out1 = self.pool(self.LRelu(self.conv13_k3_s1(out)))

        ##2
        out = self.LRelu(self.conv21_k3_s1(out1))
        out2 = self.pool(self.LRelu(self.conv22_k3_s1(out)))

        ##3
        out = self.LRelu(self.conv31_k3_s1(out2))
        out3 = self.pool(self.upsample(self.LRelu(self.conv32_k3_s1(out))))

        ##4
        out = self.LRelu(self.conv41_k3_s1(torch.cat((out3, out2), dim=1)))
        out4 = self.upsample(self.LRelu(self.conv42_k3_s1(out)))

        ##5
        out = self.LRelu(self.conv51_k3_s1(torch.cat((out4, out1), dim=1)))
        out5 = self.upsample(self.LRelu(self.conv52_k3_s1(out)))

        ##6
        out = self.LRelu(self.conv61_k3_s1(out5))
        out6 = self.conv63_k3_s1(self.LRelu(self.conv62_k3_s1(out)))

        return out6

if __name__ == '__main__':
    sim_data = torch.rand(1, 64*3, 512, 512)
    h, w = sim_data.size()[-2:]
    net = Fu_net(channels=64*3)
    pred = net(sim_data)
    print(pred.size())
