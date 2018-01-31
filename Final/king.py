import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)
    
class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, densenet, n_classes=1, sizes=(2, 3, 6, 8), psp_size=1024):
        super(PSPNet, self).__init__()
        self.feature1 = nn.Sequential(*list(densenet.features.children())[0:4])
        self.feature2 = nn.Sequential(*list(densenet.features.children())[4:6])
        self.feature3 = nn.Sequential(*list(densenet.features.children())[6:8])
        self.feature4 = nn.Sequential(*list(densenet.features.children())[8:10])
        self.feature5 = nn.Sequential(*list(densenet.features.children())[10:12])
        
        self.psp = PSPModule(psp_size, 512, sizes)
        self.up_1 = PSPUpsample(512, 256)
        self.up_2 = PSPUpsample(256 + 256, 64)
        self.up_3 = PSPUpsample(128 + 64, 32)
        self.up_4 = PSPUpsample(32 + 64, 16)
        self.up_5 = PSPUpsample(16, 8)
        
        self.final = nn.Sequential(
            nn.Conv2d(8, n_classes, kernel_size=1),
            nn.Sigmoid()
        )
        
        #for p in self.features.parameters():
        #    p.requires_grad = False
            
    def forward(self, x):
        f1 = self.feature1(x)
        f2 = self.feature2(f1)
        f3 = self.feature3(f2)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        
        p = self.psp(f5)
        p = torch.cat([self.up_1(p), f3], 1)
        p = torch.cat([self.up_2(p), f2], 1)
        p = torch.cat([self.up_3(p), f1], 1)
        p = self.up_4(p)
        p = self.up_5(p)
        
        return self.final(p)