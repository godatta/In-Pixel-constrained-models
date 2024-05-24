'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


cfg = {
    'VGG5': [6, 16, 'M', 32, 64, 'M'],
    'VGG9': [64, 'M', 256, 'M', 256, 'M', 512, 'M', 256, 'M'], #inspired by wrn Increased initial channels, \
                                                                    #and as we have less op classes, made last channels smaller 
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(256, num_classes) #make it 256 for all and 64 for vgg5

    def forward(self, x):
        out = self.features(x)
        #print(out.shape)
        #out = self.features[4:8](out)
        #print(out.shape)
        #out = self.features[8:15](out)
        #print(out.shape)
        #out = self.features[15:22](out)
        #print(out.shape)
        #out = self.features[22:29](out)
        #print(out.shape)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3 # make it 3 for vww
        layers += [nn.Conv2d(in_channels, 6, kernel_size=7, padding=1, stride=7), #1st layer use large kernel.
                                                                                            
                           nn.BatchNorm2d(6),
                           nn.ReLU(inplace=True)]
        in_channels = 6
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, stride=1), # we generally use kernel 3 and strinde 0
                                                                                            
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


net = VGG('VGG9', num_classes=2)
x = torch.randn(2,3,640,640)
y = net(x)
