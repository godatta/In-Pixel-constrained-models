'''MobileNetV2 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import math
#from ptflops import get_model_complexity_info

class customConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.register_buffer('identity_kernel', torch.ones(out_channels, in_channels, *kernel_size))
        self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size), requires_grad=True)
        with torch.no_grad():
            self.weights.data.normal_(0.0, 0.8)

    def forward(self, img):
        b, c, h, w = img.size()
        p00 = 0.1694
        p01 = 0.422378
        p10 = 0.25498
        p11 = 0.072789
        p20 = -0.17645
        p21 = -0.043589
        p30 = -0.09217
        img_unf = nn.functional.unfold(img, kernel_size=self.kernel_size,
                                       stride=self.stride, padding=self.padding).transpose(1, 2)
        identity_weights = self.identity_kernel.view(self.identity_kernel.size(0), -1)
        weights = self.weights.view(self.weights.size(0), -1)

        # f0 = (p00 + torch.zeros_like(img_unf)).matmul(identity_weights.t())
        # f1 = (p10 * (img_unf - 0.5)).matmul(identity_weights.t())
        # f2 = (p01 * torch.ones_like(img_unf)).matmul(weights.t())
        # f3 = (p20 * torch.pow(img_unf - 0.5, 2)).matmul(identity_weights.t())
        # f4 = (p11 * (img_unf - 0.5)).matmul(weights.t())
        # f5 = (p30 * torch.pow(img_unf - 0.5, 3)).matmul(identity_weights.t())
        # f6 = (p21 * torch.pow(img_unf - 0.5, 2)).matmul(weights.t())
        # f = (f0 + f1 + f2 + f3 + f4 + f5 + f6).transpose(1, 2)

        f = ((p00 + torch.zeros_like(img_unf) +
             p10 * (img_unf - 0.5) +
             p20 * torch.pow(img_unf - 0.5, 2) +
             p30 * torch.pow(img_unf - 0.5, 3)).matmul(identity_weights.t()) + \
            (p01 * torch.ones_like(img_unf) +
             p11 * (img_unf - 0.5) +
             p21 * torch.pow(img_unf - 0.5, 2)
             ).matmul(weights.t())).transpose(1, 2)

        out = f.view(b, self.out_channels, int(h/self.stride), int(w/self.stride))
        return out



class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 2),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (2, 320, 1, 1)] # Note: Changed expansion from 6 to 2, as the classifier has only 2 classes for VWW

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        #self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=5, padding=0, bias=False)#chaning the 3x3 kernel
        #to 5x5 with stride 5 and padding 1 and out channel to 8. earlier ker:3, stride:1, pad:1 , Cout:32
        #initial_weights = torch.randn(8,2,7,5)*0.8
        #self.conv1 = nn.Parameter(initial_weights, requires_grad=True)
        self.conv1 = customConv2(in_channels=3, out_channels=8, kernel_size=(5, 5), stride = 5, padding = 0)
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm2d(8) #changed 32 to 8
        self.layers = self._make_layers(in_planes=8) # changed 32 to 8
        self.conv2 = nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(320)
        self.linear = nn.Linear(320, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def first_layer_processing(self, input, stride, padding):
        p00 = 0.1694
        p01 = 0.422378
        p10 = 0.25498
        p11 = 0.072789
        p20 = -0.17645
        p21 = -0.043589
        p30 = -0.09217

        xkernel = self.conv1.shape[2]
        ykernel = self.conv1.shape[3]
        out_kernel = self.conv1.shape[0]
        in_kernel = self.conv1.shape[1]
        ximage = input.shape[2]
        yimage = input.shape[3]

        xoutput = int(((ximage - xkernel + 2 * padding) / stride) + 1)
        youtput = int(((yimage - ykernel + 2 * padding) / stride) + 1)
        output = torch.zeros((input.shape[0], out_kernel, xoutput, youtput))

        if padding != 0:
            inputPadded = torch.zeros((input.shape[0], input.shape[1], ximage + padding*2, yimage + padding*2))
            inputPadded[:, :, int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = input
        else:
            inputPadded = input

        for ch in range(out_kernel):
            for y in range(yimage):
        # Exit Convolution
                if y > yimage - ykernel:
                    break
        # Only Convolve if y has gone down by the specified Strides
                if y % stride == 0:
                    for x in range(ximage):
                # Go to next row once kernel is out of bounds
                        if x > ximage - xkernel:
                            break
                    try:
                    # Only Convolve if x has moved by the specified Strides
                        if x % stride == 0:
                            weight = self.conv1[ch, :, :, :]
                            input = inputPadded[:, :, x: x + xkernel, y: y + ykernel]
                            output[:, ch, x, y] = (p00+p10*(input-0.5)+p01*weight+p20*(input-0.5)*(input-0.5)+p11*(input-0.5)*weight+p30*(input-0.5)*(input-0.5)*(input-0.5)+p21*(input-0.5)*(input-0.5)*weight).sum(axis=(1,2,3))
                    except:
                        break

        return output

    def quantise(self, x, k, do_quantise=True):
        Max=torch.max(x)
        Min=torch.min(x)
        if Max<-Min:
            Max=-Min
        if( not do_quantise):
            return x
        Digital=torch.round(((2**k)-1)*x/Max)
        output=Max*Digital/((2**k)-1)
        return output

    def forward(self, x):
        #out = self.first_layer_processing(x, 5, 0)
        conv1weights = self.conv1.weights.data
        self.conv1.weights.data = self.quantise(self.conv1.weights.data, 6)
        out = self.conv1(x)
        self.conv1.weights.data = conv1weights
        out = out.to(device='cuda')
        out = self.quantise(out, 8)
        #print(out.shape)
        #print(out)
        out = F.relu(self.bn1(out))
        #out = self.dropout(out)
        #out = F.relu(self.bn1(self.conv1(x)))
        #print(out.shape)
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

#net = MobileNetV2(num_classes=2)    
#x = torch.randn(2, 3, 560, 560) #changed the 224 to 560.
#y = net(x)
'''
flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
print('Flops:  ' + flops)
print('Params: ' + params)
'''