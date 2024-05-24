import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LeNet_5_Caffe(nn.Module):
    """LeNet-5 without padding in the first layer.
    This is based on Caffe's implementation of Lenet-5 and is slightly different
    from the vanilla LeNet-5. Note that the first layer does NOT have padding
    and therefore intermediate shapes do not match the official LeNet-5.
    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """

    def __init__(self, save_features=None, bench_model=False):
        super().__init__()
        #self.conv1 = nn.Conv2d(1, 20, 5, padding=1, bias=True) #making kernel as 1
        #self.conv2 = nn.Conv2d(20, 50, 5, bias=True) #making kernel as 1
        #self.fc3 = nn.Linear(50 * 4 * 4, 500) # make it 7 from 4 for 1 x 1 
                                              #  conv in the earlier layers
        self.conv1 = nn.Conv2d(1, 20, 5, padding=1, stride=5 ,bias=True)
        self.conv2 = nn.Conv2d(20, 50, 5, padding=1, bias=True)
        self.fc3 = nn.Linear(50 * 4 * 4, 500)
        self.fc4 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print(x.shape)
        #x = F.max_pool2d(x, 2)
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        #x = F.max_pool2d(x, 2)
        #print(x.shape)
        x = F.relu(self.fc3(x.view(-1, 50 * 4 * 4))) # 4 * 4 to 7*7 with 1x1 all conv, and 6*6 with 5 and stride 5.
        x = F.log_softmax(self.fc4(x), dim=1)

        return x