import math
import torch
from torchvision.models import vit_b_16

import torch.nn as nn
import torch.nn.init as init
import torch.nn as nn
import torch.nn.init as init

def vit(progress=True):
    model = vit_b_16(progress=progress)
    return model

if __name__ == '__main__':
    model = vit()

    print(model)
    print(model(torch.randn(1, 3, 224, 224)).shape)