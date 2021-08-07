'''
get EfficientNet or SE_ResNext
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
from efficientnet_pytorch import EfficientNet
import resnest.torch as resnest_torch

class Identity(nn.Module):
    '''
    inentity
    '''

    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class ResModel(nn.Module):

    def __init__(self, arch, pretrained="imagenet"):
        super(ResModel, self).__init__()
        self.arch = arch
        self.pretrained = pretrained
        self.model = pretrainedmodels.__dict__[self.arch](pretrained=self.pretrained)
        # self.n_feats = model.last_linear.in_features
        self.l_0 = nn.Linear(2048, 168)
        self.l_1 = nn.Linear(2048, 11)
        self.l_2 = nn.Linear(2048, 7)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l_0 = self.l_0(x)
        l_1 = self.l_1(x)
        l_2 = self.l_2(x)
        return l_0, l_1, l_2


class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        self.in_channels = 1
        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
                                                pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class Efficient(nn.Module):
    '''
    Efficient
    '''

    def __init__(self, arch, pretrained):

        super(Efficient, self).__init__()
        self.arch = arch
        if pretrained:
            self.model = EfficientNet.from_pretrained(self.arch)
        else:
            self.model = EfficientNet.from_name(self.arch)
        # self.n_feats = model.last_linear.in_features
        self.l_0 = nn.Linear(1000, 168)
        self.l_1 = nn.Linear(1000, 11)
        self.l_2 = nn.Linear(1000, 7)

    def forward(self, x):
        # bs, _, _, _= x.shape
        x = self.model(x)
        # x = F.adaptive_avg_pool1d(x, output_size =(bs, 1)) #.reshape(bs, -1)
        l_0 = self.l_0(x)
        l_1 = self.l_1(x)
        l_2 = self.l_2(x)
        return l_0, l_1, l_2


def get_seresnext(arch: str, pretrained: str = "imagenet", channels: str = 'RGB'):

    assert channels in ['RGB', 'GREY']
    model = ResModel(arch, pretrained=pretrained)

    if channels == 'RGB':
        return model
    elif channels == 'GREY':
        model_grey = model
        # Sum over the weights to convert the kernel
        weight_grey = model.model.layer0.conv1.weight.sum(dim=1, keepdim=True)
        bias = model_grey.model.layer0.conv1.bias
        # Instantiate a new convolution module and set weights
        model_grey.model.layer0.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7),
                                                        stride=(2, 2), padding=(3, 3),
                                                        bias=False)
        model_grey.model.layer0.conv1.weight = torch.nn.Parameter(weight_grey)
        model_grey.model.layer0.conv1.bias = bias
        # model_grey.layer0.relu1 = Mish()
        # model_grey
        return model_grey
    else:
        print('wrong cannels')


def get_efficient(arch: str, pretrained: str = "imagenet", channels: str = 'RGB'):

    assert channels in ['RGB', 'GREY']
    model = Efficient(arch, pretrained)

    if channels == 'RGB':
        return model
    elif channels == 'GREY':
        model_grey = model
        # Sum over the weights to convert the kernel
        weight_grey = model.model._conv_stem.weight.sum(dim=1, keepdim=True)
        bias = model_grey.model._conv_stem.bias
        # Instantiate a new convolution module and set weights
        model_grey.model._conv_stem = Conv2dStaticSamePadding(in_channels=1,
                                                              out_channels=48,
                                                              image_size=128,
                                                              kernel_size=(3, 3),
                                                              stride=(2, 2),
                                                              bias=False,)
        # padding=(0, 1, 0, 1))
        model_grey.model._conv_stem.weight = torch.nn.Parameter(weight_grey)
        model_grey.model._conv_stem.bias = bias

        return model_grey
    else:
        print('wrong cannels')

def get_lr_seresnext(model: object, head_lr: float = 0.006, reduce_: float = 0.8):

    lr = [
        {'params': model.model.layer0.parameters(), 'lr': head_lr * reduce_ * reduce_}, 
        {'params': model.model.layer1.parameters(), 'lr': head_lr * reduce_}, 
        {'params': model.model.layer2.parameters(), 'lr': head_lr * reduce_}, 
        {'params': model.model.layer3.parameters(), 'lr': head_lr * reduce_ * .3}, 
        {'params': model.model.layer4.parameters(), 'lr': head_lr * reduce_ * .3}, 
        {'params': model.l0.parameters(), 'lr': head_lr},
        {'params': model.l1.parameters(), 'lr': head_lr},
        {'params': model.l2.parameters(), 'lr': head_lr}]

    return lr   


def get_resnest(model='resnest50_fast_1s1x64d', pretrained=True, n_classes=264):
    model = getattr(resnest_torch, model)(pretrained=pretrained)
    del model.fc
    # # use the same head as the baseline notebook.
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, n_classes))
    
    return model


def get_lr_resnest(model: object, head_lr: float = 0.001, reduce: float = 0.3):
    
    lr = [
        {'params': model.conv1.parameters(), 'lr': head_lr * reduce * reduce * reduce}, 
        {'params': model.layer1.parameters(), 'lr': head_lr * reduce * reduce}, 
        {'params': model.layer2.parameters(), 'lr': head_lr * reduce * reduce}, 
        {'params': model.layer3.parameters(), 'lr': head_lr * reduce}, 
        {'params': model.layer4.parameters(), 'lr': head_lr * reduce}, 
        {'params': model.fc.parameters(), 'lr': head_lr}]

    return lr  

def main():

    model_r = get_seresnext("se_resnext50_32x4d")
    model_e = get_efficient('efficientnet-b0')
    input_ = torch.empty(64, 3, 128, 128)
    print(model_e(input_)[0], model_r(input_)[0])


if __name__ == '__main__':
    main()
