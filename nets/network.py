from .resnet import *
import torch.nn as nn
import torch
from .globalNet import globalNet
from .refineNet import refineNet

__all__ = ['CPN50', 'CPN101', 'CPN18', 'CPN18_global']

class CPN(nn.Module):
    def __init__(self, resnet, output_shape, num_class, pretrained=True):
        super(CPN, self).__init__()
        channel_settings = [2048, 1024, 512, 256]     # resnet50
        # channel_settings = [512, 256, 128, 64]        # 要跟resnet输出的四层feature map 的channel数对应            resnet18
        # channel_settings = [512, 256, 256, 128]          # vgg
        self.resnet = resnet
        self.global_net = globalNet(channel_settings, output_shape, num_class)
        self.refine_net = refineNet(channel_settings[-1], output_shape, num_class)

    def forward(self, x):
        res_out = self.resnet(x)
        global_fms, global_outs = self.global_net(res_out)
        refine_out = self.refine_net(global_fms)

        return global_outs, refine_out

class CPN_global(nn.Module):
    def __init__(self, resnet, output_shape, num_class, pretrained=True):
        super(CPN_global, self).__init__()
        # channel_settings = [2048, 1024, 512, 256]
        channel_settings = [512, 256, 128, 64]        # 要跟resnet输出的四层feature map 的channel数对应
        self.resnet = resnet
        self.global_net = globalNet(channel_settings, output_shape, num_class)

    def forward(self, x):
        res_out = self.resnet(x)
        global_fms, global_outs = self.global_net(res_out)

        return global_outs

def CPN50(out_size,num_class,pretrained=True):
    res50 = resnet50(pretrained=pretrained)
    model = CPN(res50, output_shape=out_size,num_class=num_class, pretrained=pretrained)
    return model

def CPN101(out_size,num_class,pretrained=True):
    res101 = resnet101(pretrained=pretrained)
    model = CPN(res101, output_shape=out_size,num_class=num_class, pretrained=pretrained)
    return model

def CPN18(out_size,num_class,pretrained=True):
    res18 = resnet18(pretrained=pretrained)
    model = CPN(res18, output_shape=out_size,num_class=num_class, pretrained=pretrained)
    return model


def CPN18_global(out_size,num_class,pretrained=True):
    res18 = resnet18(pretrained=pretrained)
    model = CPN_global(res18, output_shape=out_size,num_class=num_class, pretrained=pretrained)
    return model

