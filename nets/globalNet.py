import torch.nn as nn
import torch
import math

class globalNet(nn.Module):
    def __init__(self, channel_settings, output_shape, num_class):
        super(globalNet, self).__init__()
        self.channel_settings = channel_settings           # 四个尺度的输出对应的channel数
        laterals, upsamples, predict = [], [], []
        for i in range(len(channel_settings)):
            laterals.append(self._lateral(channel_settings[i]))
            predict.append(self._predict(output_shape, num_class))
            if i != len(channel_settings) - 1:      # 除最后一个外，都进行两倍Upsample
                upsamples.append(self._upsample())
        self.laterals = nn.ModuleList(laterals)
        self.upsamples = nn.ModuleList(upsamples)
        self.predict = nn.ModuleList(predict)

        for m in self.modules():    # 卷积层和BN层参数初始化
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _lateral(self, input_size):                              # 1*1卷积+BN+RELU   传入输入通道数，输出256
        layers = []
        layers.append(nn.Conv2d(input_size, self.channel_settings[-1],
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(self.channel_settings[-1]))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _upsample(self):                          # 两倍Upsample + in256 out256 1*1卷积 + BN
        layers = []
        # layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(torch.nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(torch.nn.Conv2d(self.channel_settings[-1], self.channel_settings[-1],
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(self.channel_settings[-1]))

        return nn.Sequential(*layers)

    def _predict(self, output_shape, num_class):  # in256 out256 1*1卷积 + BN + RELU，in256 out17 3*3 s=1卷积 + Upsample至64*48 + BN
        layers = []
        layers.append(nn.Conv2d(self.channel_settings[-1], self.channel_settings[-1],
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(self.channel_settings[-1]))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(self.channel_settings[-1], num_class,
            kernel_size=3, stride=1, padding=1, bias=False))
        # layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.Upsample(size=output_shape, mode='nearest'))
        layers.append(nn.BatchNorm2d(num_class))

        return nn.Sequential(*layers)

    def forward(self, x):
        global_fms, global_outs = [], []
        # global_ups = []
        for i in range(len(self.channel_settings)):
            if i == 0:               # 第一轮，resnet第一个输出接1*1conv，输出256channel
                feature = self.laterals[i](x[i])
            else:
                # print(i)
                # print(up.shape)
                feature = self.laterals[i](x[i]) + up    # resnet的输出经过1*1conv，与前一轮upsample的feature map 逐元素相加
            global_fms.append(feature)
            if i != len(self.channel_settings) - 1:    # 非最后一轮，两倍上采样，输出256channel
                up = self.upsamples[i](feature)
                # global_ups.append(up)
            feature = self.predict[i](feature)         # 每一轮接一个预测输出,每轮输出为 17*64*48
            global_outs.append(feature)

        return global_fms, global_outs                 # global_fms 大小为8*6，16*12，32*24，64*48