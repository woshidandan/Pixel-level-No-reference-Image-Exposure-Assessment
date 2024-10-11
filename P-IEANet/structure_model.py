import math
import random
import functools
import operator
import itertools
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

class Transformer_Layer(nn.Module):
    def __init__(self, d_model=9, nhead=1, num_layers=1):
        super(Transformer_Layer, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=16)
        self.conv = nn.Conv2d(9, 3, kernel_size=1, stride=1)
        #self.fc = nn.Linear(128*128, 128*128)

    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(1))
        x = self.transformer(x,x)
        x = x.view(x.size(0), 9, 128, 128)
        x = self.conv(x)
        return x

class Conv_Layer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(Conv_Layer, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=0)
        self.padding_size = (1, 0, 1, 0)

    def forward(self, x):
        x = F.pad(x, self.padding_size, value=0)
        x = self.conv(x)
        return x


class Encoder_block(nn.Module):
    def __init__(self, in_channel, out_channel,downsample=False,
                 kernel_size=2, device='cuda:1'):
        super().__init__()

        self.conv0 = Conv_Layer(in_channel, out_channel, kernel_size)
        self.conv1 = Conv_Layer(in_channel, out_channel, kernel_size)
        self.conv2 = Conv_Layer(in_channel, out_channel, kernel_size)
        self.conv3 = Conv_Layer(in_channel, out_channel, kernel_size)
        self.conv4 = Conv_Layer(in_channel, out_channel, kernel_size)
        self.conv5 = Conv_Layer(in_channel, out_channel, kernel_size)
        self.conv6 = Conv_Layer(in_channel, out_channel, kernel_size)
        self.conv7 = Conv_Layer(in_channel, out_channel, kernel_size)
        self.conv8 = Conv_Layer(in_channel, out_channel, kernel_size)

        self.transformer0 = Transformer_Layer(in_channel)
        self.transformer1 = Transformer_Layer(in_channel)
        self.transformer2 = Transformer_Layer(in_channel)
        self.transformer3 = Transformer_Layer(in_channel)
        self.transformer4 = Transformer_Layer(in_channel)
        self.transformer5 = Transformer_Layer(in_channel)
        self.transformer6 = Transformer_Layer(in_channel)
        self.transformer7 = Transformer_Layer(in_channel)
        self.transformer8 = Transformer_Layer(in_channel)

        self.fusion0 = Conv_Layer(out_channel * 2, out_channel, kernel_size)
        self.fusion1 = Conv_Layer(out_channel * 2, out_channel, kernel_size)
        self.fusion2 = Conv_Layer(out_channel * 2, out_channel, kernel_size)
        self.fusion3 = Conv_Layer(out_channel * 2, out_channel, kernel_size)
        self.fusion4 = Conv_Layer(out_channel * 2, out_channel, kernel_size)
        self.fusion5 = Conv_Layer(out_channel * 2, out_channel, kernel_size)
        self.fusion6 = Conv_Layer(out_channel * 2, out_channel, kernel_size)
        self.fusion7 = Conv_Layer(out_channel * 2, out_channel, kernel_size)
        self.fusion8 = Conv_Layer(out_channel * 2, out_channel, kernel_size)

    def forward(self, input):

        height = input.shape[2]
        width = input.shape[3]

        short_range = self.conv0(input)
        #print("short shape:",short_range.shape)
        long_range = self.transformer0(input)
        #print("long shape:",long_range.shape)
        f0 = self.fusion0(torch.cat([short_range, long_range], dim=1))

        input_center = input[:, :, 1:-1, 1:-1]
        left = input[:, :, 0:-2, 1:-1]
        right = input[:, :, 2:, 1:-1]
        top = input[:, :, 1:-1, 0:-2]
        bottom = input[:, :, 1:-1, 2:]

        left_top = input[:, :, 0:-2, 0:-2]
        right_top = input[:, :, 2:, 0:-2]
        left_bottom = input[:, :, 0:-2, 2:]
        right_bottom = input[:, :, 2:, 2:]


        input1 = F.interpolate(input_center-left, size=(height, width))
        f11 = self.conv1(input1)
        f12 = self.transformer1(input1)
        f1 = self.fusion1(torch.cat([f11, f12], dim=1))

        ########
        input2 = F.interpolate(input_center - right, size=(height, width))
        f21 = self.conv2(input2)
        f22 = self.transformer2(input2)
        f2 = self.fusion2(torch.cat([f21, f22], dim=1))

        ########
        input3 = F.interpolate(input_center - top, size=(height, width))
        f31 = self.conv3(input3)
        f32 = self.transformer3(input3)
        f3 = self.fusion3(torch.cat([f31, f32], dim=1))

        ########
        input4 = F.interpolate(input_center - bottom, size=(height, width))
        f41 = self.conv4(input4)
        f42 = self.transformer4(input4)
        f4 = self.fusion4(torch.cat([f41, f42], dim=1))

        ########
        input5 = F.interpolate(input_center - left_top, size=(height, width))
        f51 = self.conv5(input5)
        f52 = self.transformer5(input5)
        f5 = self.fusion5(torch.cat([f51, f52], dim=1))

        ########
        input6 = F.interpolate(input_center - right_top, size=(height, width))
        f61 = self.conv6(input6)
        f62 = self.transformer6(input6)
        f6 = self.fusion6(torch.cat([f61, f62], dim=1))

        ########
        input7 = F.interpolate(input_center - left_bottom, size=(height, width))
        f71 = self.conv7(input7)
        f72 = self.transformer7(input7)
        f7 = self.fusion7(torch.cat([f71, f72], dim=1))

        ########
        input8 = F.interpolate(input_center - right_bottom, size=(height, width))
        f81 = self.conv8(input8)
        f82 = self.transformer8(input8)
        f8 = self.fusion8(torch.cat([f81, f82], dim=1))

        f = f1+f2+f3+f4+f5+f6+f7+f8

        #f = f1 + f2 + f3 + f4

        out = f0+f
        return out
