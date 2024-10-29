import torch
import torch.nn as nn
import numpy as np
import copy

class ResBlockDilated(nn.Module):
    def __init__(self, size, hidden = 64, stride = 1, dil = 2):
        super(ResBlockDilated, self).__init__()
        pad_len = dil 
        self.res = nn.Sequential(
                        nn.Conv2d(hidden, hidden, size, padding = pad_len, 
                            dilation = dil),
                        nn.BatchNorm2d(hidden),
                        nn.ReLU(),
                        nn.Conv2d(hidden, hidden, size, padding = pad_len,
                            dilation = dil),
                        nn.BatchNorm2d(hidden),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x 
        res_out = self.res(x)
        out = self.relu(res_out + identity)
        return out

class Decoder(nn.Module):
    def __init__(self, in_channel, hidden = 256, filter_size = 3, num_blocks = 1):
        super(Decoder, self).__init__()
        self.filter_size = filter_size

        self.conv_start = nn.Sequential(
                                    nn.Conv2d(in_channel, hidden, 3, 1, 1),
                                    nn.BatchNorm2d(hidden),
                                    nn.ReLU(),
                                    )
        self.res_blocks1 = self.get_res_blocks(num_blocks, hidden)
        self.res_blocks2 = self.get_res_blocks(num_blocks, hidden//2)
        self.res_blocks3 = self.get_res_blocks(num_blocks, hidden//4)
        self.res_blocks4 = self.get_res_blocks(num_blocks, hidden//8)
        self.res_blocks5 = self.get_res_blocks(num_blocks, hidden//16)

        self.res_blocks9 = self.get_res_blocks(num_blocks, hidden)
        self.res_blocks8 = self.get_res_blocks(num_blocks, hidden//2)
        self.res_blocks7 = self.get_res_blocks(num_blocks, hidden//4)
        self.res_blocks6 = self.get_res_blocks(num_blocks, hidden//8)


        self.conv_mid1 = nn.Sequential(
                                    nn.Conv2d(hidden, hidden//2, 5, 1, padding=2),
                                    nn.BatchNorm2d(hidden//2),
                                    nn.ReLU(),
                                    )
        self.conv_mid2 = nn.Sequential(
                                    nn.Conv2d(hidden//2, hidden//4, 5, 1, padding=2),
                                    nn.BatchNorm2d(hidden//4),
                                    nn.ReLU(),
                                    )
        self.conv_mid3 = nn.Sequential(
                                    nn.Conv2d(hidden//4, hidden//8, 5, 1, padding=2),
                                    nn.BatchNorm2d(hidden//8),
                                    nn.ReLU(),
                                    )
        self.conv_mid4 = nn.Sequential(
                                    nn.Conv2d(hidden//8, hidden//16, 5, 1, padding=2),
                                    nn.BatchNorm2d(hidden//16),
                                    nn.ReLU(),
                                    )

        self.conv_mid5 = nn.Sequential(
                                    nn.Conv2d(hidden//16, hidden//8, 5, 1, padding=2),
                                    nn.BatchNorm2d(hidden//8),
                                    nn.ReLU(),
                                    )
        self.conv_mid6 = nn.Sequential(
                                    nn.Conv2d(hidden//8, hidden//4, 5, 1, padding=2),
                                    nn.BatchNorm2d(hidden//4),
                                    nn.ReLU(),
                                    )
        self.conv_mid7 = nn.Sequential(
                                    nn.Conv2d(hidden//4, hidden//2, 5, 1, padding=2),
                                    nn.BatchNorm2d(hidden//2),
                                    nn.ReLU(),
                                    )
        self.conv_mid8 = nn.Sequential(
                                    nn.Conv2d(hidden//2, hidden, 5, 1, padding=2),
                                    nn.BatchNorm2d(hidden),
                                    nn.ReLU(),
                                    )

        self.conv_end = nn.Conv2d(hidden, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, input_data):
        x0 = self.conv_start(input_data) # i-h
        x1 = self.res_blocks1(x0) # h-h
        m1 = self.conv_mid1(x1)   # h/2
        x2 = self.res_blocks2(m1) # h/2
        m2 = self.conv_mid2(x2)   # h/4
        x3 = self.res_blocks3(m2)
        m3 = self.conv_mid3(x3)   # h/8
        x4 = self.res_blocks4(m3)
        m4 = self.conv_mid4(x4)   # h/16
        x5 = self.res_blocks5(m4)   # h/16

        m5 = self.conv_mid5(x5)
        x6 = self.res_blocks6(m5)   # h/8
        r1 = x4+x6
        m6 = self.conv_mid6(r1)
        x7 = self.res_blocks7(m6)   # h/4
        r2 = x3+x7
        m7 = self.conv_mid7(r2)
        x8 = self.res_blocks8(m7)   # h/2
        r3 = x2+x8
        m8 = self.conv_mid8(r3)
        x9 = self.res_blocks9(m8)   # h
        r4 = x1+x9

        out = self.conv_end(r4)
        return out

    def get_res_blocks(self, n, hidden):
        blocks = []
        for i in range(n):
            dilation = 2 ** (i + 1)
            blocks.append(ResBlockDilated(self.filter_size, hidden = hidden, dil = dilation))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks