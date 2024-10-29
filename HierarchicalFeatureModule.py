import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TopKPoolingLayer(nn.Module):
    def __init__(self, kernel_size, stride):
        super(TopKPoolingLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input_sequence):
        output_sequence = F.max_pool1d(input_sequence, kernel_size=self.kernel_size, stride=self.stride,ceil_mode=True)

        return output_sequence

class Final_TopKPoolingLayer(nn.Module):
    def __init__(self, num_patches, dim=2):
        super(Final_TopKPoolingLayer, self).__init__()
        self.dim = dim
        self.num_patches = num_patches

    def forward(self, input_sequence):
        patch_length = math.ceil(input_sequence.size(self.dim) / self.num_patches)
        remainder = input_sequence.size(self.dim) % self.num_patches
        if remainder != 0:
            padding_size = patch_length * self.num_patches - input_sequence.size(self.dim)
            input_sequence = F.pad(input_sequence, (0, padding_size))

        output_sequence = F.max_pool1d(input_sequence, kernel_size=patch_length, stride=patch_length,ceil_mode=True)

        return output_sequence

class EqualSizeConv1dWithStride(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride):
        super(EqualSizeConv1dWithStride, self).__init__()
        self.conv1d = nn.Conv1d(input_channels, output_channels, kernel_size, stride=stride, padding=((kernel_size-1)//2)*stride)

    def forward(self, x):
        return self.conv1d(x)

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        self.scale = nn.Sequential(
                        EqualSizeConv1dWithStride(input_channels, output_channels, kernel_size, stride),
                        nn.BatchNorm1d(output_channels),
                        nn.ReLU(),
                        )
        self.res = nn.Sequential(
                        EqualSizeConv1dWithStride(output_channels, output_channels, kernel_size, stride),
                        nn.BatchNorm1d(output_channels),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        scaled = self.scale(x)
        identity = scaled
        res_out = self.res(scaled)
        out = self.relu(res_out + identity)
        return out

class PatchSamplingModule(nn.Module):
    def __init__(self, input_channels, output_channels, patch_length, topk, kernel_size, stride):
        super(PatchSamplingModule, self).__init__()
        self.convolution = ConvBlock(input_channels, output_channels, kernel_size, stride)
        self.pooling = TopKPoolingLayer(patch_length, patch_length)
        self.patch_length = patch_length

    def forward(self, x):
        x = self.convolution(x)
        x = self.pooling(x)
        return x

class Final_PatchSamplingModule(nn.Module):
    def __init__(self, input_channels, output_channels, patch_length, kernel_size, stride):
        super(Final_PatchSamplingModule, self).__init__()
        self.convolution = ConvBlock(input_channels, output_channels, kernel_size, stride)
        self.pooling = Final_TopKPoolingLayer(patch_length, dim=2)
        self.patch_length = patch_length

    def forward(self, x):
        x = self.convolution(x)
        x = self.pooling(x)
        return x

class PatchSampledModel(nn.Module):
    def __init__(self, input_channels, output_channels, patch_length_1, patch_length_2, topk, kernel_size, stride):
        super(PatchSampledModel, self).__init__()

        self.module_1 = PatchSamplingModule(input_channels, output_channels, patch_length_1, topk, kernel_size, stride)
        self.module_2 = PatchSamplingModule(output_channels, output_channels*2, patch_length_1, topk, kernel_size, stride)
        self.module_3 = PatchSamplingModule(output_channels*2, output_channels*4, patch_length_1, topk, kernel_size, stride)

        self.final_samplemodule = Final_PatchSamplingModule(output_channels*4, output_channels*8, patch_length_2, kernel_size, stride)

    def forward(self, x):
        x=self.module_1(x)
        x=self.module_2(x)
        x=self.module_3(x)
        x=self.final_samplemodule(x)

        return x