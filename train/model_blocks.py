#%% Test Stuff
import numpy as np, torch
import torchvision.models as models
from torch import nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F

#%% Custom Model
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Dense_Layer(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(Dense_Layer, self).__init__()

        self.linear = nn.Linear(in_size, out_size)
        self.batch_norm = nn.BatchNorm1d(out_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.batch_norm(self.linear(x)))
        return x

class Dense_Net(torch.nn.Module):
    def __init__(self, input_size, out_size_arr):
        super(Dense_Net, self).__init__()
        
        self.layer_names = []
        sizes = [input_size] + out_size_arr

        
        for i in range(len(out_size_arr)):
            
            layer_name = f'layer{i}'
            # dense_layer = Dense_Layer(sizes[i], sizes[i + 1])
            setattr(self, layer_name, Dense_Layer(sizes[i], sizes[i + 1]))
            self.layer_names.append(layer_name)
    
    def forward(self, x):
        for ln in self.layer_names:
            x = getattr(self, ln)(x)
        
        return x


class L_Pad(nn.Module):
    def __init__(self, padding):
        super(L_Pad, self).__init__()
        self.padding = padding
    
    def forward(self, x):
        return F.pad(x, (self.padding, 0))

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        
        self.lpad1 = L_Pad(padding)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.lpad2 = L_Pad(padding)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.lpad1, self.conv1, self.relu1, self.dropout1,
                                 self.lpad2, self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights_kaiming()

    def init_weights_kaiming(self):
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        if self.downsample is not None:
            torch.nn.init.kaiming_normal_(self.downsample.weight)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, input_size, dilation_arr, num_channels_arr, kernel_size, dropout):
        super(TemporalConvNet, self).__init__()
        
        stride = 1

        layers = []
        for i, dilation_size in enumerate(dilation_arr):
            
            padding = (kernel_size - 1) * dilation_size

            in_channels = input_size if i == 0 else num_channels_arr[i-1]
            out_channels = num_channels_arr[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def get_dilation_arr(kernel, levels, seq_len):

    arr = []
    for i in range(levels):
        d = 2**i
        two_kernel_range = 2 + d * (2 - 1)
        
        # when dilating further would be pointless, deepend the array from the bottom
        if two_kernel_range <= seq_len:
            arr.append(d)
        else:
            arr.insert(0, arr[0])
    
    print('dilation_arr\n', arr)
        
    return arr

class TCN(nn.Module):
    def __init__(self, input_size, seq_len, output_size, kernel, levels, channel_size, dropout):
        super(TCN, self).__init__()

        dilation_arr = get_dilation_arr(kernel, levels, seq_len)
        num_channels_arr = [channel_size] * levels

        self.tcn = TemporalConvNet(input_size, dilation_arr, num_channels_arr,
                                    kernel_size=kernel, dropout=dropout)
        
        self.linear = nn.Linear(num_channels_arr[-1], output_size)
        # self.init_weights()

    # def init_weights(self):
    #     torch.nn.init.kaiming_normal_(self.linear)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1[:, :, -1])     

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()