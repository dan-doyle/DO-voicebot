import torch
from torch import nn
from torch.nn.utils import weight_norm

class NormReLUChannelNormalization(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(NormReLUChannelNormalization, self).__init__()
        self.epsilon = epsilon
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        max_values, _ = torch.max(torch.abs(x), dim=2, keepdim=True)
        max_values += self.epsilon
        out = x / max_values
        return out

class WaveNetActivation(nn.Module):
    def __init__(self):
        super(WaveNetActivation, self).__init__()

    def forward(self, x):
        tanh_out = torch.tanh(x)
        sigm_out = torch.sigmoid(x)
        return tanh_out * sigm_out

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size, activation, dropout=0):
        super(ResidualBlock, self).__init__()
        chomp_size = (kernel_size-1) * dilation
        padding = (kernel_size-1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                           stride=1, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(chomp_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                           stride=1, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(chomp_size)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.activation, self.dropout,
                                 self.conv2, self.chomp2, self.activation, self.dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(out_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = in_channels if i == 0 else out_channels[i-1]
            activation = NormReLUChannelNormalization() if i%2 == 0 else WaveNetActivation()
            layers += [ResidualBlock(in_channels, out_channels[i], dilation=dilation_size,
                                     kernel_size=kernel_size, activation=activation, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class PatternModel(nn.Module):
    def __init__(self, in_channels=768, out_channels=[1024, 768, 384], kernel_size=2, dropout=0):
        super(PatternModel, self).__init__()
        self.tcn = TemporalConvNet(in_channels, out_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(out_channels[-1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        y1 = self.tcn(x)
        o = self.linear(y1[:, :, -1])
        return o