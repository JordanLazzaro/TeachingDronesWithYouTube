"""
This is an implementation of a residual block for use throughout the rest of the
project.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):

    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.resblock_layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(output_channels))

    def forward(self, input):
        output = self.resblock_layers(input) + input

        return F.relu(output, inplace=True)
