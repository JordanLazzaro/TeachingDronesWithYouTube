"""
tdc_embedding.py

This file is an implementation of the Temporal Distance Classification (TDC)
embedding network "φ" specified in "Playing Hard Exploration Games by Watching
YouTube Videos" by Aytar et al.

--------------------------------------------------------------------------------

This network learns a function φ: I -> R^(N) which maps video frames from set I
to an N-Dimensional embedding space R^(N).

The weights are learned via an auxiliary task of Temporal Distance
Classification (TDC), where the embeddings are used by another function
('TDC_Classifier' in tdc_classifier.py) to predict the temporal distance of two
frames of the same video.

--------------------------------------------------------------------------------

Specs for this network are described in the paper here:

"The visual embedding function, φ, is composed of three spatial, padded,
3x3 convolutional layers with (32, 64, 64) channels and 2x2 max-pooling,
followed by three residual-connected blocks with 64 channels and no
down-sampling. Each layer is ReLU-activated and batch-normalized, and the output
fed into a 2-layer 1024-wide MLP. The network input is a 128x128x3x4 tensor
constructed by random spatial cropping of a stack of four consecutive
140x140 RGB images, sampled from our dataset. The final embedding vector is
l2-normalized."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ResBlock

class TDC_Embedding(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(TDC_Embedding, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),

            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64))

        self.fully_connected = nn.Sequential(
            nn.Linear(16384, 1024),
            nn.Linear(1024, output_channels))

    def forward(self, input):
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.fully_connected(output)

        return output
