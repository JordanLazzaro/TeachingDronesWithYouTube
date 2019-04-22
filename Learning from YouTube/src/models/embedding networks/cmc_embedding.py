"""
cmc_embedding.py

This file is an implementation of the Cross-Modal Temporal Distance
Classification (CMC) embedding network "ψ" specified in "Playing Hard
Exploration Games By Watching YouTube Videos" by Aytar et al.

--------------------------------------------------------------------------------

This network learns a function ψ: A -> R^(N) that maps frequency decomposed
audio snippets to an N-Dimensional embedding space R^(N).

The weights are learned via an auxiliary task of Cross-Modal Temporal Distance
Classification (CMC), where the embeddings are used by another function
('CMC_Classifier' in cmc_classifier.py) to predict the temporal distance between
a video frame and an associated audio snippet.

--------------------------------------------------------------------------------

Specs for this network are described in the paper here:

"The audio embedding function, ψ, is as per φ except that it has four, width-8,
1D convolutional layers with (32, 64, 128, 256) channels and 2x max-pooling, and
a single width-1024 linear layer. The input is a width-137 (6ms) sample of 256
frequency channels, calculated using STFT. ReLU-activation and
batch-normalization are applied throughout and the embedding vector is
l2-normalized."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CMC_Embedding(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(CMC_Embedding, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, 8, padding=0),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            nn.ReLU(),

            nn.Conv1d(32, 64, 8, padding=0),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.ReLU(),

            nn.Conv1d(64, 128, 8, padding=0),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            nn.ReLU(),

            nn.Conv1d(128, 256, 8, padding=0),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2),
            nn.ReLU())

        self.fully_connected = nn.Sequential(nn.Linear(512, output_channels))

    def forward(self, input):
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.fully_connected(output)

        return output
