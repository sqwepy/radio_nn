import torch
import torch.nn as nn


class EventDataCNN(nn.Module):
    def __init__(self):
        self.padding = 0
        self.kernel_size = 1
        super(EventDataCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(7, 32,self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64,self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128,self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return x
