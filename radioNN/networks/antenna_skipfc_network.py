import torch
from torch import nn


class SkipBlock(nn.Module):
    def __init__(self, dimension, repeat=1):
        super().__init__()
        self.dimension = dimension
        self.repeat = repeat

        self.model = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        output = x
        for _ in range(self.repeat):
            output = self.model(x) + x
        return output


class AntennaNetworkSkipFC(nn.Module):
    """Antenna pulse generation network."""

    def __init__(self, output_channels):
        super().__init__()

        # Calculate the output size of the CNN module
        input_sequence_size = 7 * 300

        self.fc_layers_encode = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            SkipBlock(1024, 2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            SkipBlock(512, 2),
            nn.Linear(512, 256 * 2),
            nn.LeakyReLU(),
        )

        self.fc_meta = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.ReLU(),
        )

        self.fc_layers_decode = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.LeakyReLU(),
            SkipBlock(512, 2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            SkipBlock(1024, 2),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256 * output_channels),
        )

    def forward(self, event_data, meta_data, antenna_pos):
        """Forward pass which is called at model(data)."""
        event_data = event_data.reshape(event_data.size(0), -1)
        print(meta_data)
        # combined_input = torch.cat((antenna_pos), dim=1)
        combined_output = self.fc_layers_encode(antenna_pos)

        # Separate the output
        antenna_output_meta = self.fc_meta(combined_output)
        antenna_output = self.fc_layers_decode(combined_output)
        antenna_output = antenna_output.reshape(-1, 256, 2)
        return antenna_output_meta, antenna_output
