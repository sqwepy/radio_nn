import torch
from torch import nn


class AntennaNetworkFC(nn.Module):
    """Antenna pulse generation network."""

    def __init__(self, output_channels):
        super().__init__()

        # Calculate the output size of the CNN module
        input_sequence_size = 7 * 300

        self.fc_layers_encode = nn.Sequential(
            nn.Linear(input_sequence_size + 12 + 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 2 + 2),
            nn.Tanh(),
        )

        self.fc_meta = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )

        self.fc_layers_decode = nn.Sequential(
            nn.Linear(256 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * output_channels),
            nn.Tanh(),
        )

    def forward(self, event_data, meta_data, antenna_pos):
        """Forward pass which is called at model(data)."""
        event_data = event_data.reshape(event_data.size(0), -1)
        combined_input = torch.cat((event_data, meta_data, antenna_pos), dim=1)
        combined_output = self.fc_layers_encode(combined_input)

        # Separate the output
        antenna_output_meta = combined_output[:, :2]
        antenna_output_meta = self.fc_meta(antenna_output_meta)
        antenna_output = combined_output[:, 2:]
        antenna_output = self.fc_layers_decode(antenna_output)
        antenna_output = antenna_output.reshape(-1, 256, 2)
        return antenna_output_meta, antenna_output
