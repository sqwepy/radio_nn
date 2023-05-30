import torch
from torch import nn
from radioNN.cnn_module import EventDataCNN


class AntennaNetwork(nn.Module):
    """Antenna pulse generation network."""

    def __init__(self, output_channels):
        super().__init__()

        self.event_data_cnn = EventDataCNN()

        # Calculate the output size of the CNN module
        cnn_output_size = 256 * (300 // 2 // 2 // 2 // 2)

        self.fc_layers = nn.Sequential(
            nn.Linear(cnn_output_size + 12 + 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 2 + 2),
        )

        self.fc_meta = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )

        self.deconv = nn.Sequential(nn.ConvTranspose1d(2, output_channels, 1))

    def forward(self, event_data, meta_data, antenna_pos):
        """Forward pass which is called at model(data)."""
        event_data = self.event_data_cnn(event_data)
        event_data = event_data.view(event_data.size(0), -1)
        combined_input = torch.cat((event_data, meta_data, antenna_pos), dim=1)
        combined_output = self.fc_layers(combined_input)

        # Separate the output
        antenna_output_meta = combined_output[:, :2].unsqueeze(1)
        antenna_output_meta = self.fc_meta(antenna_output_meta)
        antenna_output = combined_output[:, 2:].view(-1, 2, 256)
        antenna_output = self.deconv(antenna_output)
        antenna_output = torch.swapaxes(antenna_output, 1, 2)
        return antenna_output_meta, antenna_output
