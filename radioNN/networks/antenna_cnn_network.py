"""Antenna Pulse Generation Network."""

import torch
from torch import nn


class EventDataCNN(nn.Module):
    """CNN part of the AntennaNetwork."""

    def __init__(self) -> None:
        self.padding = 0
        self.kernel_size = 1
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(7, 32, self.kernel_size, padding=self.padding),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, self.kernel_size, padding=self.padding),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, self.kernel_size, padding=self.padding),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, self.kernel_size, padding=self.padding),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        """Forward pass called when using model(data)."""
        x = self.conv_layers(x)
        return x


class EventDataCNNDeConv(nn.Module):
    """CNN part of the AntennaNetwork."""

    def __init__(self, input_channels, output_channels) -> None:
        self.padding = 0
        self.kernel_size = 1
        super().__init__()

        self.de_conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, self.kernel_size, padding=self.padding),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, output_channels, self.kernel_size, padding=self.padding),
        )

    def forward(self, x):
        """Forward pass called when using model(data)."""
        x = self.de_conv_layers(x)
        return x


class AntennaNetworkCNN(nn.Module):
    """Antenna pulse generation network."""

    def __init__(self, output_channels) -> None:
        super().__init__()

        self.output_channels = output_channels
        self.event_data_cnn = EventDataCNN()
        self._meta_size = 8
        self._antenna_pos_size = 3
        # Calculate the output size of the CNN module
        cnn_output_size = 256 * (1142 // 2 // 2 // 2 // 2)  #here is the problem 1142 because of grammage steps (small dataset) adn 1015 (big dataset)

        self.cnn_inner_channels = 2
        self.fc_layers = nn.Sequential(
            nn.Linear(cnn_output_size + self._meta_size + self._antenna_pos_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * self.cnn_inner_channels + 2),
        )

        self.fc_meta = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )

        self.deconv = EventDataCNNDeConv(self.cnn_inner_channels, self.output_channels)

    def forward(self, event_data, meta_data, antenna_pos):
        """Forward pass which is called at model(data)."""
        meta_indices = [
            #    0, #"Sim number"
            1,  # "Cos(Zenith Angle)"
            2,  # "X_max"
            3,  # "density at X_max"
            4,  # "height at X_max"
            5,  # "E_em"
            6,  # "sin(geomagnetic_angle)"
            #    7, #"B inclination"
            #    8, #"B declination"
            #    9, #"B strength"
            #    10, #"primary particle"
            11,  # "primary energy"
            12,  # "Azimuthal angle"
        ]
        assert self._meta_size == len(meta_indices)
        assert self._antenna_pos_size == antenna_pos.shape[-1]
        event_data = self.event_data_cnn(event_data)
        event_data = event_data.reshape(event_data.size(0), -1)
        combined_input = torch.cat(
            (event_data, meta_data[:, meta_indices], antenna_pos), dim=1
        )
        combined_output = self.fc_layers(combined_input)

        # Separate the output
        antenna_output_meta = combined_output[:, :2]
        antenna_output_meta = self.fc_meta(antenna_output_meta)
        antenna_output = combined_output[:, 2:].reshape(
            -1, self.cnn_inner_channels, 256
        )
        antenna_output = self.deconv(antenna_output)
        antenna_output = torch.swapaxes(antenna_output, 1, 2)
        return antenna_output_meta, antenna_output
