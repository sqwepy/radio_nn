import torch
from torch import nn


class AntennaNetworkFC(nn.Module):
    """Antenna pulse generation network."""

    def __init__(self, output_channels=3) -> None:
        super().__init__()

        # Calculate the output size of the CNN module
        self.output_channels = output_channels

        self.fc_layers_encode = nn.Sequential(
            nn.Linear(10, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
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
        )

        self.fc_layers_decode = nn.Sequential(
            nn.Linear(256 * 2, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256 * self.output_channels),
        )

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
            # 11,  # "primary energy"
            12,  # "Azimuthal angle"
        ]
        event_data = event_data.reshape(event_data.size(0), -1)
        # combined_input = torch.cat((event_data, meta_data, antenna_pos), dim=1)
        combined_input = torch.cat(
            (meta_data[:, meta_indices].reshape((-1, len(meta_indices))), antenna_pos),
            dim=1,
        )
        combined_output = self.fc_layers_encode(combined_input)

        # Separate the output
        antenna_output_meta = self.fc_meta(combined_output)
        antenna_output = self.fc_layers_decode(combined_output)
        antenna_output = antenna_output.reshape(-1, 256, self.output_channels)
        return antenna_output_meta, antenna_output
