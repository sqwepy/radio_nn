import torch
from torch import nn


def make_layer(block, planes, blocks, stride=1, inplanes=64):
    """
    Make Layer by repeating Blocks.

    Parameters
    ----------
    block
    planes
    blocks
    stride
    inplanes

    Returns
    -------

    """
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv1d(
                inplanes,
                planes * block.expansion,
                1,
                stride=stride,
            ),
            nn.BatchNorm1d(planes * block.expansion),
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            downsample,
        )
    )
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
            )
        )

    return nn.Sequential(*layers), inplanes


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, outplanes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            inplanes, outplanes, kernel_size=3, padding=1, stride=stride
        )
        self.bn1 = nn.BatchNorm1d(outplanes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(outplanes, outplanes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(outplanes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class EventDataResCNN(nn.Module):
    """CNN part of the AntennaNetwork."""

    def __init__(self):
        super(EventDataResCNN, self).__init__()
        self.conv1 = nn.Conv1d(
            7, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d(5)
        self.layer1, inplanes = make_layer(BasicBlock, 64, 2)
        self.layer2, inplanes = make_layer(
            BasicBlock, 128, 2, stride=2, inplanes=inplanes
        )
        self.layer3, inplanes = make_layer(
            BasicBlock, 256, 2, stride=2, inplanes=inplanes
        )
        self.layer4, inplanes = make_layer(
            BasicBlock, 512, 2, stride=2, inplanes=inplanes
        )

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class EventDataCNNResDeConv(nn.Module):
    """CNN part of the AntennaNetwork."""

    def __init__(self, input_channels, output_channels):
        super(EventDataCNNResDeConv, self).__init__()
        self.conv1 = nn.Conv1d(
            input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1, inplanes = make_layer(BasicBlock, 64, 2)
        self.layer2, inplanes = make_layer(
            BasicBlock, 128, 2, stride=1, inplanes=inplanes
        )
        self.layer3, inplanes = make_layer(
            BasicBlock, 256, 2, stride=1, inplanes=inplanes
        )
        self.layer4, inplanes = make_layer(
            BasicBlock, 512, 2, stride=1, inplanes=inplanes
        )
        self.conv2 = nn.Conv1d(512, output_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        """Forward pass called when using model(data)."""
        x = self.conv2(x)
        return x


class AntennaNetworkResNet(nn.Module):
    """Antenna pulse generation network."""

    def __init__(self, output_channels):
        super().__init__()

        self.event_data_cnn = EventDataResCNN()

        # Calculate the output size of the CNN module
        cnn_output_size = 2560 # it is 512 * (300 // 2 // 2 // 2 // 2 //2 // 2) but
        # round up

        self.output_channels = output_channels
        self._meta_size = 8
        self._antenna_pos_size = 3
        self.cnn_inner_channels = 2
        self.fc_layers = nn.Sequential(
            nn.Linear(cnn_output_size + self._meta_size + self._antenna_pos_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * self.cnn_inner_channels),
        )

        self.fc_meta = nn.Sequential(
            nn.Linear(256 * self.cnn_inner_channels, 256),
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

        self.deconv = EventDataCNNResDeConv(self.cnn_inner_channels,
                                            self.output_channels)

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
        event_data = event_data.view(event_data.size(0), -1)
        combined_input = torch.cat((event_data, meta_data[:, meta_indices], antenna_pos), dim=1)
        combined_output = self.fc_layers(combined_input)

        # Separate the output
        antenna_output_meta = self.fc_meta(combined_output)
        antenna_output = combined_output.reshape(-1, self.cnn_inner_channels, 256)
        antenna_output = self.deconv(antenna_output)
        antenna_output = torch.swapaxes(antenna_output, 1, 2)
        return antenna_output_meta, antenna_output
