"""
Process network class which takes of setup training and inference
"""
import os

import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from radioNN.networks.antenna_resnet_network import AntennaNetworkResNet
from radioNN.dataloader import AntennaDataset, custom_collate_fn
from radioNN.networks.antenna_cnn_network import AntennaNetworkCNN
from radioNN.networks.antenna_fc_network import AntennaNetworkFC


class NetworkProcess:
    def __init__(self, percentage=100, one_shower=None):
        """
        Create the classes to be processed while training the network.

        Parameters
        ----------
        percentage: Percentage of data to be used.
        one_shower: if not None, use only the shower of given number.

        Returns
        -------
        criterion: Loss function
        dataloader: Dataloader Class to load data.
        device: cpu or gpu
        model: Model Class
        optimizer: Optimization Algorithm
        """
        radio_data_path = "/home/sampathkumar/radio_data"
        memmap_mode = "r"
        if not os.path.exists(radio_data_path):
            radio_data_path = "/home/pranav/work-stuff-unsynced/radio_data"
            memmap_mode = "r"
        assert os.path.exists(radio_data_path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if one_shower is not None:
            print(
                f"Using the data from {radio_data_path} in {self.device} with "
                f"memmap "
                f"mode: {memmap_mode} using only shower {one_shower}"
            )
        else:
            print(
                f"Using the data from {radio_data_path} in {self.device} with "
                f"memmap "
                f"mode: {memmap_mode} using {percentage}% of data"
            )
        input_data_file = os.path.join(radio_data_path, "input_data.npy")
        input_meta_file = os.path.join(radio_data_path, "meta_data.npy")
        antenna_pos_file = os.path.join(radio_data_path, "antenna_pos_data.npy")
        output_meta_file = os.path.join(radio_data_path, "output_meta_data.npy")
        output_file = os.path.join(radio_data_path, "output_gece_data.npy")
        self.dataset = AntennaDataset(
            input_data_file,
            input_meta_file,
            antenna_pos_file,
            output_meta_file,
            output_file,
            mmap_mode=memmap_mode,
            percentage=percentage,
            one_shower=one_shower,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=64,
            shuffle=True,
            num_workers=4,
            collate_fn=custom_collate_fn,
        )
        self.output_channels = self.dataset.output.shape[-1]
        print(self.output_channels)
        assert 2 <= self.output_channels <= 3
        self.model = AntennaNetworkResNet(self.output_channels).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-1)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, verbose=True
        )

    def train(self, loss_obj=False):
        """
        Train the given model using given data, criteria and optimizer

        Parameters
        ----------
        model: Model Class
        dataloader: Dataloader Class to load data.
        criterion: Loss function
        optimizer: Optimization Algorithm
        device: cpu or gpu
        loss_obj: If True, just return a single batch loss.

        Returns
        -------

        """
        running_loss = 0.0
        valid_batch_count = 0

        for batch in tqdm.tqdm(self.dataloader, leave=False):
            if batch is None:
                tqdm.tqdm.write(f"Skipped batch {batch}")
                continue

            event_data, meta_data, antenna_pos, output_meta, output = batch
            # Event shape is flipped. It should be [batch, 300, 7] but it is
            # [batch, 7, 300].
            # TODO: Fix it in the input file and stop swapaxes.
            event_data = torch.swapaxes(event_data, 1, 2)
            event_data, meta_data, antenna_pos = (
                event_data.to(self.device),
                meta_data.to(self.device),
                antenna_pos.to(self.device),
            )
            output_meta, output = output_meta.to(self.device), output.to(
                self.device
            )

            self.optimizer.zero_grad()

            pred_output_meta, pred_output = self.model(
                event_data, meta_data, antenna_pos
            )

            loss_meta = self.criterion(pred_output_meta, output_meta)
            loss_output = self.criterion(pred_output, output)
            loss = loss_output  # + loss_meta

            if loss_obj:
                return loss
            loss.backward()
            self.optimizer.step()

            if valid_batch_count % 100 == 0 and valid_batch_count != 0:
                tqdm.tqdm.write(f"Batch Loss is {loss.item()}")

            running_loss += loss.item()

            valid_batch_count += 1

        return running_loss / valid_batch_count
