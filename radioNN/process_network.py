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


def fit_plane_and_return_3d_grid(pos):
    import numpy as np
    from scipy.sparse.linalg import lsqr

    design_matrix = np.ones_like(pos)
    design_matrix[:, :2] = pos[:, :2]
    fit = lsqr(design_matrix, pos[:, 2])[0]
    xs = np.linspace(np.min(pos[:, 0]), np.max(pos[:, 0]), 100)
    ys = np.linspace(np.min(pos[:, 1]), np.max(pos[:, 1]), 100)
    X, Y = np.meshgrid(xs, ys)
    Z = fit[0] * X + fit[1] * Y + fit[2]
    return np.dstack([X, Y, Z]).reshape(-1, 3)


class NetworkProcess:
    def __init__(
        self,
        percentage=100,
        one_shower=None,
        model_class=AntennaNetworkFC,
        batch_size=4,
    ):
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
        self.input_data_file = os.path.join(radio_data_path, "input_data.npy")
        self.input_meta_file = os.path.join(radio_data_path, "meta_data.npy")
        self.antenna_pos_file = os.path.join(
            radio_data_path, "antenna_pos_data.npy"
        )
        self.output_meta_file = os.path.join(
            radio_data_path, "output_meta_data.npy"
        )
        self.output_file = os.path.join(radio_data_path, "output_gece_data.npy")
        self.dataset = AntennaDataset(
            self.input_data_file,
            self.input_meta_file,
            self.antenna_pos_file,
            self.output_meta_file,
            self.output_file,
            mmap_mode=memmap_mode,
            percentage=percentage,
            one_shower=one_shower,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=custom_collate_fn,
        )
        self.output_channels = self.dataset.output.shape[-1]
        print(self.output_channels)
        assert 2 <= self.output_channels <= 3
        print(f"Using {model_class}")
        self.model = model_class(self.output_channels).to(self.device)
        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            verbose=True,
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

        for batch in tqdm.autonotebook.tqdm(self.dataloader, leave=False):
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
            loss_output = self.criterion(250 * pred_output, 250 * output)
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

    def pred_one_shower(self, one_shower):
        # TODO : Use dataloader.return_single_shower
        one_sh_dataset = AntennaDataset(
            self.input_data_file,
            self.input_meta_file,
            self.antenna_pos_file,
            self.output_meta_file,
            self.output_file,
            mmap_mode="r",
            one_shower=one_shower,
        )
        dataloader = DataLoader(
            one_sh_dataset,
            batch_size=len(one_sh_dataset),
            shuffle=False,
            num_workers=4,
            collate_fn=custom_collate_fn,
        )
        assert len(dataloader) == 1
        for batch in dataloader:
            if batch is None:
                raise RuntimeError("Not a valid Shower {one_shower}")

            event_data, meta_data, antenna_pos, output_meta, output = batch
            # TODO: Fix it in the input file and stop swapaxes.
            event_data = torch.swapaxes(event_data, 1, 2)
            event_data, meta_data, antenna_pos = (
                event_data.to(self.device),
                meta_data.to(self.device),
                antenna_pos.to(self.device),
            )
            with torch.no_grad():
                pred_output_meta, pred_output = self.model(
                    event_data, meta_data, antenna_pos
                )

            return pred_output_meta.cpu().numpy(), pred_output.cpu().numpy()

    def pred_one_shower_entire_array(self, one_shower):
        # TODO : Use dataloader.return_single_shower
        one_sh_dataset = AntennaDataset(
            self.input_data_file,
            self.input_meta_file,
            self.antenna_pos_file,
            self.output_meta_file,
            self.output_file,
            mmap_mode="r",
            one_shower=one_shower,
        )
        dataloader = DataLoader(
            one_sh_dataset,
            batch_size=len(one_sh_dataset),
            shuffle=False,
            num_workers=4,
            collate_fn=custom_collate_fn,
        )
        assert len(dataloader) == 1
        for batch in dataloader:
            if batch is None:
                raise RuntimeError("Not a valid Shower {one_shower}")

            event_data, meta_data, antenna_pos, output_meta, output = batch
            # TODO: Fix it in the input file and stop swapaxes.
            antenna_pos = torch.Tensor(
                fit_plane_and_return_3d_grid(antenna_pos.cpu().numpy())
            )
            event_data = torch.swapaxes(event_data, 1, 2)
            assert torch.all(event_data == event_data[0])
            assert torch.all(meta_data == meta_data[0])
            event_data, meta_data, antenna_pos = (
                event_data.to(self.device),
                meta_data.to(self.device),
                antenna_pos.to(self.device),
            )
            print(event_data.shape)
            print(
                event_data[0]
                .expand(antenna_pos.shape[0], *event_data[0].shape)
                .shape
            )
            with torch.no_grad():
                pred_output_meta, pred_output = self.model(
                    event_data[0].expand(
                        antenna_pos.shape[0], *event_data[0].shape
                    ),
                    meta_data[0].expand(
                        antenna_pos.shape[0], *meta_data[0].shape
                    ),
                    antenna_pos,
                )

            return pred_output_meta.cpu().numpy(), pred_output.cpu().numpy()
