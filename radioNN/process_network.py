"""
Process network, contains the setup and training functions.

TODO: Convert it into a class

"""
import os

import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from radioNN.dataloader import AntennaDataset, custom_collate_fn
from radioNN.networks.antenna_cnn_network import AntennaNetworkCNN
from radioNN.networks.antenna_fc_network import AntennaNetworkFC


def train(model, dataloader, criterion, optimizer, device, loss_obj=False):
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

    for batch in tqdm.tqdm(dataloader, leave=False):
        if batch is None:
            tqdm.tqdm.write(f"Skipped batch {batch}")
            continue

        event_data, meta_data, antenna_pos, output_meta, output = batch
        # Event shape is flipped. It should be [batch, 300, 7] but it is
        # [batch, 7, 300].
        # TODO: Fix it in the input file and stop swapaxes.
        event_data = torch.swapaxes(event_data, 1, 2)
        event_data, meta_data, antenna_pos = (
            event_data.to(device),
            meta_data.to(device),
            antenna_pos.to(device),
        )
        output_meta, output = output_meta.to(device), output.to(device)

        optimizer.zero_grad()

        pred_output_meta, pred_output = model(
            event_data, meta_data, antenna_pos
        )

        loss_meta = criterion(pred_output_meta, output_meta)
        loss_output = criterion(pred_output, output)
        loss = loss_output  # + loss_meta

        if loss_obj:
            return loss
        loss.backward()
        optimizer.step()

        if valid_batch_count % 100 == 0 and valid_batch_count != 0:
            tqdm.tqdm.write(f"Batch Loss is {loss.item()}")

        running_loss += loss.item()

        valid_batch_count += 1

    return running_loss / valid_batch_count


def network_process_setup(percentage=100, one_shower=None):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using the data from {radio_data_path} in {device} with memmap "
        f"mode: {memmap_mode} using {percentage}% of data"
    )
    input_data_file = os.path.join(radio_data_path, "input_data.npy")
    input_meta_file = os.path.join(radio_data_path, "meta_data.npy")
    antenna_pos_file = os.path.join(radio_data_path, "antenna_pos_data.npy")
    output_meta_file = os.path.join(radio_data_path, "output_meta_data.npy")
    output_file = os.path.join(radio_data_path, "output_gece_data.npy")
    dataset = AntennaDataset(
        input_data_file,
        input_meta_file,
        antenna_pos_file,
        output_meta_file,
        output_file,
        mmap_mode=memmap_mode,
        percentage=percentage,
        one_shower=one_shower,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn,
    )
    output_channels = dataset.output.shape[-1]
    print(output_channels)
    assert 2 <= output_channels <= 3
    model = AntennaNetworkCNN(output_channels).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    return criterion, dataloader, device, model, optimizer, scheduler
