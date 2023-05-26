import os

import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from radioNN.main_network import AntennaNetwork
from radioNN.dataloader import AntennaDataset, custom_collate_fn


def train(model, dataloader, criterion, optimizer, device, loss_obj=False):
    model.train()
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

        pred_output_meta, pred_output = model(event_data, meta_data, antenna_pos)

        loss_meta = criterion(pred_output_meta, output_meta)
        loss_output = criterion(pred_output, output)
        loss = loss_meta + loss_output

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
    RADIO_DATA_PATH = "/home/sampathkumar/radio_data"
    memmap_mode = "r"
    if not os.path.exists(RADIO_DATA_PATH):
        RADIO_DATA_PATH = "/home/pranav/work-stuff-unsynced/radio_data"
        memmap_mode = "r"
    assert os.path.exists(RADIO_DATA_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using the data from {RADIO_DATA_PATH} in {device} with memmap "
        f"mode: {memmap_mode} using {percentage}% of data"
    )
    input_data_file = os.path.join(RADIO_DATA_PATH, "input_data.npy")
    input_meta_file = os.path.join(RADIO_DATA_PATH, "meta_data.npy")
    antenna_pos_file = os.path.join(RADIO_DATA_PATH, "antenna_pos_data.npy")
    output_meta_file = os.path.join(RADIO_DATA_PATH, "output_meta_data.npy")
    output_file = os.path.join(RADIO_DATA_PATH, "output_gece_data.npy")
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
    model = AntennaNetwork(output_channels).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return criterion, dataloader, device, model, optimizer
