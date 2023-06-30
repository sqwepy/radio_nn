"""
Dataloader classes.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


def custom_collate_fn(batch):
    """
    Filter the bad data

    Used internally by the dataloader.
    Parameters
    ----------
    batch: Input patch

    Returns
    -------
    collated batch as an array.
    """
    filtered_batch = []
    for event_data, meta_data, antenna_pos, output_meta, output in batch:
        is_finite = (
            torch.isfinite(event_data).all()
            and torch.isfinite(meta_data).all()
            and torch.isfinite(antenna_pos).all()
            and torch.isfinite(output_meta).all()
            and torch.isfinite(output).all()
        )

        has_non_zero_meta_data = (meta_data[0] != 0) and (meta_data[2] != 0)

        if is_finite and has_non_zero_meta_data:
            filtered_batch.append(
                (event_data, meta_data, antenna_pos, output_meta, output)
            )

    if not filtered_batch:
        return None

    return default_collate(filtered_batch)


class AntennaDataset(Dataset):
    """
    Class to load the antenna dataset.
    """

    def __init__(
        self,
        input_data_file,
        input_meta_file,
        antenna_pos_file,
        output_meta_file,
        output_file,
        mmap_mode=None,
        percentage=100,
        one_shower=None,
    ):
        """
        Initialize the antenna dataset as memmap arrays.

        Parameters
        ----------
        input_data_file
        input_meta_file
        antenna_pos_file
        output_meta_file
        output_file
        mmap_mode
        percentage
        one_shower
        """
        # TODO: Make this into a seperate class
        self.input_data = np.load(input_data_file, mmap_mode=mmap_mode)
        self.input_meta = np.load(input_meta_file, mmap_mode=mmap_mode)
        self.antenna_pos = np.load(antenna_pos_file, mmap_mode=mmap_mode)
        self.output_meta = np.load(output_meta_file, mmap_mode=mmap_mode)
        self.output = np.load(output_file, mmap_mode=mmap_mode)
        self.percentage = percentage
        self.one_shower = one_shower
        if self.one_shower is not None:
            self.total_events = 1 * self.antenna_pos.shape[1]
        else:
            self.total_events = (
                self.input_data.shape[0] * self.antenna_pos.shape[1]
            )
            num_samples = int(self.total_events * self.percentage / 100)
            self.indices = np.sort(
                np.random.choice(
                    np.arange(self.total_events),
                    size=num_samples,
                    replace=False,
                )
            )

    def __len__(self):
        if self.one_shower is not None:
            return self.total_events
        return len(self.indices)

    def __getitem__(self, idx):
        if self.one_shower is not None:
            event_idx = self.one_shower
            antenna_idx = idx
        else:
            selected_idx = self.indices[idx]
            event_idx = selected_idx // self.antenna_pos.shape[1]
            antenna_idx = selected_idx % self.antenna_pos.shape[1]

        # TODO: Check how much data we need to give here
        event_data = (
            torch.log(
                torch.tensor(
                    self.input_data[event_idx, :, 4:], dtype=torch.float32
                )
                + 1e-14
            )
            / 30
        )
        meta_data = torch.tensor(
            self.input_meta[event_idx], dtype=torch.float32
        )[1:]
        meta_data[2] = meta_data[2] / 100
        meta_data[3] = torch.log(meta_data[3])
        meta_data[4] = torch.log(meta_data[4])
        meta_data[5] = torch.log(meta_data[5])
        meta_data[10] = meta_data[10] / 5000
        meta_data[11] = torch.log(meta_data[11])

        antenna_pos = torch.tensor(
            self.antenna_pos[event_idx, antenna_idx], dtype=torch.float32
        )
        output_meta = torch.tensor(
            self.output_meta[event_idx, antenna_idx], dtype=torch.float32
        )
        output_meta = torch.sign(output_meta) * torch.log(
            torch.abs(output_meta) + 1e-14
        )
        output = torch.tensor(
            self.output[event_idx, antenna_idx], dtype=torch.float32
        )

        return event_data, meta_data / 20, antenna_pos, output_meta, output

    def data_of_single_shower(self, one_shower):
        # TODO: Seperate the preprocessing into seperate function
        one_shower_event_idx = one_shower
        inp_d = np.log(self.input_data[one_shower_event_idx, :, 4:] + 1e-14)
        outp_m = self.output_meta[one_shower_event_idx, :]
        inp_m = np.copy(self.input_meta[one_shower_event_idx])
        inp_m[2] = inp_m[2] / 100
        inp_m[3] = np.log(inp_m[3])
        inp_m[4] = np.log(inp_m[4])
        inp_m[5] = np.log(inp_m[5])
        inp_m[10] = inp_m[10] / 5000
        inp_m[11] = np.log(inp_m[11])
        return (
            inp_d / 30,
            inp_m[1:] / 20,
            self.antenna_pos[one_shower_event_idx, :] / 250,
            np.sign(outp_m) * np.log(np.abs(outp_m) + 1e-14),
            self.output[one_shower_event_idx, :],
        )
