"""Dataloader classes."""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from radioNN.data.filters import DefaultFilter
from radioNN.data.transforms import DefaultTransform, Identity


def custom_collate_fn(batch):
    """
    Filter the bad data.

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
    """Class to load the antenna dataset."""

    def __init__(
        self,
        input_data_file,
        input_meta_file,
        antenna_pos_file,
        output_meta_file,
        output_file,
        fluence_file=None,
        mmap_mode=None,
        percentage=100,
        one_shower=None,
        transform=DefaultTransform,
        filter=DefaultFilter,
        device="cpu",
    ) -> None:
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
        self.device = device

        if transform is not None:
            self.transform = transform()
        else:
            self.transform = Identity()

        self.filter = filter(
            self.input_data,
            self.input_meta,
            self.antenna_pos,
            self.output_meta,
            self.output,
            self.percentage,
        )

        if self.one_shower is not None:
            self.total_events = 1 * self.antenna_pos.shape[1]
        else:
            self.total_events = self.input_data.shape[0] * self.antenna_pos.shape[1]
            self.indices = self.filter.get_indices()

    def __len__(self) -> int:
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
        (
            event_data,
            meta_data,
            antenna_pos,
            output_meta,
            output,
        ) = self.transform(
            torch.tensor(self.input_data[event_idx], dtype=torch.float32).to(
                self.device
            ),
            torch.tensor(self.input_meta[event_idx], dtype=torch.float32).to(
                self.device
            ),
            torch.tensor(
                self.antenna_pos[event_idx, antenna_idx], dtype=torch.float32
            ).to(self.device),
            torch.tensor(
                self.output_meta[event_idx, antenna_idx], dtype=torch.float32
            ).to(self.device),
            torch.tensor(self.output[event_idx, antenna_idx], dtype=torch.float32).to(
                self.device
            ),
        )
        return event_data, meta_data, antenna_pos, output_meta, output

    def return_hfit_of_shower(self : "AntennaDataset", one_shower : int) -> np.poly1d:
        """
        Return fit for xmax to height conversion.

        Parameters
        ----------
        one_shower: Int - Shower index

        Returns
        -------
        np.poly1d: Class which has the fit for the conversion
        """
        mask_indices = (np.array([str(i)[:-2] for i in np.int64(self.input_meta[:, 0])],
                                 dtype=str) == str(
            np.int64(self.input_meta[one_shower, 0]))[:-2])
        x = self.input_meta[mask_indices, 2] / 700
        y = self.input_meta[mask_indices, 4] / 7e5
        # plt.scatter(x,y)
        fitparam_1, res_1, rank_1, singval_1, rcond_1 = np.polyfit(x, y, 2,
                                                                   full=True)
        pfit_1 = np.poly1d(fitparam_1)
        return pfit_1

    def data_of_single_shower(self, one_shower):
        one_shower_event_idx = one_shower
        inp_d = self.input_data[one_shower_event_idx]
        inp_m = np.copy(self.input_meta[one_shower_event_idx])
        ant_pos = np.copy(self.antenna_pos[one_shower_event_idx, :])
        outp_m = self.output_meta[one_shower_event_idx]
        outp_d = self.output[one_shower_event_idx]
        (
            event_data,
            meta_data,
            antenna_pos,
            output_meta,
            output,
        ) = self.transform(inp_d, inp_m, ant_pos, outp_m, outp_d)
        return event_data, meta_data, antenna_pos, output_meta, output

    def return_data(self, percentage=None):
        if percentage is not None:
            # TODO: This segment is botched
            num_samples = int(self.total_events * percentage / 100)
            indices = np.sort(
                np.random.choice(
                    np.arange(self.total_events),
                    size=num_samples,
                    replace=False,
                )
            )
        else:
            indices = self.indices

        selected_idx = indices
        event_idx = selected_idx // self.antenna_pos.shape[1]
        antenna_idx = selected_idx % self.antenna_pos.shape[1]
        (
            event_data,
            meta_data,
            antenna_pos,
            output_meta,
            output,
        ) = self.transform(
            torch.tensor(self.input_data[event_idx], dtype=torch.float32),
            torch.tensor(np.copy(self.input_meta[event_idx]), dtype=torch.float32),
            torch.tensor(self.antenna_pos[event_idx, antenna_idx], dtype=torch.float32),
            torch.tensor(self.output_meta[event_idx, antenna_idx], dtype=torch.float32),
            torch.tensor(self.output[event_idx, antenna_idx], dtype=torch.float32),
        )
        return event_data, meta_data, antenna_pos, output_meta, output
