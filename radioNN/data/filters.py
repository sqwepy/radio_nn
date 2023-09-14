"""
Filters to be used by the dataloader class
"""
import torch
import numpy as np


class DefaultFilter:
    def __init__(
        self,
        input_data,
        input_meta,
        antenna_pos,
        output_meta,
        output,
        percentage,
    ):
        self.input_data = input_data
        self.input_meta = input_meta
        self.antenna_pos = antenna_pos
        self.output_meta = output_meta
        self.output = output
        self.percentage = percentage

    def _get_antenna_indices(self):
        x_pos = np.abs(self.antenna_pos[self.shower_indices, :, 0])
        y_pos = np.abs(self.antenna_pos[self.shower_indices, :, 1])
        self.antenna_mask = (np.abs(np.arctan2(y_pos, x_pos)) > 0.5).flatten()
        # return np.tile(np.arange(240), self.shower_indices.shape[0])
        return np.tile(np.arange(240), self.shower_indices.shape[0])[
            self.antenna_mask
        ]

    def _get_shower_indices(self):
        num_samples = int(self.input_data.shape[0] * self.percentage / 100)
        return np.random.choice(
            np.arange(self.input_data.shape[0]),
            size=num_samples,
            replace=False,
        )

    def get_indices(self):
        self.shower_indices = self._get_shower_indices()
        self.antenna_indices = self._get_antenna_indices()
        print(self.antenna_indices.shape)
        print(self.shower_indices.shape)
        indices = (
            np.repeat(
                self.shower_indices,
                240,
            )
            * 240
        )[self.antenna_mask] + self.antenna_indices
        return np.sort(indices)
