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

    def get_antenna_indices(self):
        return np.arange(240)

    def get_shower_indices(self):
        num_samples = int(self.input_data.shape[0] * self.percentage / 100)
        return np.random.choice(
            np.arange(self.input_data.shape[0]),
            size=num_samples,
            replace=False,
        )

    def get_indices(self):
        shower_indices = self.get_shower_indices()
        antenna_indices = self.get_antenna_indices()
        indices = np.repeat(shower_indices, len(antenna_indices)) * len(
            antenna_indices
        ) + np.tile(antenna_indices, shower_indices.shape[0])
        return np.sort(indices)
