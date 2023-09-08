"""
Filters to be used by the dataloader class
"""
import torch
import numpy


def default_filter(event_data, meta_data, antenna_pos, output_meta, output):
    xmax_range = 600 < meta_data[2] < 800
    return True  # xmax_range
