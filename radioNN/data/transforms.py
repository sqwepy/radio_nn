"""
Tranforms to be used by the dataloader class
"""
import numpy as np
import torch
import numpy


class Identity:
    """
    Identity transformation
    """

    def __call__(self, event_data, meta_data, antenna_pos, output_meta, output):
        return event_data, meta_data, antenna_pos, output_meta, output


class DefaultTransform:
    def __call__(self, event_data, meta_data, antenna_pos, output_meta, output):

        if isinstance(event_data, numpy.ndarray):
            module = numpy
        elif isinstance(event_data, torch.Tensor):
            module = torch

        # The transpose is so that this transform is generalized for various
        # input shapes

        event_data = module.log(event_data.T[4:] + 1e-14).T
        #  2: xmax
        meta_data.T[2] /= 700
        #  3: density at xmax - Useless, barely varies in value
        #  4: Height(xmax) - from 1% data, we see div is more sensible than log
        meta_data.T[4] /= 700 * 1e3
        #  5: Eem
        meta_data.T[5] = module.log(meta_data.T[5] + 1e-14) / 20
        # 10: PRMPAR
        # No nothing
        # TODO: (Think about what to do for the 5000 value outlier)
        # 11: Primary Energy
        meta_data.T[11] = module.log(meta_data.T[11] + 1e-14)

        meta_data = meta_data.T[1:].T
        output_meta = module.sign(output_meta) * module.log(
            module.abs(output_meta) + 1e-14
        )
        # Event shape is flipped. It should be [batch, 300, 7] but it is
        # [batch, 7, 300].
        # TODO: Fix it in the input file and stop swapaxes.
        return (
            event_data.T / 30,
            meta_data,
            antenna_pos / 250,
            output_meta,
            output,
        )
