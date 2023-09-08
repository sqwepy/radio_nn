"""
Tranforms to be used by the dataloader class
"""
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
        event_data = module.log(event_data[:, 4:] + 1e-14)
        meta_data[2] = meta_data[2] / 100
        meta_data[3] = module.log(meta_data[3])
        meta_data[4] = meta_data[4] / 30000  # from 1% data, we see this is
        # more sensible than log
        meta_data[5] = module.log(meta_data[5])
        meta_data[10] = meta_data[10] / 5000
        meta_data[11] = module.log(meta_data[11])

        meta_data = meta_data[1:]
        output_meta = module.sign(output_meta) * module.log(
            module.abs(output_meta) + 1e-14
        )
        return (
            event_data / 30,
            meta_data / 20,
            antenna_pos / 250,
            output_meta,
            output,
        )
