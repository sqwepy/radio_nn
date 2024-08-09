"""Filters to be used by the dataloader class."""

import warnings

import numpy as np
from tqdm import tqdm

warnings.simplefilter("ignore", np.RankWarning)


def get_thinning_factor(pulses, pos_array, outp_meta, antenna_mask, index=0):
    ff = np.fft.rfftfreq(256, 1e-9) * 1e-6  # convert to MHz
    mask = (ff > 30) & (ff < 80)
    frequency_slope = np.zeros((len(pulses), pulses.shape[-1], 2))
    ans = np.zeros(len(pulses))
    for i in np.arange(pulses.shape[0]):
        if not antenna_mask[i]:
            ans[i] = np.inf
        for iPol in range(pulses.shape[-1]):
            filtered_spec = np.fft.rfft(pulses[i, :, iPol])
            # Fit slope
            mask2 = filtered_spec[mask] > 0
            if np.sum(mask2):
                xx = ff[mask][mask2]
                yy = np.log10(np.abs(filtered_spec[mask][mask2]) + 1e-14)
                z = np.polyfit(xx, yy, 1)
                frequency_slope[i][iPol] = z
        freq_slope = frequency_slope[i, index, 0]
        ans[i] = freq_slope
    return ans


def thin_or_not(pulses, pos_array, outp_meta, antenna_mask, index=0):
    ans = get_thinning_factor(pulses, pos_array, outp_meta, antenna_mask, index=0)
    min_index = np.argmin(ans)
    lateral_distance = np.sqrt(pos_array[:, 0] ** 2 + pos_array[:, 1] ** 2)
    fin_ans = np.where(
        lateral_distance < 0.85 * lateral_distance[min_index], True, False
    )
    return fin_ans


def skip_vb_axis(ant_pos: np.ndarray) -> np.ndarray:
    """
    Mask for skipping the vb axis.

    Parameters
    ----------
    ant_pos : np.ndarray - Positions of antennas

    Returns
    -------
    np.ndarray - Bool - Mask
    """
    tan_value_close_to_xaxis = 0.5
    x_pos = np.abs(ant_pos[:, 0])
    y_pos = np.abs(ant_pos[:, 1])
    return (np.abs(np.arctan2(y_pos, x_pos)) > tan_value_close_to_xaxis).flatten()


def only_positive_vvb_axis(ant_pos: np.ndarray) -> np.ndarray:
    """
    Mask for keeping only the vvb axis.

    Parameters
    ----------
    ant_pos : np.ndarray - Positions of antennas

    Returns
    -------
    np.ndarray - Bool - Mask
    """
    x_pos = np.abs(ant_pos[:, 0])
    xvalue_close_to_yaxis = 10
    mask = np.abs(x_pos) < xvalue_close_to_yaxis
    return mask

def peak_cutoff(pulses: np.ndarray) -> np.ndarray:
    """
    Mask for keeping only high amplitude pulses.

    Parameters
    ----------
    pulses : np.ndarray - Positions of antennas

    Returns
    -------
    np.ndarray - Bool - Mask
    """
    cutoff_value = 1e-5
    peak = np.max(pulses[: , : , 0], axis=1)
    return peak > cutoff_value



def all_antennas(ant_pos: np.ndarray) -> np.ndarray:
    """
    Mask for keeping all antennas.

    Parameters
    ----------
    ant_pos : np.ndarray - Positions of antennas

    Returns
    -------
    np.ndarray - Bool - Mask
    """
    x_pos = np.abs(ant_pos[:, 0])
    mask = np.abs(x_pos) > 0
    return mask


class DefaultFilter:
    def __init__(
        self,
        input_data,
        input_meta,
        antenna_pos,
        output_meta,
        output,
        percentage,
    ) -> None:
        self.input_data = input_data
        self.input_meta = input_meta
        self.antenna_pos = antenna_pos
        self.output_meta = output_meta
        self.output = output
        self.percentage = percentage

    def _get_antenna_mask(self, index):
        antenna_mask = all_antennas(self.antenna_pos[index])
        #antenna_mask &= thin_or_not(
        #    self.output[index],
        #    self.antenna_pos[index],
        #    self.output_meta[index],
        #    antenna_mask,
        #)
        antenna_mask &= peak_cutoff(self.output[index])
        # return np.tile(np.arange(240), self.shower_indices.shape[0])
        return antenna_mask

    def _get_shower_mask(self):
        shower_mask = np.ones(shape=self.input_data.shape[0], dtype=np.bool_)
        shower_mask &= self.input_meta[:, 1] > 0.5  # Throw away bad showers
        shower_mask &= self.input_meta[:, 2] > 300  #  Throw away bad showers
        # shower_mask &= self.input_meta[:, 2] < 900  #  xmax < 900
        print("Total showers after filter", np.sum(shower_mask))
        return shower_mask

    def _get_shower_indices(self):
        shower_mask = self._get_shower_mask()
        num_samples = int(np.sum(shower_mask) * self.percentage / 100)
        return np.random.choice(
            np.arange(self.input_data.shape[0], dtype=int)[shower_mask],
            size=num_samples,
            replace=False,
        )

    def get_indices(self):
        shower_indices = self._get_shower_indices()
        indices = np.array([], dtype=int)
        for index in tqdm(shower_indices):
            antenna_mask = self._get_antenna_mask(index)
            antenna_indices = np.arange(240, dtype=int)[antenna_mask]
            overall_index = int(index * 240) + antenna_indices
            indices = np.concatenate((indices, overall_index))
        return np.sort(indices)
