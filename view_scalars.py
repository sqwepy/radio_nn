import numpy as np
from matplotlib import pyplot as plt
from radiotools.analyses import energy_fluence


def get_fluences(pulses, pos_array, outp_meta):
    def process_trace(trace, outp_meta):
        result = np.zeros((256, 4))
        result[:, 0] = outp_meta[0] + np.arange(256) * 1e-9
        result[:, 1 : trace.shape[1] + 1] = trace
        return result

    energy_fluences = np.zeros((len(pulses), 3))
    for index in range(len(pulses)):
        trace_vB = pulses[index]  # 0,1,2,3: t, vxB, vxvxB, v
        trace_vB = process_trace(trace_vB, outp_meta[0])
        ef = energy_fluence.calculate_energy_fluence_vector(
            trace_vB[:, 1:], trace_vB[:, 0], remove_noise=True
        )
        energy_fluences[index] = ef
    x_pos, y_pos = pos_array[:, 0], pos_array[:, 1]
    if len(energy_fluences.shape) == 1:
        energy_fluences = np.array([energy_fluences]).T
    return energy_fluences


def get_fluences0(pulses, pos_array, outp_meta):
    return get_fluences(pulses, pos_array, outp_meta)[:, 0]


def get_fluences1(pulses, pos_array, outp_meta):
    return get_fluences(pulses, pos_array, outp_meta)[:, 1]


def get_polarity(pulses, pos_array, outp_meta):
    pulse_max = np.max(pulses, axis=1)
    pulse_min = np.min(pulses, axis=1)
    polarity = np.where(np.abs(pulse_max) > np.abs(pulse_min), 1, -1)
    return polarity


def get_polarity0(pulses, pos_array, outp_meta):
    return get_polarity(pulses, pos_array, outp_meta)[:, 0]


def get_polarity1(pulses, pos_array, outp_meta):
    return get_polarity(pulses, pos_array, outp_meta)[:, 1]


def get_n2polarity0(pulses, pos_array, outp_meta):
    return np.copy(pulses[:, 128, 0])


def get_n2polarity1(pulses, pos_array, outp_meta):
    return np.copy(pulses[:, 128, 1])


def get_timing0(pulses, pos_array, outp_meta):
    return outp_meta[:, 0]

def get_timing1(pulses, pos_array, outp_meta):
    return outp_meta[:, 1]

def get_npolarity(pulses, pos_array, outp_meta):
    pulse_max = np.argmax(pulses, axis=1)
    pulse_min = np.argmin(pulses, axis=1)
    polarity = np.where(np.abs(pulse_max) < np.abs(pulse_min), 1, -1)
    return polarity


def get_npolarity0(pulses, pos_array, outp_meta):
    return get_npolarity(pulses, pos_array, outp_meta)[:, 0]


def get_npolarity1(pulses, pos_array, outp_meta):
    return get_npolarity(pulses, pos_array, outp_meta)[:, 1]


def get_max_value0(pulses, pos_array, outp_meta):
    return np.max(np.abs(pulses), axis=1)[:, 0]


def get_max_value1(pulses, pos_array, outp_meta):
    return np.max(np.abs(pulses), axis=1)[:, 1]


def get_smax_value0(pulses, pos_array, outp_meta):
    pol = 0
    index = np.argmax(np.abs(pulses[:, :, pol]), axis=1)
    print(index.shape)
    smax_ans = np.zeros(pulses.shape[0])
    for i, j in enumerate(index):
        smax = pulses[i, j, pol]
        smax_ans[i] = np.sign(smax)
    return smax_ans


def get_smax_value1(pulses, pos_array, outp_meta):
    pol = 1
    index = np.argmax(np.abs(pulses[:, :, pol]), axis=1)
    print(index.shape)
    smax_ans = np.zeros(pulses.shape[0])
    for i, j in enumerate(index):
        smax = pulses[i, j, pol]
        smax_ans[i] = np.sign(smax)
    return smax_ans


def get_thinning_factor(pulses, pos_array, outp_meta, index=0):
    ff = np.fft.rfftfreq(256, 1e-9) * 1e-6  # convert to MHz
    mask = (ff > 30) & (ff < 80)
    frequency_slope = np.zeros((len(pulses), pulses.shape[-1], 2))
    ans = np.zeros(len(pulses))
    for i in range(len(pulses)):
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


def thin_or_not(pulses, pos_array, outp_meta, index=0):
    ans = get_thinning_factor(pulses, pos_array, outp_meta, index=0)
    min_index = np.argmin(ans)
    lateral_distance = np.sqrt(pos_array[:, 0] ** 2 + pos_array[:, 1] ** 2)
    fin_ans = np.where(lateral_distance < lateral_distance[min_index], 1, -1)
    return fin_ans


def get_lateral_distance(pulses, pos_array, outp_meta):
    return np.sqrt(pos_array[:, 0] ** 2 + pos_array[:, 1] ** 2)


def get_cross_correlation0(pulses, pos_array, outp_meta, pulses2=None):
    cross = np.sum(pulses[:, :, 0] * pulses2[:, :, 0], axis=1)
    absone = np.sqrt(np.sum(pulses[:, :, 0] * pulses[:, :, 0], axis=1))
    abstwo = np.sqrt(np.sum(pulses2[:, :, 0] * pulses2[:, :, 0], axis=1))
    return cross / (absone * abstwo)


def get_cross_correlation1(pulses, pos_array, outp_meta, pulses2=None):
    cross = np.sum(pulses[:, :, 1] * pulses2[:, :, 1], axis=1)
    absone = np.sqrt(np.sum(pulses[:, :, 1] * pulses[:, :, 1], axis=1))
    abstwo = np.sqrt(np.sum(pulses2[:, :, 1] * pulses2[:, :, 1], axis=1))
    return cross / (absone * abstwo)


def get_mse0(pulses, pos_array, outp_meta, pulses2=None):
    return np.sum((pulses[:, :, 0] - pulses2[:, :, 0]) ** 2, axis=1)


def get_logmax_value(pulses, pos_array, outp_meta):
    return np.log10(np.max(np.abs(pulses), axis=1)[:, 0] + 1e-14)


def get_peak(trace_x, trace_y):
    Etotal = np.sqrt(trace_x**2 + trace_y**2)
    t_0_indices = np.argmax(Etotal)  # the bin of t_0
    return t_0_indices


def get_peak_hilbert(trace_x):
    from scipy.signal import hilbert

    hilbert_signal = hilbert(trace_x**2)
    t_0_indices = np.argmax(np.imag(hilbert_signal))  # the bin of t_0
    return t_0_indices


def center_pulses(pulses, peak_finder=get_peak):
    # calculate Etotal
    new_pulses = np.zeros_like(pulses)
    print(pulses.shape)
    for j in range(pulses.shape[-1]):
        for i, pulse in enumerate(pulses):
            trace = pulse[:, j]
            t_0_indices = peak_finder(trace)
            # calculate shift in bins:
            shift = int(t_0_indices - 128)
            final_length = 256
            trace_new = np.zeros(final_length)
            # Create a new array with zeros and appropriate padding
            # for negative shift
            if shift > 0:
                # for positiive shift
                trace_new[0:-shift] = trace[shift:]
                # trace_z_new[0: -shift] = trace_z[shift:]

            elif shift < 0:
                # for negative shift
                shift = np.abs(shift)
                trace_new[shift:] = trace[0:-shift]
                # trace_z_new[shift:] = trace_z[0:-shift]
            else:
                trace_new = trace
            new_pulses[i, :, j] = trace_new
        # pulse[:, 2] = trace_x_new
    return new_pulses
