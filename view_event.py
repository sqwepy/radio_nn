from RadioPlotter.viewevent import view_footprint
import os
import torch
import numpy as np
import argparse

from RadioPlotter.viewplots import view_plots
from radioNN.networks.antenna_fc_network import AntennaNetworkFC

# from RadioPlotter.viewplots import view_plots
from radioNN.data.loader import AntennaDataset

from radioNN.networks.antenna_skipfc_network import AntennaNetworkSkipFC
from radioNN.data.transforms import sph2cart
import matplotlib.pyplot as plt

from view_scalars import (
    center_pulses,
    get_cross_correlation0,
    get_cross_correlation1,
    get_fluences0,
    get_fluences1,
    get_lateral_distance,
    get_logmax_value,
    get_max_value0,
    get_max_value1,
    get_mse0,
    get_n2polarity0,
    get_n2polarity1,
    get_npolarity0,
    get_npolarity1,
    get_peak,
    get_peak_hilbert,
    get_polarity0,
    get_polarity1,
    get_smax_value0,
    get_smax_value1,
    get_thinning_factor,
    thin_or_not,
)

fnt_size = 20
plt.rc("text", usetex=True)

from radioNN.process_network import NetworkProcess

radio_data_path = "/home/sampathkumar/radio_data"
memmap_mode = "r"
if not os.path.exists(radio_data_path):
    radio_data_path = "/home/pranav/work-stuff-unsynced/radio_data"
    memmap_mode = "r"
assert os.path.exists(radio_data_path)
input_data_file = os.path.join(radio_data_path, "input_data.npy")
input_meta_file = os.path.join(radio_data_path, "meta_data.npy")
antenna_pos_file = os.path.join(radio_data_path, "antenna_pos_data.npy")
output_meta_file = os.path.join(radio_data_path, "output_meta_data.npy")
output_file = os.path.join(radio_data_path, "output_gece_data.npy")
dataset = AntennaDataset(
    input_data_file,
    input_meta_file,
    antenna_pos_file,
    output_meta_file,
    output_file,
    mmap_mode=memmap_mode,
    one_shower=33,
    # percentage=0.01,
    # return_fluence=False,
)

output_file = os.path.join(radio_data_path, "output_vBvvB_data.npy")
dataset2 = AntennaDataset(
    input_data_file,
    input_meta_file,
    antenna_pos_file,
    output_meta_file,
    output_file,
    mmap_mode=memmap_mode,
    one_shower=33,
    # return_fluence=False,
    # percentage=0.01,
)


def make_table(input_meta):
    """
    Meta Data format:
    index 0: Sim number
    index 1: Cos(Zenith Angle)
    index 2: X_max
    index 3: density at X_max
    index 4: height at X_max
    index 5: E_em
    index 6: sin(geomagnetic_angle)
    index 7: B inclination
    index 8: B declination
    index 9: B strength
    index 10: primary particle
    index 11: primary energy
    index 12: Azimuthal angle
    """
    return {
        "Sim number": input_meta[0],
        "Cos(Zenith Angle)": input_meta[1],
        "X_max": input_meta[2],
        "density at X_max": input_meta[3],
        "height at X_max": input_meta[4],
        "E_em": input_meta[5],
        "sin(geomagnetic_angle)": input_meta[6],
        "B inclination": input_meta[7],
        "B declination": input_meta[8],
        "B strength": input_meta[9],
        "primary particle": input_meta[10],
        "primary energy": input_meta[11],
        "Azimuthal angle": input_meta[12],
    }


# choice = 26387
# choice = 14345
print("tot", dataset.total_events)
choice = np.random.choice(np.unique(np.arange(26387)), size=2)[0]
print(choice)
data = dataset.data_of_single_shower(choice)
pos, meta, real = data[2], data[3], data[4]
sph2cart(pos)
data = dataset2.data_of_single_shower(choice)
pos1, meta1, real1 = data[2], data[3], data[4]
sph2cart(pos1)
process = NetworkProcess(
    model_class=AntennaNetworkFC,
    # one_shower=one_shower,
    percentage=0.1,
    batch_size=8,
    wb=False,
)
#model_name = "2309Sep26Tue_155429"
#model_name = "2405May14Tue_172854"
#model_name = "2405May18Sat_030851"
model_name = "2405May24Fri_145354"
#model_name = "2406Jun10Mon_144306"
#model_name = "2406Jun20Thu_200714"

process.output_channels = 3


#model_name = "2407Jul29Mon_213725"
process.model = AntennaNetworkFC(process.output_channels)
state_checkpoint = torch.load(
    f"/home/pranav/MEGA/work-stuff/radio_nn/runs/{model_name}/SavedState",
)
process.model.load_state_dict(state_checkpoint['model_state_dict'])
process.model.eval()
sim = process.pred_one_shower(choice)[1]
sim = sim.numpy()


def spectrum(pulse):
    ff = np.fft.rfftfreq(256, 1e-9) * 1e-6  # convert to MHz
    mask = (ff > 30) & (ff < 80)
    spec = np.log10(np.abs(np.fft.rfft(pulse)))[mask]
    return spec


print(make_table(data[1]))
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p" "--plotter",
    action="store_true",
    help="Use interactive Plotter instead of event viewer",
)
opt = parser.parse_args()
if opt.p__plotter:
    mask = np.abs(pos[:, 0]) < 1000
    pos = pos[mask]
    pos1 = pos1[mask]
    real1 = real1[mask]
    sim = sim[mask]
    meta1 = meta1[mask]
    meta = meta[mask]
    view_plots(
        {
            "vbvvb": {
                "pos": pos1,
                "real": real1,
                "meta": meta1,
            },
            "gece": {
                "pos": pos,
                "real": real,
                "meta": meta,
            },
            "vbvvb_sim": {
                "pos": pos,
                "real": sim,
                "meta": meta,
            },
        },
        {
            "vB Fluence vs Lateral Distance": {
                "xfunc": get_lateral_distance,
                "yfunc": get_fluences0,
                "xscale": "linear",
                "yscale": "linear",
            },
            "vvB Fluence vs Lateral Distance": {
                "xfunc": get_lateral_distance,
                "yfunc": get_fluences1,
                "xscale": "linear",
                "yscale": "linear",
            },
            "Coorel0 w sim vs Distance": {
                "xfunc": get_lateral_distance,
                "yfunc": lambda x, y, z: get_cross_correlation0(x, y, z, pulses2=sim),
                "xscale": "linear",
                "yscale": "linear",
            },
            "Coorel1 w sim vs Distance": {
                "xfunc": get_lateral_distance,
                "yfunc": lambda x, y, z: get_cross_correlation1(x, y, z, pulses2=sim),
                "xscale": "linear",
                "yscale": "linear",
            },
            "Coorel0 w sim vs Fluence": {
                "xfunc": get_fluences0,
                "yfunc": lambda x, y, z: get_cross_correlation0(x, y, z, pulses2=sim),
                "xscale": "log",
                "yscale": "linear",
            },
            "Coorel1 w sim vs Fluence": {
                "xfunc": get_fluences1,
                "yfunc": lambda x, y, z: get_cross_correlation1(x, y, z, pulses2=sim),
                "xscale": "log",
                "yscale": "linear",
            },
            "Rel vb Fluence vs Lateral Distance": {
                "xfunc": get_lateral_distance,
                "yfunc": lambda x, y, z: (
                    (get_fluences0(x, y, z) - get_fluences0(sim, y, z))
                    / get_fluences0(x, y, z)
                ),
                "xscale": "linear",
                "yscale": "linear",
            },
            "Rel vvb Fluence vs Lateral Distance": {
                "xfunc": get_lateral_distance,
                "yfunc": lambda x, y, z: (
                    (get_fluences1(x, y, z) - get_fluences1(sim, y, z))
                    / get_fluences1(x, y, z)
                ),
                "xscale": "linear",
                "yscale": "linear",
            },
            "Rel vvb Fluence vs fluence": {
                "xfunc": get_fluences0,
                "yfunc": lambda x, y, z: (
                    (get_fluences0(x, y, z) - get_fluences0(sim, y, z))
                    / get_fluences0(x, y, z)
                ),
                "xscale": "log",
                "yscale": "linear",
            },
            "Rel vb Fluence vs fluence": {
                "xfunc": get_fluences1,
                "yfunc": lambda x, y, z: (
                    (get_fluences1(x, y, z) - get_fluences1(sim, y, z))
                    / get_fluences1(x, y, z)
                ),
                "xscale": "log",
                "yscale": "linear",
            },
        }
        # pulse_process=spectrum,
    )
# endregion
# thinning_interactive(real, pos, meta)
# thinning_interactive(sim, pos, meta)
else:
    view_footprint(
        {
            "gece": {
                "pos": pos,
                "real": real,
                "meta": meta,
                "hack": True,
            },
            "gece_shifted": {
                "pos": pos,
                "real": center_pulses(real, peak_finder=get_peak_hilbert),
                "meta": meta,
                "hack": True,
            },
            "vbvvb_sim": {
                "pos": pos,
                "real": sim,
                "meta": meta,
                "hack": False,
            },
            "vbvvb": {
                "pos": pos1,
                "real": real1,
                "meta": meta1,
                "hack": False,
            },
        },
        {
            "Energy Fluence 0 ": get_fluences0,
            "Energy Fluence 1": get_fluences1,
            "Polarity 0 ": get_polarity0,
            "Polarity 1": get_polarity1,
            "Coorel": lambda x, y, z: get_cross_correlation0(x, y, z, pulses2=sim),
            "mse": lambda x, y, z: get_mse0(x, y, z, pulses2=sim),
            # "N Polarity 0 ": get_npolarity0,
            # "N Polarity 1": get_npolarity1,
            # "N2 Polarity 0 ": get_n2polarity0,
            # "N2 Polarity 1": get_n2polarity1,
            "SMax Value 0": get_smax_value0,
            "SMax Value 1": get_smax_value1,
            "Log Max Value": get_logmax_value,
            "Unthinned Pulses": thin_or_not,
        },
    )
