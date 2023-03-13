#! /usr/bin/env python
"""
Load all the data from HDF5 files to Memmap file to be used by Pytorch later.
"""
import h5py
import numpy as np
from numpy.lib.format import open_memmap
import os.path as path
import argparse

RADIO_DATA_PATH = "/home/pranav/work-stuff-unsynced/radio_data"


def density_at_X(f_h5, X):
    X_bins = f_h5["atmosphere"]["Atmosphere"][:-1, 1]
    h = f_h5["atmosphere"]["Atmosphere"][:-1, 0]
    density = f_h5["atmosphere"]["Density"][np.argmin(np.abs(X_bins - X))]
    h_max = h[np.argmin(np.abs(X_bins - X))]
    return density, h_max


def split_sim_number(nn):
    """Split the number to two parts."""
    a = nn // 100
    b = nn % 100
    return a, b


def flush_input_meta(SIM_NUMBER, f_h5, index):
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
    meta_data_file = path.join(RADIO_DATA_PATH, "meta_data.npy")
    meta_data = open_memmap(meta_data_file, mode="w+")
    assert max(meta_data[:, 0]) + 1 == index
    # check and set always
    assert np.all(np.abs(meta_data[index, :]) == 0)
    meta_data[index, 0] = int(SIM_NUMBER)
    meta_data[index, 1] = np.cos(f_h5["highlevel"].attrs["zenith"])
    meta_data[index, 2] = f_h5["highlevel"].attrs["gaisser_hillas_dEdX"][2]
    d_xmax, h_xmax = density_at_X(f_h5["highlevel"].attrs["gaisser_hillas_dEdX"][2])
    meta_data[index, 3] = d_xmax
    meta_data[index, 4] = h_xmax
    meta_data[index, 5] = f_h5["highlevel"].attrs["Eem"]
    meta_data[index, 6] = np.sin(f_h5["highlevel"].attrs["geomagnetic_angle"])
    meta_data[index, 7] = f_h5["highlevel"].attrs["magnetic_field_inclination"]
    meta_data[index, 8] = f_h5["highlevel"].attrs["magnetic_field_declination"]
    meta_data[index, 9] = f_h5["highlevel"].attrs["magnetic_field_strength"]
    meta_data[index, 10] = f_h5["inputs"].attrs["PRMPAR"]
    meta_data[index, 11] = f_h5["highlevel"].attrs["energy"]
    meta_data[index, 12] = np.deg2rad(f_h5["highlevel"].attrs["azimuth"])
    meta_data.flush()


def flush_output_meta(f_h5, index):
    output_meta_file = path.join(RADIO_DATA_PATH, "output_meta_data.npy")
    output_meta = open_memmap(output_meta_file, mode="w+")
    assert np.all(np.abs(output_meta[index, :]) == 0)
    antennas_trace_gece_f_h5 = f_h5[f"/highlevel/traces/ge_ce"]
    antennas_trace_vBvvB_f_h5 = f_h5[f"/highlevel/traces/vB_vvB"]
    label_index = 0
    for label in antennas_trace_vBvvB_f_h5.keys():
        if label.split("_")[0] != "pos":
            continue
        label_index += 1
        output_meta[index, label_index] = np.array(
            np.min(antennas_trace_gece_f_h5[label][:, 0]),
            np.min(antennas_trace_vBvvB_f_h5[label][:, 0]),
        )
        assert 0 < label_index < 240
    output_meta.flush()


def flush_output_vBvvB(f_h5, index):
    output_vBvvB_file = path.join(RADIO_DATA_PATH, "output_vBvvB_data.npy")
    output_vBvvB = open_memmap(output_vBvvB_file, mode="w+")
    assert np.all(np.abs(output_vBvvB[index, :]) == 0)
    antennas_trace_vBvvB_f_h5 = f_h5[f"/highlevel/traces/vB_vvB"]
    label_index = 0
    for label in antennas_trace_vBvvB_f_h5.keys():
        if label.split("_")[0] != "pos":
            continue
        label_index += 1
        output_vBvvB[index, label_index] = antennas_trace_vBvvB_f_h5[label][:, 1:]
        assert 0 < label_index < 240
    output_vBvvB.flush()


def flush_output_gece(f_h5, index):
    output_gece_file = path.join(RADIO_DATA_PATH, "output_gece_data.npy")
    output_gece = open_memmap(output_gece_file, mode="w+")
    assert np.all(np.abs(output_gece[index, :]) == 0)
    antennas_trace_gece_f_h5 = f_h5[f"/highlevel/traces/ge_ce"]
    label_index = 0
    for label in antennas_trace_gece_f_h5.keys():
        if label.split("_")[0] != "pos":
            continue
        label_index += 1
        output_gece[index, label_index] = antennas_trace_gece_f_h5[label][:, 1:-1]
        assert 0 < label_index < 240
    output_gece.flush()


def flush_antenna_pos(f_h5, index):
    antenna_pos_file = path.join(RADIO_DATA_PATH, "antenna_pos_data.npy")
    antenna_pos = open_memmap(antenna_pos_file, mode="w+")
    assert np.all(np.abs(antenna_pos[index, :]) == 0)
    antennas_pos_f_h5 = f_h5[f"/highlevel/positions/ge_ce"]
    label_index = 0
    for label in antennas_pos_f_h5.keys():
        if label.split("_")[0] != "pos":
            continue
        label_index += 1
        antenna_pos[index, label_index] = antennas_pos_f_h5[label]
        assert 0 < label_index < 240
    antenna_pos.flush()


def flush_input_data(f_h5, index):
    input_data_file = path.join(RADIO_DATA_PATH, "input_data.npy")
    input_data = open_memmap(input_data_file, mode="w+")
    assert np.all(np.abs(input_data[index, :]) == 0)
    energy_deposit = f_h5["atmosphere"]["EnergyDeposit"][:, :4]
    number_of_particles = f_h5["atmosphere"]["NumberOfParticles"][:, :4]
    path_along_shower = f_h5["atmosphere"]["Atmosphere"][:-1, 2]
    ref_index = None
    input_data.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sim_number",
        metavar="N",
        type=str,
        nargs="+",
        help="The Simulation number to be processed",
    )
    args = parser.parse_args()
    SIM_NUMBER = args.sim_number
    file_name = f"SIM{SIM_NUMBER}.hdf5"

    f_h5 = h5py.File(file_name, "r")

    sim_1, sim_2 = split_sim_number(int(SIM_NUMBER))
    idx = (sim_1 - 1) * 27 + (sim_2 - 1)

    flush_input_meta(SIM_NUMBER, f_h5, idx)

    flush_antenna_pos(f_h5, idx)

    flush_output_gece(f_h5, idx)

    flush_output_vBvvB(f_h5, idx)

    flush_output_meta(f_h5, idx)

    flush_input_data(f_h5, idx)
