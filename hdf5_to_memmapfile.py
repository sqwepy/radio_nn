#! /usr/bin/env python
"""
Load all the data from HDF5 files to Memmap file to be used by Pytorch later.
"""
import h5py
import numpy as np
from numpy.lib.format import open_memmap
import os.path as path
import argparse
import os
import fnmatch
import re

RADIO_DATA_PATH = "/Users/denis/Desktop/BachelorThesis/data/177113844/1"


def density_at_Xmax(f_h5):
    x_max = f_h5['atmosphere'].attrs['Gaisser-Hillas-Fit'][2]
    X = np.array(f_h5["atmosphere"]["Atmosphere"][:, 0])
    n = np.argmin(x_max-X)
    h = f_h5["atmosphere"]["Atmosphere"][:, 1]
    print(n)
    density = f_h5["atmosphere"]["Density"][n]
    h_max = h[n]
    return density, h_max


def split_sim_number(nn):
    """Split the number to two parts."""
    a = nn // 100
    b = nn % 100
    return a, b


def flush_input_meta(output_path,SIM_NUMBER, f_h5, index, dtypeInit):
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

    meta_data_file = path.join(output_path, "meta_data.npy")
    meta_data = open_memmap(meta_data_file, mode="r+", dtype=f'{dtypeInit}')
    
    #assert max(meta_data[:, 0]) < int(SIM_NUMBER)
    # check and set always
    #assert np.all(np.abs(meta_data[index, :]) == 0)
    meta_data[index, 0] = int(SIM_NUMBER) 
    meta_data[index, 1] = np.cos(float(f_h5["CoREAS"].attrs['ShowerZenithAngle']))
    
    meta_data[index, 2] = f_h5['atmosphere'].attrs['Gaisser-Hillas-Fit'][2]
    
    d_xmax, h_xmax = density_at_Xmax(f_h5)
    meta_data[index, 3] = d_xmax
    meta_data[index, 4] = h_xmax
    meta_data[index, 5] = f_h5["highlevel"].attrs["Eem"] 
    
    print(f_h5["CoREAS"].attrs['GeomagneticAngle']) 
    meta_data[index, 6] = np.sin(float(f_h5["CoREAS"].attrs['GeomagneticAngle']))
    meta_data[index, 7] = f_h5["CoREAS"].attrs['MagneticFieldInclinationAngle']
    meta_data[index, 8] = f_h5["CoREAS"].attrs['RotationAngleForMagfieldDeclination']
    meta_data[index, 9] = f_h5["CoREAS"].attrs['MagneticFieldStrength']
    meta_data[index, 10] = f_h5["inputs"].attrs["PRMPAR"]
    
    meta_data[index, 11] = f_h5["inputs"].attrs['ERANGE'][0]
    meta_data[index, 12] = np.deg2rad(float(f_h5["CoREAS"].attrs['ShowerAzimuthAngle']))
    meta_data.flush()


def flush_output_meta(output_path,f_h5, index, dtypeInit):
    output_meta_file = path.join(output_path, "output_meta_data.npy")
    output_meta = open_memmap(output_meta_file, mode="r+", dtype=f'{dtypeInit}')
    assert np.all(np.abs(output_meta[index, :]) == 0)
    antennas_trace_gece_f_h5 = f_h5[f"/highlevel/traces/ge_ce"]
    antennas_trace_vBvvB_f_h5 = f_h5[f"/highlevel/traces/vB_vvB"]
    label_index = 0
    for label in antennas_trace_vBvvB_f_h5.keys():
        if label.split("_")[0] != "pos":
            continue
        assert 0 <= label_index < 160  #Anzahl antennen oder position antennen???
        output_meta[index, label_index] = np.array(
            np.min(antennas_trace_gece_f_h5[label][:, 0]),
            np.min(antennas_trace_vBvvB_f_h5[label][:, 0]),
        )
        label_index += 1
    output_meta.flush()


def flush_output_vBvvB(output_path,f_h5, index, dtypeInit):
    output_vBvvB_file = path.join(output_path, "output_vBvvB_data.npy")
    output_vBvvB = open_memmap(output_vBvvB_file, mode="r+", dtype=f'{dtypeInit}')
    assert np.all(np.abs(output_vBvvB[index, :]) == 0)
    antennas_trace_vBvvB_f_h5 = f_h5[f"/highlevel/traces/vB_vvB"]
    label_index = 0
    for label in antennas_trace_vBvvB_f_h5.keys():
        if label.split("_")[0] != "pos":
            continue
        assert 0 <= label_index < 160
        output_vBvvB[index, label_index] = antennas_trace_vBvvB_f_h5[label][:, 1:]
        label_index += 1
    output_vBvvB.flush()


def flush_output_gece(output_path,f_h5, index, dtypeInit):
    output_gece_file = path.join(output_path, "output_gece_data.npy")
    output_gece = open_memmap(output_gece_file, mode="r+", dtype=f'{dtypeInit}')
    assert np.all(np.abs(output_gece[index, :]) == 0)
    antennas_trace_gece_f_h5 = f_h5[f"/highlevel/traces/ge_ce"]
    label_index = 0
    for label in antennas_trace_gece_f_h5.keys():
        if label.split("_")[0] != "pos":
            continue
        assert 0 <= label_index < 160
        output_gece[index, label_index] = antennas_trace_gece_f_h5[label][:, 1:-1]
        label_index += 1
    output_gece.flush()


def flush_antenna_pos(output_path,f_h5, index, dtypeInit):
    antenna_pos_file = path.join(output_path, "antenna_pos_data.npy")
    antenna_pos = open_memmap(antenna_pos_file, mode="r+", dtype=f'{dtypeInit}')
    assert np.all(np.abs(antenna_pos[index, :]) == 0)
    antennas_pos_f_h5 = f_h5[f"/highlevel/positions/vB_vvB"]
    label_index = 0
    for label in antennas_pos_f_h5.keys():
        if label.split("_")[0] != "pos":
            continue
        assert 0 <= label_index < 160
        antenna_pos[index, label_index] = antennas_pos_f_h5[label]
        label_index += 1
    antenna_pos.flush()


def flush_input_data(output_path,f_h5, index, dtypeInit):
    input_data_file = path.join(output_path, "input_data.npy")
    input_data = open_memmap(input_data_file, mode="r+", dtype=f'{dtypeInit}')
    assert np.all(np.abs(input_data[index, :]) == 0)
    
    energy_deposit = f_h5["atmosphere"]["EnergyDeposit"][:, :4]
    number_of_particles = f_h5["atmosphere"]["NumberOfParticles"][:, :4]
    
    path_along_shower = f_h5["atmosphere"]["Atmosphere"][:, 1] #???? missing
    
    ref_index = f_h5["atmosphere"]["Ref Index"] 
    density = f_h5["atmosphere"]["Density"] 
    assert (
        len(energy_deposit)
        == len(number_of_particles)
        == len(path_along_shower)
        == len(ref_index)
        == len(density)
    )
    x_len = len(density)
    input_data[index, :x_len, :4] = energy_deposit
    input_data[index, :x_len, 4:8] = number_of_particles
    input_data[index, :x_len, 8] = path_along_shower #missing = height
    input_data[index, :x_len, 9] = ref_index
    input_data[index, :x_len, 10] = density
    print("X_Length", x_len)
    input_data.flush()
    
    
def write_memmapfile(output_path,SIM_name,SIM_NUMBER,f_h5,idx):
    
    dtypeInit = "float32"
    
    flush_input_meta(output_path,SIM_NUMBER, f_h5, idx, dtypeInit)

    flush_antenna_pos(output_path,f_h5, idx, dtypeInit)

    flush_output_gece(output_path,f_h5, idx, dtypeInit)

    flush_output_vBvvB(output_path,f_h5, idx, dtypeInit)

    flush_output_meta(output_path,f_h5, idx, dtypeInit)

    flush_input_data(output_path,f_h5, idx, dtypeInit)
    
    print(f'Processing memmap done: {SIM_name} Index: {idx}')
    
if __name__ == "__main__":    #PLAYING CODE
    

    path_memmapfile = "/Users/denis/Desktop/BachelorThesis/memmaps/177113844/1"
    path_data = "/Users/denis/Desktop/BachelorThesis/data/177113844/1/proton/SIM000005.hdf5"
    
    f_h5 = h5py.File("/Users/denis/Desktop/BachelorThesis/data/177113844/1/proton/SIM000005.hdf5",'r')

    write_memmapfile("/Users/denis/Desktop/BachelorThesis/memmaps/test",'SIM000005',5,f_h5,1)


