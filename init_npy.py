#! /usr/bin/env python
"""
Create npy files to be used by open_memmap
"""
import numpy as np
import os

memmaps_file_path = '/Users/denis/Desktop/BachelorThesis/memmaps'
HDF5_file_path = '/Users/denis/Desktop/BachelorThesis/data/177113844/1'
log_file_path = '/Users/denis/Desktop/BachelorThesis/log'

csv_file_path = '/Users/denis/Desktop/BachelorThesis/csv'

total_amount_of_measurements = len(os.listdir(f'{HDF5_file_path}/iron')) + len(os.listdir(f'{HDF5_file_path}/proton'))
parameters = 11
event_level_parameters = 13
number_of_antennas = 160
grammage_steps = 1142
time_bins = 256
dimensions_antenna_positions_vB_vvB = 3
dimensions_antenna_traces_vB_vvB = 3
dimensions_antenna_traces_ge_ce = 2
time_ge_ce_and_vB_vvB = 2

RADIO_DATA_PATH = "/Users/denis/Desktop/BachelorThesis/memmaps/test2"

def _init_(path_,total_amount_of_measurements = 33, parameters = 11, event_level_parameters = 13, number_of_antennas = 160,grammage_steps = 1142, time_bins = 256, dimensions_antenna_positions_vB_vvB = 3, dimensions_antenna_traces_vB_vvB = 3,dimensions_antenna_traces_ge_ce = 2,time_ge_ce_and_vB_vvB = 2):
    '''
    Parameters:
    
        path_: path where the memmap files should be created

        total_amount_of_measurements: proton + iron measurements (total amount of SIM files)

        parameters: used parameters

        event_level_parameters: used event level parameters

        number_of_antennas: the total number of antennas

        grammage_steps: how many X steps were taken

        time_bins: how many time_bins there is

        dimensions_antenna_positions_vB_vvB: XYZ

        dimensions_antenna_traces_vB_vvB: XYZ,

        dimensions_antenna_traces_ge_ce: X' Y'

        time_ge_ce_and_vB_vvB: time of ge_ce and vB_vvB
    
    '''
    dtypeInit = "float32"
    
    file_name = os.path.join(path_, "input_data")
    #os.remove(file_name)
    #(number of simulations, grammage steps, parameters)
    # for iron 12, proton 21
    fp1 = np.memmap(file_name, dtype=f"{dtypeInit}", mode="w+", shape=(total_amount_of_measurements, grammage_steps, parameters),offset=0)
    #print(fp1.shape)
    fp1[:] = 0
    np.save(file_name, fp1)
    file_name = os.path.join(path_, "meta_data")
    #os.remove(file_name)
    #(number of simulations, event level parameters)
    fp2 = np.memmap(file_name, dtype=f"{dtypeInit}", mode="w+", shape=(total_amount_of_measurements, event_level_parameters),offset=0)
    #print(fp2.shape)
    fp2[:] = 0
    np.save(file_name, fp2)
    file_name = os.path.join(path_, "antenna_pos_data")
    #os.remove(file_name)
    #(number of simulations, number of antennas, ant positions)
    fp3 = np.memmap(file_name, dtype=f"{dtypeInit}", mode="w+", shape=(total_amount_of_measurements, number_of_antennas, dimensions_antenna_positions_vB_vvB),offset=0)
    #print(fp3.shape)
    fp3[:] = 0
    np.save(file_name, fp3)
    file_name = os.path.join(path_, "output_meta_data")
    #os.remove(file_name)
    #(number of simulations, number of antennas, time (s))

    fp4 = np.memmap(file_name, dtype=f"{dtypeInit}", mode="w+", shape=(total_amount_of_measurements, number_of_antennas, time_ge_ce_and_vB_vvB),offset=0)
    #print(fp4.shape)
    fp4[:] = 0
    np.save(file_name, fp4)
    #file_name = os.path.join(path_, "output_gece_data")
    ##os.remove(file_name)
    #fp5 = np.memmap(file_name, dtype=f"{dtypeInit}", mode="w+", shape=(total_amount_of_measurements, number_of_antennas, time_bins, dimensions_antenna_traces_ge_ce),offset=0)
    ##print(fp5.shape)
    #fp5[:] = 0
    #np.save(file_name, fp5)
    #(number of simulations, number of antennas, time bins, polarizations)
    file_name = os.path.join(path_, "output_vBvvB_data")
    #os.remove(file_name)
    fp6 = np.memmap(file_name, dtype=f"{dtypeInit}", mode="w+", shape=(total_amount_of_measurements, number_of_antennas, time_bins, dimensions_antenna_traces_vB_vvB),offset=0)
    #print(fp6.shape)
    fp6[:] = 0
    np.save(file_name, fp6)
    
if __name__ == "__main__":    #PLAYING CODE
    _init_(RADIO_DATA_PATH,total_amount_of_measurements, parameters, event_level_parameters, number_of_antennas,grammage_steps, time_bins, dimensions_antenna_positions_vB_vvB, dimensions_antenna_traces_vB_vvB,dimensions_antenna_traces_ge_ce,time_ge_ce_and_vB_vvB)
