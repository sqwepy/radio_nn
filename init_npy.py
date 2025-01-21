#! /usr/bin/env python
"""
Create npy files to be used by open_memmap
"""
import numpy as np
import os

RADIO_DATA_PATH = "/home/pranav/work-stuff-unsynced/radio_data"
file_name = os.path.join(RADIO_DATA_PATH, "input_data")
os.remove(file_name)
fp1 = np.memmap(file_name, dtype="float32", mode="w+", shape=(2158 * 27, 300, 11))
print(fp1.shape)
np.save(file_name, fp1)
file_name = os.path.join(RADIO_DATA_PATH, "meta_data")
os.remove(file_name)
fp2 = np.memmap(file_name, dtype="float32", mode="w+", shape=(2158 * 27, 13))
print(fp2.shape)
np.save(file_name, fp2)
file_name = os.path.join(RADIO_DATA_PATH, "antenna_pos_data")
os.remove(file_name)
fp3 = np.memmap(file_name, dtype="float32", mode="w+", shape=(2158 * 27, 240, 3))
print(fp3.shape)
np.save(file_name, fp3)
file_name = os.path.join(RADIO_DATA_PATH, "output_meta_data")
os.remove(file_name)
fp4 = np.memmap(file_name, dtype="float32", mode="w+", shape=(2158 * 27, 240, 2))
print(fp4.shape)
np.save(file_name, fp4)
file_name = os.path.join(RADIO_DATA_PATH, "output_gece_data")
os.remove(file_name)
fp5 = np.memmap(file_name, dtype="float32", mode="w+", shape=(2158 * 27, 240, 256, 2))
print(fp5.shape)
np.save(file_name, fp5)
file_name = os.path.join(RADIO_DATA_PATH, "output_vBvvB_data")
os.remove(file_name)
fp6 = np.memmap(file_name, dtype="float32", mode="w+", shape=(2158 * 27, 240, 256, 3))
print(fp6.shape)
np.save(file_name, fp6)
