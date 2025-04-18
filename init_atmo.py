import os
import numpy as np
import traceback
from radiotools.atmosphere.models import Atmosphere

lofar_sim_path = '/home/denis/lofar_hdf5_sims_batch2'
atm_dir_path = '/home/denis/atmosphere_files'

for isim, sim_dir in enumerate(os.listdir(lofar_sim_path)):
    if sim_dir.isdigit() == False:
        continue
    sim_nr = int(sim_dir) # just getting the event number
    # first generate constants for each atmospheric model
    atm_file_path = os.path.join(atm_dir_path, f"ATMOSPHERE_{sim_dir}.DAT")
    try:
        try:
            atm_model = Atmosphere(gdas_file=atm_file_path, curved=False)
        except FileNotFoundError:
            print(f"Atmosphere file not found for {sim_dir}")
            continue
    except SystemExit:
        traceback.print_exc()
        continue