import re
import h5py
import os
import fnmatch
import csv
import sys
import subprocess
import numpy as np
import psutil
from datetime import datetime, timedelta
from tqdm import tqdm

from init_npy import _init_
from coreas_to_hdf5 import FilesTransformHdf5ToHdf5
from hdf5_to_memmapfile import write_memmapfile, write_csv_file

from init_npy import parameters, event_level_parameters, number_of_antennas, time_bins, dimensions_antenna_positions_vB_vvB, dimensions_antenna_traces_vB_vvB,dimensions_antenna_traces_ge_ce,time_ge_ce_and_vB_vvB

def sorting_files(files):
    sorted_files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))
    return sorted_files

def time_difference(past_datetime):
    """
    Calculates the difference between the current time and a given past time.

    Parameters:
    - past_time (str): A timestamp string (e.g., "2024-02-08 14:30:00").
    - format (str): The format of the given timestamp (default: "%Y-%m-%d %H:%M:%S").

    Returns:
    - timedelta: The difference between the current time and past_time.
    - str: A formatted string showing the difference in days, hours, minutes, and seconds.
    """
    try:
        current_datetime = datetime.now()  # Get current time

        difference = current_datetime - past_datetime  # Calculate time difference

        # Formatting the difference in a readable way
        days = difference.days
        seconds = difference.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        formatted_diff = f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds"
        return formatted_diff

    except ValueError as e:
        return f"Invalid time format: {e}"

def create_log_file(file_path,time_for_operation, log_file="logs/file_log.txt"):
    '''
    Parameters:
    
        file_path: The path of the file to log.
        log_file: The path of the log file (default: "logs/file_log.txt").
    '''
    # Ensure the log file directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Ensure the file exists before logging (optional)
    if os.path.isfile(file_path):
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"It took {time_for_operation} to write {file_path}\n")
            
    elif os.path.isdir(file_path):
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"It took {time_for_operation} to write {file_path}\n")
        
    else:
        print(f"Warning: The file '{file_path}' does not exist, skipping log.")



def is_file_locked(filepath):
    """Check if an HDF5 file is locked by another process."""
    'IMPORTANT CHECK'
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' does not exist.")
        return True

    try:
        # Try opening in append mode ('a') to check if it's locked
        with h5py.File(filepath, 'a'):
            #print(f"File '{filepath}' is NOT locked anymore. You can access it.")
            return False
    except BlockingIOError:
        print(f"File '{filepath}' is still LOCKED by another process!")
        return True
    except Exception as e:
        print(f"Unexpected error: {e}")
        return True
    
def close_hdf5_if_locked(filepath):
    """
    Check if an HDF5 file is locked by a process and close it if possible.
    Only works if the same user has opened the file.
    This happened to me after I forgot to close it in my jupyter notebook, 
    so I will check if the file is opened now before trying to acces it,
    so the automatic process doesn't fail.
    """
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' does not exist.")
        return False

    # Check which processes are using this file
    for process in psutil.process_iter(['pid', 'name', 'open_files']):
        try:
            if process.info['open_files']:
                for file in process.info['open_files']:
                    if file.path == filepath:
                        print(f"File is locked by process {process.info['name']} (PID {process.info['pid']}). Attempting to close...")
                        process.terminate()  # Tries to close the process
                        process.wait(timeout=5)  # Wait up to 5 seconds for it to close
                        print(f"Process {process.info['pid']} closed. File should be accessible now.")
                        return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue  # Process may have closed already
        
    #print(f"File '{filepath}' is not locked. You can use it.")
    return False  # File was not locked

def create_folder(base_path, folder_name):
    """
    Creates a folder inside the given base path if it does not exist.

    Parameters:
        base_path : The directory where the folder should be created.
        folder_name (str): The name of the folder to create.

    Returns:
        The full path of the created (or existing) folder.
    
    Raises:
        Exception: If folder creation fails.
    """
    folder_path = os.path.join(base_path, folder_name)  # Creating full path

    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)  # Create folder
            print(f"Folder '{folder_name}' was created in '{base_path}'.")
        except Exception as e:
            raise Exception(f"Error creating folder: {e}")
    else:
        print(f"Folder '{folder_name}' already exists.")

    return folder_path  # Return the path if needed


def find_Iron_and_Proton(path_):
    
    try:
        matching_particle = [f for f in os.listdir(f'{path_}') if fnmatch.fnmatch(f, "iron") or fnmatch.fnmatch(f, "proton")]
        
    except FileNotFoundError:
        print(f"Folder not found: {path_}")
        
    except PermissionError:
        print(f"No permission to access: {path_}")
    
    return matching_particle
    
    
def find_SIM(path_):
      
    try:
        #print(f"SIM Contents of '{path_}':")
        matching_SIM = [f for f in os.listdir(f'{path_}') if fnmatch.fnmatch(f, "SIM*.hdf5")]
        #for file in matching_SIM:
                #print(file)
        
    except FileNotFoundError:
        print(f"Folder not found: {path_}")
        
    except PermissionError:
        print(f"No permission to access: {path_}")
        
    return matching_SIM


def getting_SIM_number(SIM):
    
    pattern = re.compile(r"SIM(\d+)\.hdf5")
    match = pattern.match(SIM)
    
    if match:
        # Extracting the number and converting it to an integer (removing zeros
        SIM_NUMBER = int(match.group(1))
        #print(SIM_NUMBER)
                
    return SIM_NUMBER

def getting_amount_of_SIM_and_GrammageSteps(HDF5_file_path,proton_or_iron = True):
    amount_of_sims = []
    
    all_grammage_steps = []
                
    all_dx = []
                
    
    for ifolder, folders in tqdm(enumerate(os.listdir(f'{HDF5_file_path}')),desc='Initialization: ',total=len(os.listdir(f'{HDF5_file_path}'))):
        folder_paths = f'{HDF5_file_path}/{folders}'
        if not os.path.isdir(folder_paths):
            continue
        for folders2 in os.listdir(f'{folder_paths}'):
            Proton_Iron_paths = f'{folder_paths}/{folders2}'
            amount_of_sims.append(len(os.listdir(f'{Proton_Iron_paths}/iron')) + len(os.listdir(f'{Proton_Iron_paths}/proton')))
            
            for particle in os.listdir(f'{Proton_Iron_paths}'):
                
                if particle == 'proton':
                    pass
                else:
                    proton_or_iron = False
        
                matching_SIMs = find_SIM(f'{Proton_Iron_paths}/{particle}')
                sorted_SIM = sorting_files(matching_SIMs) #Sorted via number
                
                for SIM in sorted_SIM:
                    chosen_SIM = f'{Proton_Iron_paths}/{particle}/{SIM}'
                    close_hdf5_if_locked(chosen_SIM)
                    is_file_locked(chosen_SIM)
                    f_h5 = h5py.File(chosen_SIM, "r")
                    x = f_h5["atmosphere"]["NumberOfParticles"][:, 0] #could be the wrong size thing
                    f_h5.close()
                    
                    all_grammage_steps.append(len(x))
                    all_dx.append(x[1]-x[0])
                    
                    
    grammage_steps = min(all_grammage_steps)
    print(f'Amount of Grammage steps: {grammage_steps}')
    print(f'Min dx: {min(all_dx)}, Max dx: {max(all_dx)}')
    Sim_amount = np.sum(amount_of_sims)
    
    return int(Sim_amount),int(grammage_steps)



def initializing(memmaps_file_path,memmap_folder_name,HDF5_file_path,test=False,i=1):
    

    amount_of_measurements,grammage_steps = getting_amount_of_SIM_and_GrammageSteps(HDF5_file_path)
    
    create_folder(memmaps_file_path,memmap_folder_name)
    
    memmap_file_path = f'{memmaps_file_path}/{memmap_folder_name}'
    
    if test:
        _init_(memmap_file_path,i, parameters, event_level_parameters, number_of_antennas,grammage_steps, time_bins, dimensions_antenna_positions_vB_vvB, dimensions_antenna_traces_vB_vvB,dimensions_antenna_traces_ge_ce,time_ge_ce_and_vB_vvB)
    else:
        _init_(memmap_file_path,amount_of_measurements, parameters, event_level_parameters, number_of_antennas,grammage_steps, time_bins, dimensions_antenna_positions_vB_vvB, dimensions_antenna_traces_vB_vvB,dimensions_antenna_traces_ge_ce,time_ge_ce_and_vB_vvB)
    return memmap_file_path,grammage_steps



def converting_one_dataset(j,memmap_file_path,HDF5_file_path,csv_file_path,log_file_path,grammage_steps,proton_or_iron = True):
    
    start_datetime = datetime.now() 
    
    #IRON AND PROTON
            
    matching_particles = find_Iron_and_Proton(HDF5_file_path)
    
    if matching_particles[0] == 'proton':
        pass
    else:
        matching_particles = matching_particles[::-1]
        
    #GETTING EVERY MEASUREMENT FOR IRON AND PROTON
    
   
    for particle in matching_particles:
        
        if particle == 'proton':
            pass
        else:
            proton_or_iron = False
        
        matching_SIMs = find_SIM(f'{HDF5_file_path}/{particle}')
        sorted_SIM = sorting_files(matching_SIMs) #Sorted via number

        for SIM in sorted_SIM:
            
            start_sim_datetime = datetime.now() 
            
            idx = j
            
            SIM_NUMBER = getting_SIM_number(SIM)
            
            print(f'IDX value:{idx}')
                
            print('--------------------------------')
            print(f'Processing: {SIM} ...')
            
            chosen_SIM = f'{HDF5_file_path}/{particle}/{SIM}'
            
            if proton_or_iron:
                write_csv_file('Proton_SIM_vs_Index',csv_file_path,chosen_SIM,idx)
            else:
                write_csv_file('Iron_SIM_vs_Index',csv_file_path,chosen_SIM,idx)
                
            
            close_hdf5_if_locked(chosen_SIM)
            
            is_file_locked(chosen_SIM)
            
            FilesTransformHdf5ToHdf5(chosen_SIM)
            
            
            f_h5 = h5py.File(chosen_SIM, "r")

            write_memmapfile(memmap_file_path,chosen_SIM,SIM_NUMBER,f_h5,idx,csv_file_path,grammage_steps)
            
            
            f_h5.close()
            
            sim_duration = time_difference(start_sim_datetime)
            
            create_log_file(chosen_SIM,sim_duration,f'{log_file_path}/SIM_log.txt')
            
            
            j += 1
            
    print('--------------------------------')
    print('................................')
    print(f'{memmap_file_path} FINISHED!')
    print('................................')
    print('--------------------------------')
    
    
    time_for_operation = time_difference(start_datetime)

    create_log_file(HDF5_file_path,time_for_operation,f'{log_file_path}/MEASUREMENTS_log.txt')
    
    return j

def run_auto(memmaps_file_path,HDF5_file_path,log_file_path,csv_file_path):
    
    j = 0 
    
    memmap_file_path,grammage_steps = initializing(memmaps_file_path,'memmap',HDF5_file_path)
    
    for ifolders, folders in tqdm(enumerate(os.listdir(f'{HDF5_file_path}')),desc='Converting Progress: ', total=len(os.listdir(f'{HDF5_file_path}'))):
        
        folder_paths = f'{HDF5_file_path}/{folders}'
        
        if not os.path.isdir(folder_paths):
            continue
        
        for folders2 in os.listdir(f'{folder_paths}'):
            
            Proton_Iron_paths = f'{folder_paths}/{folders2}'
            
            new_j = converting_one_dataset(j,memmap_file_path,Proton_Iron_paths,csv_file_path,log_file_path,grammage_steps)
            
            j = new_j
            
def test_conversion(memmaps_file_path,HDF5_file_path,log_file_path,csv_file_path):
    j = 0
    
    memmap_file_path = initializing(memmaps_file_path,'memmap',HDF5_file_path,test=True)
    
    folders = os.listdir(f'{HDF5_file_path}')[0]
        
    folder_paths = f'{HDF5_file_path}/{folders}'
        
    for folders2 in os.listdir(f'{folder_paths}'):
            
        Proton_Iron_paths = f'{folder_paths}/{folders2}'
            
        new_j = converting_one_dataset(j,memmap_file_path,Proton_Iron_paths,csv_file_path,log_file_path)
            
        j = new_j
    

if __name__ == "__main__":
    
    memmaps_file_path = '/cr/work/stiben'
    HDF5_file_path = '/cr/work/stiben/Testdata'
    log_file_path = '/cr/work/stiben/log'
    csv_file_path = '/cr/work/stiben/csv'
    
    #memmaps_file_path = '/Users/denis/Desktop/BachelorThesis'
    #HDF5_file_path = '/Users/denis/Desktop/BachelorThesis/data'
    #log_file_path = '/Users/denis/Desktop/BachelorThesis/log'
    #csv_file_path = '/Users/denis/Desktop/BachelorThesis/csv'
    
    test = False
    
    if test:
        test_conversion(memmaps_file_path,HDF5_file_path,log_file_path,csv_file_path)
    else:
        run_auto(memmaps_file_path,HDF5_file_path,log_file_path,csv_file_path)