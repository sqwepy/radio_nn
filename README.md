Setting up the auto_conversion:

Change the path of MEMMAP_file_path to where you want to store your MEMMAP file.

Change the path of DATA_file_path to where the '141414141' type of folders are, which have to have a structure like '41941091/8/iron or proton/simXXXXXX.hdf5 

Auto conversion works on hdf5 files that have all the crucial info and also works on files that already have the highlevel info. If a file already has the highlevel info, the file just gets written to the memmap file.

After auto_conversion is done:

In the first path there is going to appear several folders. 

The csv sheets in the csv folder have data on what index is what antenna and on what sim index is an iron or a proton measurement.
This info is important to navigate through the memmap file.

The log files have info on how long it took to write folders or sims to the memmap file. Also has info on what sims didn't have the crucial info (Failed sims) and on the initialization process.