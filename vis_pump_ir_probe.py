import os

import numpy as np
from numpy import ndarray

from matplotlib import pyplot as plt

def load_data_set(path):
    # Check whether combined data set
    # was already created before hand
    # in that case in can just be
    # loaded
    if "combined_data" in os.listdir(path):
        print("Combined data already exists loading directly.")

        f_path = os.path.join(path, "combined_data")
        
        data = np.load(os.path.join(f_path, "data.npy"))
        weights = np.load(os.path.join(f_path, "weights.npy"))
        counts = np.load(os.path.join(f_path, "counts.npy"))
        s2s_std = np.load(os.path.join(f_path, "s2s_std.npy"))
        
        return data, weights, counts, s2s_std
    
    # Get name of folder and
    # ergo filename in this folder
    filename = os.path.basename(path)
    # Get delay count
    delay_path = os.path.join(path, "scans")
    t = os.listdir(delay_path)
    t.sort()
    n_delays = int(t[-1][-3:]) + 1
    print("Detected {} delays".format(n_delays))
    # Get scan count
    scan_path = os.path.join(delay_path, "delay000")
    t = os.listdir(scan_path) 
    t.sort()
    n_scans = int(t[-1][1:7]) + 1 # Not safe 
    print("Detected {} scans".format(n_scans))

    # Get the size of one data array
    for file in t[:4]: # There should be 4 different files for each scan
        # The if statement should be superflous
        # because all files have the same dimension
        # (I believe)
        p = os.path.join(scan_path, file) # Path to file
        if "weights" in file:
            weights_shape = np.load(p).shape
        elif "counts" in file:
            counts_shape = np.load(p).shape
        elif "s2s_std" in file:
            s2s_std_shape = np.load(p).shape
        else:
            data_shape = np.load(p).shape

    # Preallocate arrays
    # Transmission data of scans for each delay
    data = np.zeros((n_scans, n_delays, *data_shape))
    # Inverse variance of transmission data for
    # each scan for each delay
    weights = np.zeros((n_scans, n_delays, *weights_shape))
    # Counts of each state of each scan for each delay
    counts = np.zeros((n_scans, n_delays, *counts_shape))
    #? Standard deviation of shot to shot signal
    #? of each scan for each delay
    s2s_std = np.zeros((n_scans, n_delays, *s2s_std_shape))

    for delay in range(n_delays):
        for scan in range(n_scans):
            # Generate file paths
            # There is probably a more efficient 
            # way of doing this using glob or os.walk
            delay_folder = str(delay).zfill(3) # bad naming
            delay_str = "d{}_".format(delay_folder)
            scan_str = "s{}_".format(str(scan).zfill(6))

            f_path = os.path.join(delay_path, "delay{}".format(delay_folder))
            
            data_name = scan_str + delay_str + filename + ".npy"
            weights_name = scan_str + delay_str + "weights_" + filename + ".npy"
            counts_name = scan_str + delay_str + "counts_" + filename + ".npy"
            s2s_std_name = scan_str + delay_str + "s2s_std_" + filename + ".npy"

            # Load data into array
            data[scan, delay] = np.load(os.path.join(f_path, data_name))
            weights[scan, delay] = np.load(os.path.join(f_path, weights_name))
            counts[scan, delay] = np.load(os.path.join(f_path, counts_name))
            s2s_std[scan, delay] = np.load(os.path.join(f_path, s2s_std_name))

    # Save data to compact files
    new_dir = os.path.join(path, "combined_data")
    os.mkdir(new_dir)

    np.save(os.path.join(new_dir, "data"), data)
    np.save(os.path.join(new_dir, "weights"), weights)
    np.save(os.path.join(new_dir, "counts"), counts)
    np.save(os.path.join(new_dir, "s2s_std"), s2s_std)

    return data, weights, counts, s2s_std

def average_transmission_with_counts(data: ndarray, counts: ndarray):
    # Average data
    avg_data = np.average(data, axis=0, weights=counts)
    # Calculate absorption
    absorption = -np.log10(avg_data)
    # Calculate difference signal
    signal = np.take(absorption, 1, axis=-1) - np.take(absorption, 0, axis=-1)

    return signal

def average_transmission_with_weights(data: ndarray, weights: ndarray):
    # Average data
    avg_data = np.average(data, axis=0, weights=weights)
    # Calculate absorption
    absorption = -np.log10(avg_data)
    # Calculate difference signal
    signal = np.take(absorption, 1, axis=-1) - np.take(absorption, 0, axis=-1)

    return signal
     
def average_signal_with_s2s(data: ndarray, s2s_std: ndarray):
    # Calculate signal for each scan
    # Calculate absorption
    absorption = -np.log10(data)
    # Calculate difference signal
    signal = np.take(absorption, 1, axis=-1) - np.take(absorption, 0, axis=-1)

    # Average signals
    # For this calculate inverse variance of 
    # s2s difference signal for each scan
    # to use as weights
    weights = np.float_power(s2s_std, -2)
    avg_signal = np.average(signal, axis=0, weights=weights)

    return avg_signal

def average_signal_without_weights(data: ndarray, s2s_std: ndarray):
    # Calculate signal for each scan
    # Calculate absorption
    absorption = -np.log10(data)
    # Calculate difference signal
    signal = np.take(absorption, 1, axis=-1) - np.take(absorption, 0, axis=-1)

    # Average signals
    avg_signal = np.average(signal, axis=0)

    return avg_signal


if __name__ == "__main__":
    d, w, c, s = load_data_set(r"C:\Users\H-Lab\Desktop\DiPeptide python\20200902_2020-09-02_DiPe_1000shots_6muJ_Python_000")

    t1 = average_transmission_with_counts(d, c)
    
