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

    # Later dynamically figure out number of positions
    n_pos = 2
    # Preallocate arrays
    # Transmission data of scans for each delay
    data = np.zeros((n_pos, n_scans, n_delays, *data_shape))
    # Inverse variance of transmission data for
    # each scan for each delay
    weights = np.zeros((n_pos, n_scans, n_delays, *weights_shape))
    # Counts of each state of each scan for each delay
    counts = np.zeros((n_pos, n_scans, n_delays, *counts_shape))
    #? Standard deviation of shot to shot signal
    #? of each scan for each delay
    s2s_std = np.zeros((n_pos, n_scans, n_delays, *s2s_std_shape))

    for pos in range(n_pos):
        for delay in range(n_delays):
            for scan in range(n_scans):
                # Generate file paths
                # There is probably a more efficient 
                # way of doing this using glob or os.walk
                delay_folder = str(delay).zfill(3) # bad naming
                pos_str = "pos{}_".format(str(pos).zfill(3))
                delay_str = "d{}_".format(delay_folder)
                scan_str = "s{}_".format(str(scan).zfill(6))

                f_path = os.path.join(delay_path, "delay{}".format(delay_folder))
                
                data_name = scan_str + delay_str + pos_str + filename + ".npy"
                weights_name = scan_str + delay_str + pos_str + "weights_" + filename + ".npy"
                counts_name = scan_str + delay_str + pos_str + "counts_" + filename + ".npy"
                s2s_std_name = scan_str + delay_str + pos_str + "s2s_std_" + filename + ".npy"

                # Load data into array
                data[pos, scan, delay] = np.load(os.path.join(f_path, data_name))
                weights[pos, scan, delay] = np.load(os.path.join(f_path, weights_name))
                counts[pos, scan, delay] = np.load(os.path.join(f_path, counts_name))
                s2s_std[pos, scan, delay] = np.load(os.path.join(f_path, s2s_std_name))

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
    avg_data = np.average(data, axis=1, weights=counts)
    # Calculate absorption
    absorption = -np.log10(avg_data)
    # Calculate difference signal
    signal = np.take(absorption, 1, axis=-1) - np.take(absorption, 0, axis=-1)

    return signal

def average_transmission_with_weights(data: ndarray, weights: ndarray):
    #! Needs work!!!
    # Average data
    avg_data = np.average(data, axis=0, weights=weights)
    # Calculate absorption
    absorption = -np.log10(avg_data)
    # Calculate difference signal
    signal = np.take(absorption, 1, axis=-1) - np.take(absorption, 0, axis=-1)

    return signal
     
def average_signal_with_s2s(data: ndarray, s2s_std: ndarray):
    #! Needs work!!!
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

def average_signal_without_weights(data: ndarray):
    #! Needs work!!!
    # Calculate signal for each scan
    # Calculate absorption
    absorption = -np.log10(data)
    # Calculate difference signal
    signal = np.take(absorption, 1, axis=-1) - np.take(absorption, 0, axis=-1)

    # Average signals
    avg_signal = np.average(signal, axis=0)

    return avg_signal

def generate_legacy_data_format(difference_signal, delays, probe_axis):
    #! Needs work!!!
    old_formats = []
    for pos in range(2):
        # The old format has the following shape:
        # one row for each delay and each pixel pair
        # such that the columns are:
        # (delay, wavenumber, difference signal, error)
        # error is probably the s2s standard deviation 
        # of a given pixel for a given delay

        # The number of rows in the .txt file thus is
        # number of delays * number of pixel pairs

        # For the time being we are setting the weights to 1

        old_format = np.ones((delays.size*probe_axis.size, 4))
        old_format[:, 0] = np.repeat(delays, probe_axis.size)
        old_format[:, 1] = np.tile(probe_axis, delays.size)
        old_format[:, 2] = difference_signal[pos].flatten()
        old_formats.append(old_format)
        
    return old_formats

if __name__ == "__main__":
    path = r"C:\Users\H-Lab\Documents\data_analysis\split sample cell test daten\2020-09-08\20200907_first split sample test_000"
    d, w, c, s = load_data_set(path)
    print(d.shape)
    # for p in range(2):
    #     np.save("pos{}_complete_data_set".format(p), d[p])
    #     np.save("pos{}_weights_complete_data_set".format(p), w[p])
    #     np.save("pos{}_counts_complete_data_set".format(p), c[p])
    #     np.save("pos{}__s2s_std_complete_data_set".format(p), s[p])
    t1 = average_transmission_with_counts(d, c)
    print(t1.shape)

    # t2 = average_transmission_with_weights(d, w)
    # t3 = average_signal_with_s2s(d, s)
    # t4 = average_signal_without_weights(d)

    # Load delays (ignore weights)
    delays = np.load(os.path.join(path, "delay_file_20200907_first split sample test_000.npy"))[:, 0]

    # Load wn axis (probe axis)
    probe_axis = np.load(os.path.join(path, "probe_wn_axis_20200907_first split sample test_000.npy"))

    old_formats = generate_legacy_data_format(t1, delays, probe_axis)
    for pos, old_format in enumerate(old_formats):
        np.savetxt(os.path.join(path, "pos{}_old_format.txt".format(pos)), old_format, delimiter="\t")

    # np.savetxt(os.path.join(path, "average_transmission_with_counts"), t1)
    # np.savetxt(os.path.join(path, "average_transmission_with_weights"), t2)
    # np.savetxt(os.path.join(path, "average_signal_with_s2s"), t3)
    # np.savetxt(os.path.join(path, "average_signal_without_weights"), t4)

    # nrows = 8
    # ncols = 8
    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    # for freq in range(t1.shape[1]):
    #     ax_idx = np.unravel_index(freq, (nrows, ncols))
    #     axes[ax_idx].plot(t1[:, freq], label="average_transmission_with_counts", linewidth= 0.5)
    #     axes[ax_idx].plot(t2[:, freq], label="average_transmission_with_weights", linewidth= 0.5)
    #     axes[ax_idx].plot(t3[:, freq], label="average_signal_with_s2s", linewidth= 0.5)
    #     axes[ax_idx].plot(t4[:, freq], label="average_signal_without_weights", linewidth= 0.5)

    #     axes[ax_idx].legend(fontsize=5)
    # # fig, ax = plt.subplots()

    # # ax.plot(t1[:, 20], label="average_transmission_with_counts", linewidth= 0.5)
    # # ax.plot(t2[:, 20], label="average_transmission_with_weights", linewidth= 0.5)
    # # ax.plot(t3[:, 20], label="average_signal_with_s2s", linewidth= 0.5)
    # # ax.plot(t4[:, 20], label="average_signal_without_weights", linewidth= 0.5)
    
    # # ax.set_xscale("log")
    plt.show()
