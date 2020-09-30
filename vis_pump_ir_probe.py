import os

import numpy as np
from numpy import ndarray

from matplotlib import pyplot as plt

def load_data_set(path):
    # Get name of folder and
    # ergo filename in this folder
    filename = os.path.basename(path)
    # Load delay
    # and probe_wn_axis etc.
    delays = np.load(os.path.join(path, "delay_file_" + filename +".npy")) 
    probe_axis = np.load(os.path.join(path, "probe_wn_axis_" + filename +".npy"))

    # Check whether combined data set
    # was already created before hand
    # in that case it can just be
    # loaded
    if "combined_data" in os.listdir(path):
        print("Combined data already exists loading directly.")

        f_path = os.path.join(path, "combined_data")
        
        data = np.load(os.path.join(f_path, "data.npy"))
        weights = np.load(os.path.join(f_path, "weights.npy"))
        counts = np.load(os.path.join(f_path, "counts.npy"))
        s2s_std = np.load(os.path.join(f_path, "s2s_std.npy"))
        
        return data, weights, counts, s2s_std, delays, probe_axis
    
    # Get delay count
    n_delays = delays.shape[0]
    print("Detected {} delays".format(n_delays))

    # Get scan count
    # For this we go into into the folder
    # where the scans are saved for the last delay
    # and take the last entry of the sorted list
    # Because we only want to detect complete
    # scans
    #! Change this to not throw away scan data later
    scan_path = os.path.join(
                            path,
                            "scans",
                            "delay{}".format(str(n_delays-1).zfill(3))
                            )
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

            f_path = os.path.join(path, "scans", "delay{}".format(delay_folder))
            
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

    return data, weights, counts, s2s_std, delays, probe_axis

def load_average_data(path):
    avg_path = os.path.join(path, "averaged_data")
    
    # Get delay count
    delay_path = os.path.join(path, "scans")
    t = os.listdir(delay_path)
    t.sort()
    n_delays = int(t[-1][-3:]) + 1
    print("Detected {} delays".format(n_delays))

    # Get size of one array
    data_shape = np.load(os.path.join(avg_path, os.listdir(avg_path)[0])).shape

    # Preallocate arrays
    data = np.zeros((n_delays, *data_shape))
    
    # Load data into array
    for delay, file in enumerate(os.listdir(avg_path)):
        # Generate a path for each file
        file_path = os.path.join(
                                avg_path,
                                file
                                )
        # write data in the corresponding
        # delay dimension of transmission
        data[delay] = np.load(file_path)
    
    return data

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

def average_signal_without_weights(data: ndarray):
    # Calculate signal for each scan
    # Calculate absorption
    absorption = -np.log10(data)
    # Calculate difference signal
    signal = np.take(absorption, 1, axis=-1) - np.take(absorption, 0, axis=-1)

    # Average signals
    avg_signal = np.average(signal, axis=0)

    return avg_signal

def generate_legacy_data_format(difference_signal, delays, probe_axis):
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
    old_format[:, 2] = difference_signal.flatten()
    
    return old_format


if __name__ == "__main__":
    path = r"C:\Users\H-Lab\Desktop\20200923__000"
    # Load data set
    d, w, c, s, delays, probe_axis = load_data_set(path)
    
    # Load averaged data
    d = load_average_data(path)

    # Load delays (ignore weights)
    delays = np.load(os.path.join(path, "delay_file_20200923__000.npy"))[:, 0]

    # Load wn axis (probe axis)
    probe_axis = np.load(os.path.join(path, "probe_wn_axis_20200923__000.npy"))
    
    # Calculate spectra 
    signal = - np.log10(d)
    difference_signal = signal[:, :, 1] - signal[:, :, 0]
    old_format = generate_legacy_data_format(difference_signal, delays, probe_axis)
    np.savetxt(os.path.join(path, "old_format.txt"), old_format, delimiter="\t")
    
    # ------------------- Different data analysis methods -----------------
    # Calculate spectra using different averaging methods
    # t1_pos1 = average_transmission_with_counts(d, c)
    # t2 = average_transmission_with_weights(d, w)
    # t3 = average_signal_with_s2s(d, s)
    # t4 = average_signal_without_weights(d)

    # # Load second position
    # path = r"C:\Users\H-Lab\Documents\data_analysis\split_sample_pos000"
    # # Load data set
    # d, w, c, s = load_data_set(path)
    # # Calculate spectra using different averaging methods
    # t1_pos0 = average_transmission_with_counts(d, c)

    ## Save averaged data to text files
    # np.savetxt(os.path.join(path, "average_transmission_with_counts"), t1)
    # np.savetxt(os.path.join(path, "average_transmission_with_weights"), t2)
    # np.savetxt(os.path.join(path, "average_signal_with_s2s"), t3)
    # np.savetxt(os.path.join(path, "average_signal_without_weights"), t4)

    ## Plot signal with respect to time for each pair of pixels
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
    
    ## Plot given pixel pair (20)
    # fig, ax = plt.subplots()

    # ax.plot(t1[:, 20], label="average_transmission_with_counts", linewidth= 0.5)
    # ax.plot(t2[:, 20], label="average_transmission_with_weights", linewidth= 0.5)
    # ax.plot(t3[:, 20], label="average_signal_with_s2s", linewidth= 0.5)
    # ax.plot(t4[:, 20], label="average_signal_without_weights", linewidth= 0.5)
    
    ## Plot signal with respect to frequency for each delay
    # nrows = 4
    # ncols = 7
    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    # for delay in range(t1_pos0.shape[0]):
    #     ax_idx = np.unravel_index(delay, (nrows, ncols))
    #     # axes[ax_idx].plot(t1_pos1[delay], label="pos 1 raw", linewidth= 0.5)
    #     # axes[ax_idx].plot(t1_pos0[delay], label="pos 0 raw", linewidth= 0.5)
    #     # axes[ax_idx].plot(t1_pos1[delay]-t1_pos1[0], label="pos 1 minus 40ps", linewidth= 0.5)
    #     # axes[ax_idx].plot(t1_pos0[delay]-t1_pos0[0], label="pos 0 minus 40ps", linewidth= 0.5)
    #     axes[ax_idx].plot(t1_pos1[delay]-t1_pos0[delay], label="pos 0 minus pos 1", linewidth= 0.5)
    #     axes[ax_idx].plot((t1_pos1[delay] - t1_pos1[0]) - (t1_pos0[delay] - t1_pos0[0]), label="pos 0 minus pos 1 minus 40 ps", linewidth= 0.5)
        
    #     # axes[ax_idx].plot(t2[delay], label="average_transmission_with_weights", linewidth= 0.5)
    #     # axes[ax_idx].plot(t3[delay], label="average_signal_with_s2s", linewidth= 0.5)
    #     # axes[ax_idx].plot(t4[delay], label="average_signal_without_weights", linewidth= 0.5)

    #     axes[ax_idx].legend(fontsize=5)
    
    # plt.show()

    
    # Plot whole data set as heatmap
    # fig, ax = plt.subplots(ncols=2)
    # t1 = t1_pos1
    # positive_delays = delays > 0 
    # # small_delays = delays < 10000
    # # positive_delays = small_delays == positive_delays
    
    # print(positive_delays)
    
    # X, Y = np.meshgrid(delays[positive_delays], probe_axis)
    
    # ax[0].pcolormesh(
    #             X,
    #             Y,
    #             (t1[positive_delays, :].T),
    #             cmap="seismic")
    # ax[0].set_xscale("log")
    
    # t1_bg = t1 - t1[0]
    # ax[1].pcolormesh(
    #             X,
    #             Y,
    #             t1_bg[positive_delays, :].T,
    #             cmap="seismic")
    # ax[1].set_xscale("log")
    
    # plt.show()
    


    # # Meshgrid for 2d plots
    #     X, Y = np.meshgrid(self.delays[:,0], probe_axis)
    #     # # Single scan
    #     self.plot_ref["single time-signal heatmap"] = self.axes["single time-signal heatmap"].pcolormesh(
    #                                                     X,
    #                                                     Y,
    #                                                     signal[0].T, # ????
    #                                                     cmap = self.cmap
    #     )
