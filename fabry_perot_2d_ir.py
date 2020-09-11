#%%
import os

import numpy as np
from numpy import ndarray

from matplotlib import pyplot as plt

def load_data_set(path):
    # Get name of folder and
    # ergo filename in this folder
    filename = os.path.basename(path)

    # Load delays, pump pixels
    # and probe_wn_axis etc.
    pump_pixels = np.load(os.path.join(path, "pump_pixels_" + filename +".npy"))
    delays = np.load(os.path.join(path, "delay_file_" + filename +".npy")) 
    probe_axis = np.load(os.path.join(path, "probe_wn_axis_" + filename +".npy"))

    # Check whether combined data set
    # was already created before hand
    # in that case in can just be
    # loaded
    if "combined_data" in os.listdir(path):
        print("Combined data already exists loading directly.")

        f_path = os.path.join(path, "combined_data")
        
        data = np.load(os.path.join(f_path, "data_" + filename + ".npy"))
        weights = np.load(os.path.join(f_path, "weights_" + filename + ".npy"))
        counts = np.load(os.path.join(f_path, "counts_" + filename + ".npy"))
        s2s_std = np.load(os.path.join(f_path, "s2s_std_" + filename + ".npy"))
        pump_spectrum = np.load(os.path.join(f_path, "pump_spectrum_" + filename + ".npy"))

        return data, weights, counts, s2s_std, pump_spectrum, pump_pixels, delays, probe_axis

    # Get delay count
    # Do this by using the delay file
    n_delays = delays.shape[0]
    print("Detected {} delays".format(n_delays))
    
    # Get pump pixel count
    # Do this by using the pump pixels file
    n_pump_pixels = pump_pixels.shape[0]
    print("Detected {} pump pixel".format(n_pump_pixels))

    # Get scan count
    # For this we go into into the folder
    # where the scans are saved for the last pump pixel
    # and the last delay
    # and take the last entry of the sorted list
    # Because we only want to detect complete
    # scans
    #! Change this to not throw away data later
    scan_path = os.path.join(
                            path,
                            "scans",
                            "pump_pixel{}".format(str(n_pump_pixels-1).zfill(3)),
                            "delay{}".format(str(n_delays-1).zfill(3))
                            )
    t = os.listdir(scan_path) 
    t.sort()
    n_scans = int(t[-1][1:7]) + 1 # Not safe 
    print("Detected {} scans".format(n_scans))

    # # Get the size of one data array
    for file in t[:4]: # There should be 4 different files for each scan
        # The if statement should be superfluous
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
    
    # For 2D- IR with fabry perot we also 
    # have a pump spectrum for each scan
    # for each pump pixel
    # Get the size of this array too
    pump_path = os.path.join(path, "scans", "pump_pixel000")
    for file in os.listdir(pump_path):
        if "pump_spectrum" in file:
            pump_spectrum_shape = np.load(os.path.join(pump_path, file)).shape
            break

    # Preallocate arrays
    # Always set dimension such
    # that they are identical to the ones
    # used in measurement software
    # Go into module for experiment and
    # search for "self.data = np.zeros("
    # to see the correct format
    # The axis holding the scans
    # should always be axis=0!
    # Transmission data of scans for each delay
    # and each pump pixel
    data = np.zeros((n_scans, n_pump_pixels, n_delays, *data_shape))
    # Inverse variance of transmission data for
    # each scan for each delay
    weights = np.zeros((n_scans, n_pump_pixels, n_delays, *weights_shape))
    # Counts of each state of each scan for each delay
    counts = np.zeros((n_scans, n_pump_pixels, n_delays, *counts_shape))
    #? Standard deviation of shot to shot signal
    #? of each scan for each delay
    s2s_std = np.zeros((n_scans, n_pump_pixels, n_delays, *s2s_std_shape))
    # Pump spectrum for each pump pixel and scan
    pump_spectrum = np.zeros((n_scans, n_pump_pixels, *pump_spectrum_shape))

    # Load scan data
    # There is probably a more efficient 
    # way of doing this using glob or os.walk
    for pump_pixel in range(n_pump_pixels):
        pump_folder = "pump_pixel{}".format(str(pump_pixel).zfill(3))
        pump_str = "p{}_".format(str(pump_pixel).zfill(3))
        for delay in range(n_delays):
            delay_folder =  "delay{}".format(str(delay).zfill(3))
            delay_str = "d{}_".format(str(delay).zfill(3))
            
            for scan in range(n_scans):
                scan_str = "s{}_".format(str(scan).zfill(6))
                
                # We only need to load pump spectrum once for all delays
                if delay == 0:
                    p_path = os.path.join(path, "scans", pump_folder)
                    p_name = scan_str + pump_str + "pump_spectrum_" + filename + ".npy" 
                    pump_spectrum[scan, pump_pixel] = np.load(os.path.join(p_path, p_name))
                
                # Generate file paths
                f_path = os.path.join(path, "scans", pump_folder, delay_folder)
                
                data_name = scan_str + pump_str + delay_str + filename + ".npy"
                weights_name = scan_str + pump_str + delay_str + "weights_" + filename + ".npy"
                counts_name = scan_str + pump_str + delay_str + "counts_" + filename + ".npy"
                s2s_std_name = scan_str + pump_str + delay_str + "s2s_std_" + filename + ".npy"

                # Load data into arrays
                data[scan, pump_pixel, delay] = np.load(os.path.join(f_path, data_name))
                weights[scan, pump_pixel, delay] = np.load(os.path.join(f_path, weights_name))
                counts[scan, pump_pixel, delay] = np.load(os.path.join(f_path, counts_name))
                s2s_std[scan, pump_pixel, delay] = np.load(os.path.join(f_path, s2s_std_name))

    # Save data to compact files
    new_dir = os.path.join(path, "combined_data")
    os.mkdir(new_dir)

    np.save(os.path.join(new_dir, "data_" + filename), data)
    np.save(os.path.join(new_dir, "weights_" + filename), weights)
    np.save(os.path.join(new_dir, "counts_" + filename), counts)
    np.save(os.path.join(new_dir, "s2s_std_" + filename), s2s_std)
    np.save(os.path.join(new_dir, "pump_spectrum_" + filename), pump_spectrum)

    return data, weights, counts, s2s_std, pump_spectrum, pump_pixels, delays, probe_axis

def load_average_data(path):
    #! Needs work!
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
    # Average interleaves (in transmission space!!!)
    scatter_free_transmission = np.average(data, axis=3) #?
    # Sum counts on interleave axis
    counts = counts.sum(axis=3)
    # Average scans
    avg_data = np.average(scatter_free_transmission, axis=0, weights=counts)
    # Calculate absorption
    absorption = -np.log10(avg_data)
    # Calculate difference signal
    signal = np.take(absorption, 1, axis=-1) - np.take(absorption, 0, axis=-1)

    return signal

def average_transmission_with_weights(data: ndarray, weights: ndarray):
    #! Needs work!
    # Average data
    avg_data = np.average(data, axis=0, weights=weights)
    # Calculate absorption
    absorption = -np.log10(avg_data)
    # Calculate difference signal
    signal = np.take(absorption, 1, axis=-1) - np.take(absorption, 0, axis=-1)

    return signal
     
def average_signal_with_s2s(data: ndarray, s2s_std: ndarray):
    #! Needs work!
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
    #! Needs work!
    # Calculate signal for each scan
    # Calculate absorption
    absorption = -np.log10(data)

    # Calculate difference signal
    signal = np.take(absorption, 1, axis=-1) - np.take(absorption, 0, axis=-1)
    # Average interleaves
    scatter_free_signal = np.average(signal, axis=3)
    # Average signals
    avg_signal = np.average(scatter_free_signal, axis=0)

    return avg_signal

def generate_legacy_data_format(difference_signal, delays, probe_axis, pump_pixels):
    # The old format has the following shape:
    # Not simple
    # See SL_lineshape_fitting_report.pdf p. 4

    # Generate pump axis from pump pixels
    pump_axis = probe_axis[pump_pixels]
    
    # For each delay they have probe_axis + 1 rows
    # They have pump_axis + 1 columns
    old_format = np.zeros((delays.size*(probe_axis.size+1), pump_axis.size + 1))
    probe_axis_with_space = np.append(probe_axis, 0)
   
    pump_axis_inds = np.arange(probe_axis.size, delays.size*(probe_axis.size+1), probe_axis.size+1)

    # Add probe axis in col 0
    old_format[:, 0] = np.tile(probe_axis_with_space, delays.size)
    # Add pump axis in proper row and columns
    old_format[pump_axis_inds, 1:] = pump_axis
    # Add delay entries in same row as pump axis
    old_format[pump_axis_inds, 0] = delays

    old_format_idx = np.tile(np.arange(probe_axis.size), delays.size).reshape(delays.size, probe_axis.size) + np.arange(0, delays.size*probe_axis.size, probe_axis.size+1)[:, np.newaxis]
    d = np.swapaxes(difference_signal, 0, 1)
    d = np.swapaxes(d, 1, 2)
    old_format[old_format_idx.flatten(), 1:] = d.reshape(delays.size*probe_axis.size, pump_axis.size)
    
    return old_format

#%%
if __name__ == "__main__":
    path = r"C:\Users\H-Lab\Documents\data_analysis\20200910_2ITx_2D-IR_test_180_002"
    
    # Load data set
    d, w, c, s, pump_spec, pump_pixels, delays, probe_axis = load_data_set(path)
    print(d.shape)
    # ------------------- Different data analysis methods -----------------
    # Calculate spectra using different averaging methods
    t1 = average_transmission_with_counts(d, c)
    t2 = average_transmission_with_weights(d, w)
    t3 = average_signal_with_s2s(d, s)
    t4 = average_signal_without_weights(d)
#%%
    # ------- Convert to old format
    print(t1.shape)
    print(t4.shape)
    old_format = generate_legacy_data_format(t1, delays[:,0], probe_axis, pump_pixels[:,0].astype(int))
    print(np.array_equal(t1[:, 2, :].T, old_format[66:98, 1:]))
    print(t1[:, 2, :].T.shape, old_format[33:65, 1:].shape)
    np.savetxt(os.path.join(path, "old_format.txt"), old_format, delimiter="\t")
    
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
    
    # # Plot signal with respect to frequency for each delay
    # nrows = 3
    # ncols = 6
    # pump_pixel_idx = 16
    # print(pump_pixels)
    # print(pump_pixels[pump_pixel_idx, 0])
    # # plt.plot(pump_spec[pump_pixel_idx,]) 
    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    # for delay in range(t1.shape[1]):
    #     ax_idx = np.unravel_index(delay, (nrows, ncols))        
    #     # axes[ax_idx].plot(t1[pump_pixel_idx, delay], label="average_transmission_with_counts", linewidth= 1)
    #     # axes[ax_idx].plot(t4[pump_pixel_idx, delay], label="average_signal_without_weights", linewidth= 1)
    #     axes[ax_idx].plot(t1[pump_pixel_idx, delay]-t4[pump_pixel_idx, delay], label="phase cycling: transmission vs signal", linewidth= 0.5)

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

# %%
