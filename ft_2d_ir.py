# %%
import os

import numpy as np
from numpy import ndarray

import data_processing_offline as dp
# from data_processing import process_ft2dir_data

from matplotlib import pyplot as plt

def load_data_set(path):
    # Get name of folder and
    # ergo filename in this folder
    filename = os.path.basename(path)
    # Load delay
    # and probe_wn_axis etc.
    # vis_delays = np.load(os.path.join(path, "vis_delay_file_" + filename +".npy")) 
    ir_delays = np.load(os.path.join(path, "delay_file_" + filename +".npy")) 
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
        
        return data, weights, counts, ir_delays, probe_axis
    
    # Get delay count
    n_delays = ir_delays.shape[0]
    print("Detected {} delays".format(n_delays))

    # Get scan count
    # For this we go into into the folder
    # where the scans are saved for the last delay
    # and take the last entry of the sorted list
    # Because we only want to detect complete
    # scans
    #! Change this to not throw away incomplete scan data later
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
    for file in t[:4]: # There should be 3 different files for each scan
        # The if statement should be superfluous
        # because all files have the same dimension
        # (I believe)
        p = os.path.join(scan_path, file) # Path to file
        if "weights" in file:
            weights_shape = np.load(p).shape
        elif "counts" in file:
            counts_shape = np.load(p).shape
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

            # Load data into array
            data[scan, delay] = np.load(os.path.join(f_path, data_name))
            weights[scan, delay] = np.load(os.path.join(f_path, weights_name))
            counts[scan, delay] = np.load(os.path.join(f_path, counts_name))

    # Save data to compact files
    new_dir = os.path.join(path, "combined_data")
    os.mkdir(new_dir)

    np.save(os.path.join(new_dir, "data"), data)
    np.save(os.path.join(new_dir, "weights"), weights)
    np.save(os.path.join(new_dir, "counts"), counts)

    # data dimensions (in this case)
    # (same for weights and counts):
    # scans
    # delays
    # pixel+1 (last entry is pyro interferogram)
    # interferometer position
    # uv/vis chopper on/off

    return data, weights, counts, ir_delays, probe_axis

# still useful, but maybe broken in this particular case
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

# deprecated / does not apply for time domain data
def average_time_domain_with_counts(data: ndarray, counts: ndarray):
    # Average data (transmission)
    avg_data = np.average(data, axis=0, weights=counts)
    # Calculate absorption
    absorption = -np.log10(avg_data)
    # Calculate difference signal
    signal = np.take(absorption, 1, axis=-1) - np.take(absorption, 0, axis=-1)

    return signal

# deprecated / does not apply for time domain data
def average_time_domain_with_weights(data: ndarray, weights: ndarray):
    # Average data
    avg_data = np.average(data, axis=0, weights=weights)
    # Calculate absorption
    absorption = -np.log10(avg_data)
    # Calculate difference signal
    signal = np.take(absorption, 1, axis=-1) - np.take(absorption, 0, axis=-1)

    return signal

def generate_legacy_data_format(difference_signal, delays, probe_axis, pump_axis):
    # Format for consumption by MATLAB CLS analysis toolbox
    # The old format has the following shape:
    # Not simple
    # See SL_lineshape_fitting_report.pdf p. 4
    
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
    
    old_format[old_format_idx.flatten(), 1:] = difference_signal.reshape(delays.size*probe_axis.size, pump_axis.size)
    
    return old_format

def generate_frequency_domain_data(
        time_domain_data,
        interferograms,
        counts,
        zero_pad_factor = 8,
        window_function = "cos_square"
        ):

    # There are two possibilities of averaging
    # the data. 
    # Variant 0: Calculate frequency domain data
    # then average scans
    # Variant 1: Average time domain data then
    # transform to frequency domain
    if len(time_domain_data.shape) == 4: # Variant 0
        delay_axis = 1
        # # Process 0th scan and 0th delay to get
        # # the frequency domain array size
        frequency_domain_data, interferogram_info = dp.process_ft2dir_data(
                        time_domain_data[0, 0, :, :],
                        interferograms[0, 0, :],
                        window_function = window_function,
                        zero_pad_factor = zero_pad_factor
                    )

        frequency_domain_data = np.zeros(
            (
            time_domain_data.shape[0], # scans
            time_domain_data.shape[1], # delays
            *frequency_domain_data.shape, # pixel, freq domain
            )
        )

        opa_pump_spectrum = np.zeros(
            (
            time_domain_data.shape[0], # scans
            time_domain_data.shape[1], # delays
            interferogram_info[2].size,
            )
        )

        opa_range = np.array([]) # This bogus when averaging interferograms
        
        pump_frequency_axis = interferogram_info[3]

        for scan in range(time_domain_data.shape[0]):
            for delay in range(time_domain_data.shape[1]):
                frequency_domain_data[scan, delay, :, :], interferogram_info  = dp.process_ft2dir_data(
                        time_domain_data[scan, delay, :, :],
                        interferograms[scan, delay],
                        window_function = window_function,
                        zero_pad_factor = zero_pad_factor
                    )
                
                opa_pump_spectrum[scan, delay, :] = np.abs(interferogram_info[2])
                if opa_range.size < interferogram_info[4][1].size:
                    opa_range = interferogram_info[4][1]

        return frequency_domain_data, opa_pump_spectrum, opa_range, pump_frequency_axis
        
    else: # Variant 1
        delay_axis = 0
        # # Process 0th delay to get
        # # the frequency domain array size
        frequency_domain_data, interferogram_info = dp.process_ft2dir_data(
                        time_domain_data[0, :, :],
                        interferograms[0],
                        window_function = window_function,
                        zero_pad_factor = zero_pad_factor
                    )

        frequency_domain_data = np.zeros(
            (
            time_domain_data.shape[0], # delays
            *frequency_domain_data.shape, # pixel, freq domain
            )
        )

        opa_pump_spectrum = np.zeros(
            (
            time_domain_data.shape[0], # delays
            interferogram_info[2].size,
            )
        )

        opa_range = np.array([]) # This bogus when averaging interferograms
        
        pump_frequency_axis = interferogram_info[3]

        for delay in range(time_domain_data.shape[0]):
            frequency_domain_data[delay, :, :], interferogram_info  = dp.process_ft2dir_data(
                    time_domain_data[delay, :, :],
                    interferograms[delay],
                    window_function = window_function,
                    zero_pad_factor = zero_pad_factor
                )
            
            opa_pump_spectrum[delay, :] = np.abs(interferogram_info[2])
            if opa_range.size < interferogram_info[4][1].size:
                opa_range = interferogram_info[4][1]

        return frequency_domain_data, opa_pump_spectrum, opa_range, pump_frequency_axis

def data_explorer(data):
    pass


# note: question is the order of processing steps:
# - fourier transform
# - averaging
# - calculation of viper signal
# we know that ft should come before averaging because
# of phase shift during long measurements
# suggestion: ft, viper, averaging

def variant0(data, counts, window_function=""):
    # ----- Variant 0
    # Calculate frequency domain data for each scan
    # and then average
    time_domain_absorption_v0 = - np.log10(data[:,:,:-1])
    interferograms = data[:,:,-1]
    v0 = generate_frequency_domain_data(time_domain_absorption_v0, interferograms, counts, window_function=window_function)
    opa_range = v0[2] #[0] #GW This fixes the dimension problem, but I don't really understand why
    pump_axis = v0[3][opa_range]
    frequency_domain_data_v0 = v0[0]

    # Average
    avg_frequency_domain_data_v0 = np.average(frequency_domain_data_v0, axis=0)

    ir_2d = avg_frequency_domain_data_v0[:, :, opa_range]
    ir_2d_txt = generate_legacy_data_format(ir_2d, ir_delays[:,0], probe_axis, pump_axis)

    np.savetxt(os.path.join(path, "ir_2d_v0_" + window_function + ".txt"), ir_2d_txt)

def variant1(data, counts, window_function=""):
    # ----- Variant 1 
    # Average time domain data over scans then calculate 
    # frequency domain data 
    scan_averaged_data = np.average(
            data,
            axis = 0,
            weights = counts
        )
    time_domain_absorption_v1 = - np.log10(scan_averaged_data[:, :-1])
    interferograms = scan_averaged_data[:, -1]
    counts_v1 = counts.sum(axis=0)

    v1 = generate_frequency_domain_data(time_domain_absorption_v1, interferograms, counts_v1)

    opa_range = v1[2] #[0] #GW This fixes the dimension problem, but I don't really understand why
    pump_axis = v1[3][opa_range]
    frequency_domain_data_v1 = v1[0]

    ir_2d = frequency_domain_data_v1[:, :, opa_range]
    ir_2d_txt = generate_legacy_data_format(ir_2d, ir_delays[:,0], probe_axis, pump_axis)
    
    np.savetxt(os.path.join(path, "ir_2d_v1_" + window_function + ".txt"), ir_2d_txt)

# %%
if __name__ == "__main__":
    path = r"C:\data\Dropbox (Wille Biophysik)\Wille Biophysik Team Folder\hendrik_sample_2dir_data\20201023_20201023_RDC_Hexan_FT2DIR_test_mA_000"
    # Load data set
    d, w, c, ir_delays, probe_axis = load_data_set(path)
    print(d.shape)
    # plot interferogram 
    from matplotlib import pyplot as plt
    scan = 0
    delay = 0
    plt.plot(d[scan, delay , -1,:]) # -1 is the interferogram!!!
    plt.show()
# %%
    apodization_functions = [
                            "",
                            "cos_square",
                            "boxcar",
                            "triang",
                            "blackman",
                            "hamming",
                            "hann",
                            "bartlett",
                            "flattop",
                            "parzen",
                            "bohman",
                            "blackmanharris",
                            "nuttall",
                            "barthann"]

    # cheating to curb the output
    # apodization_functions = ["cos_square"]

    for apo_func in apodization_functions:
        variant0(d, c, apo_func)
        variant1(d, c, apo_func)