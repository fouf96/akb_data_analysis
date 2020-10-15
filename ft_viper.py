import os

import numpy as np
from numpy import ndarray

import data_processing as dp

from matplotlib import pyplot as plt

def load_data_set(path):
    # Get name of folder and
    # ergo filename in this folder
    filename = os.path.basename(path)
    # Load delay
    # and probe_wn_axis etc.
    vis_delays = np.load(os.path.join(path, "vis_delay_file_" + filename +".npy")) 
    ir_delays = np.load(os.path.join(path, "ir_delay_file_" + filename +".npy")) 
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
        
        return data, weights, counts, vis_delays, ir_delays, probe_axis
    
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

    return data, weights, counts, vis_delays, ir_delays, probe_axis

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

def average_time_domain_with_counts(data: ndarray, counts: ndarray):
    # Average data (transmission)
    avg_data = np.average(data, axis=0, weights=counts)
    # Calculate absorption
    absorption = -np.log10(avg_data)
    # Calculate difference signal
    signal = np.take(absorption, 1, axis=-1) - np.take(absorption, 0, axis=-1)

    return signal

def average_time_domain_with_weights(data: ndarray, weights: ndarray):
    # Average data
    avg_data = np.average(data, axis=0, weights=weights)
    # Calculate absorption
    absorption = -np.log10(avg_data)
    # Calculate difference signal
    signal = np.take(absorption, 1, axis=-1) - np.take(absorption, 0, axis=-1)

    return signal

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

def generate_frequency_domain_data(
        time_domain_data,
        interferograms,
        counts,
        zero_pad_factor = 8,
        window_function = "cos_square"
        ):

    # We cannot preallocate an array
    # that will hold the frequency domain data
    # because we do not know the length that
    # will result from cutting of at the zerobin
    # and the zeropadding
    # for delay in 
    # Check if last axis is chopper or
    # interferometer states
    # ------------------------------------------------
    if time_domain_data.shape[-1] == 2:
        chopper_states = time_domain_data.shape[-1]
        if len(time_domain_data.shape) == 5:
            # # Process 0th scan and 0th delay to get
            # # the frequency domain array size
            frequency_domain_data, interferogram_info = dp.process_ft2dir_data(
                            time_domain_data[0, 0, :-1, :, 0],
                            interferograms[0, 0, :, 0],
                            window_function = window_function,
                            zero_pad_factor = zero_pad_factor
                        )

            frequency_domain_data = np.zeros(
                (
                time_domain_data.shape[0], # scans
                time_domain_data.shape[1], # delays
                *frequency_domain_data.shape, # pixel, freq domain
                chopper_states # chopper states
                )
            )

            opa_pump_spectrum = np.zeros(
                (
                time_domain_data.shape[0], # scans
                time_domain_data.shape[1], # delays
                interferogram_info[2].size,
                chopper_states # chopper states
                )
            )

            opa_range = [np.array([]), np.array([])]     
            
            pump_frequency_axis = interferogram_info[2]

            # Average interferogram over chopper states
            interferograms = np.average(
                interferograms,
                axis = -1,
                weights = counts[:,:, -1]
            )

            for scan in range(time_domain_data.shape[0]):
                for delay in range(time_domain_data.shape[1]):
                    for chopper_state in range(chopper_states):
                        frequency_domain_data[scan, delay, :, :, chopper_state], interferogram_info  = dp.process_ft2dir_data(
                                time_domain_data[scan, delay, :-1, :, chopper_state],
                                interferograms[scan, delay],
                                window_function = window_function,
                                zero_pad_factor = zero_pad_factor
                            )
                        
                        opa_pump_spectrum[scan, delay, :, chopper_state] = np.abs(interferogram_info[2])
                        if opa_range[chopper_state].size < interferogram_info[4][1].size:
                            opa_range[chopper_state] = interferogram_info[4][1]

            return frequency_domain_data, opa_pump_spectrum, opa_range, pump_frequency_axis

    # ------------------------------------------------
        else:
            # # Process 0th scan and 0th delay to get
            # # the frequency domain array size
            frequency_domain_data, interferogram_info = dp.process_ft2dir_data(
                            time_domain_data[0, :-1, :, 0],
                            time_domain_data[0, -1, :, 0],
                            window_function = window_function,
                            zero_pad_factor = zero_pad_factor
                        )

            frequency_domain_data = np.zeros(
                (
                time_domain_data.shape[0], # delay
                *frequency_domain_data.shape, # pixel, freq domain
                chopper_states # chopper states
                )
            )

            opa_pump_spectrum = np.zeros(
                (
                time_domain_data.shape[0], # delay
                interferogram_info[2].size,
                chopper_states # chopper states
                )
            )

            opa_range = [np.array([]), np.array([])]     
            
            pump_frequency_axis = interferogram_info[2]
            
            # Average interferogram over chopper states
            print(interferograms.shape)
            print(counts.shape)
            interferograms = np.average(
                interferograms,
                axis = -1,
                weights = counts[:, -1]
            )

            for delay in range(time_domain_data.shape[0]):
                for chopper_state in range(chopper_states):
                    frequency_domain_data[delay, :, :, chopper_state], interferogram_info  = dp.process_ft2dir_data(
                            time_domain_data[delay, :-1, :, chopper_state],
                            time_domain_data[delay, -1, :, chopper_state],
                            window_function = window_function,
                            zero_pad_factor = zero_pad_factor
                        )
                    
                    opa_pump_spectrum[delay, :, chopper_state] = np.abs(interferogram_info[2])
                    if opa_range[chopper_state].size < interferogram_info[4][1].size:
                        opa_range[chopper_state] = interferogram_info[4][1]
                    
            return frequency_domain_data, opa_pump_spectrum, opa_range, pump_frequency_axis
        
            
    else:
        if len(time_domain_data.shape) == 4:
            delay_axis = 1
        else:
            delay_axis = 0


    pass

def data_explorer(data):
    pass

if __name__ == "__main__":
    path = r"C:\Users\H-Lab\Documents\data_analysis\20201013_VIPA_ITX_000"
    # Load data set
    d, w, c, ir_delays, vis_delays, probe_axis = load_data_set(path)

    # # ----- Variant 0
    # # Calculate frequency domain data for each scan
    # # and calculate VIPER in the frequency domain
    # time_domain_absorption_v0 = - np.log10(d[:,:,:-1])
    # interferograms = d[:,:,-1]
    # v0 = generate_frequency_domain_data(time_domain_absorption_v0, d[:,:,-1], c)
    # frequency_domain_data_v0 = v0[0]
    # viper_v0 = frequency_domain_data_v0.take(1, axis=-1) - frequency_domain_data_v0.take(0, axis=-1)
    # viper_v0_avg = np.average(viper_v0, axis=0)

    # ----- Variant 1 
    # Average time domain data over scans then calculate 
    # frequency domain data 
    # and calculate VIPER in the frequency domain
    scan_averaged_data = np.average(
            d,
            axis = 0,
            weights = c
        )
    time_domain_absorption_v1 = - np.log10(scan_averaged_data[:, :-1])
    interferograms = scan_averaged_data[:, -1]
    counts_v1 = c.sum(axis=0)

    v1 = generate_frequency_domain_data(time_domain_absorption_v1, interferograms, counts_v1)

