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

    # data dimensions (in this case)
    # (same for weights and counts):
    # scans
    # delays
    # pixel+1 (last entry is pyro interferogram)
    # interferometer position
    # uv/vis chopper on/off

    return data, weights, counts, vis_delays, ir_delays, probe_axis

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
                            time_domain_data[0, 0, :, :, 0],
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

            opa_range = [np.array([]), np.array([])] # This bogus when averaging interferograms
            
            pump_frequency_axis = interferogram_info[3]

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
                                time_domain_data[scan, delay, :, :, chopper_state],
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
                            time_domain_data[0, :, :, 0],
                            interferograms[0,:,0],
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
            
            pump_frequency_axis = interferogram_info[3]
            
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
                            time_domain_data[delay, :, :, chopper_state],
                            interferograms[delay],
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
        else:
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

def variant0_possibility0(data, counts, window_function=""):
    # # # ----- Variant 0
    # # # Calculate frequency domain data for each scan
    # # # and calculate VIPER in the frequency domain
    time_domain_absorption_v0 = - np.log10(data[:,:,:-1])
    interferograms = data[:,:,-1]
    v0 = generate_frequency_domain_data(time_domain_absorption_v0, interferograms, counts, window_function=window_function)
    opa_range = v0[2][0]
    pump_axis = v0[3][opa_range]
    frequency_domain_data_v0 = v0[0]

    # Two possibilities - calculate VIPER for each scan then average 
    # or other way around
    # Possibility 0
    # Average then calculate VIPER
    avg_frequency_domain_data_v0 = np.average(frequency_domain_data_v0, axis=0)

    ir_2d = avg_frequency_domain_data_v0[:, :, opa_range, 0]
    ir_2d_txt = generate_legacy_data_format(ir_2d, ir_delays[:,0], probe_axis, pump_axis)
    
    ir_2d_plus_viper = avg_frequency_domain_data_v0[:, :, opa_range, 1]
    ir_2d_plus_viper_txt = generate_legacy_data_format(ir_2d_plus_viper, ir_delays[:, 0], probe_axis, pump_axis)
    
    viper_v0_p0 = avg_frequency_domain_data_v0[:, :,opa_range, 1] - avg_frequency_domain_data_v0[:, :, opa_range, 0]
    viper_v0_p0_txt = generate_legacy_data_format(viper_v0_p0, ir_delays[:, 0], probe_axis, pump_axis)
    
    np.savetxt(os.path.join(path, "ir_2d_v0_p0_" + window_function + "_.txt"), ir_2d_txt)
    np.savetxt(os.path.join(path,"ir_2d_plus_viper_v0_p0_" + window_function + "_.txt"), ir_2d_plus_viper_txt)
    np.savetxt(os.path.join(path,"viper_v0_p0_" + window_function + "_.txt"), viper_v0_p0_txt)

def variant0_possibility1(data, counts, window_function=""):
    # # # ----- Variant 0
    # # # Calculate frequency domain data for each scan
    # # # and calculate VIPER in the frequency domain
    time_domain_absorption_v0 = - np.log10(data[:,:,:-1])
    interferograms = data[:,:,-1]
    v0 = generate_frequency_domain_data(time_domain_absorption_v0, interferograms, counts, window_function=window_function)
    opa_range = v0[2][0]
    pump_axis = v0[3][opa_range]
    frequency_domain_data_v0 = v0[0]

    # Two possibilities - calculate VIPER for each scan then average 
    # or other way around
    # #Possibility 1
    # # Calculate VIPER then average
    viper_v0_p1 = frequency_domain_data_v0.take(1, axis=-1) - frequency_domain_data_v0.take(0, axis=-1)
    viper_v0_p1_avg = np.average(viper_v0_p1, axis=0)

    # Average together freq domain data to get 2D-IR and 2D-IR + VIPER
    avg_frequency_domain_data_v0 = np.average(frequency_domain_data_v0, axis=0)
    
    ir_2d = avg_frequency_domain_data_v0[:, :, opa_range, 0]
    ir_2d_txt = generate_legacy_data_format(ir_2d, ir_delays[:,0], probe_axis, pump_axis)
    
    ir_2d_plus_viper = avg_frequency_domain_data_v0[:, :, opa_range, 1]
    ir_2d_plus_viper_txt = generate_legacy_data_format(ir_2d_plus_viper, ir_delays[:, 0], probe_axis, pump_axis)
    
    viper_v0_p1_txt = generate_legacy_data_format(viper_v0_p1_avg[:, :,opa_range], ir_delays[:, 0], probe_axis, pump_axis)
    
    np.savetxt(os.path.join(path, "ir_2d_v0_p1_" + window_function + "_.txt"), ir_2d_txt)
    np.savetxt(os.path.join(path,"ir_2d_plus_viper_v0_p1_" + window_function + "_.txt"), ir_2d_plus_viper_txt)
    np.savetxt(os.path.join(path,"viper_v0_p1_" + window_function + "_.txt"), viper_v0_p1_txt)

def variant1(data, counts, window_function=""):
    # ----- Variant 1 
    # Average time domain data over scans then calculate 
    # frequency domain data 
    # and calculate VIPER in the frequency domain
    scan_averaged_data = np.average(
            data,
            axis = 0,
            weights = counts
        )
    time_domain_absorption_v1 = - np.log10(scan_averaged_data[:, :-1])
    interferograms = scan_averaged_data[:, -1]
    counts_v1 = counts.sum(axis=0)

    v1 = generate_frequency_domain_data(time_domain_absorption_v1, interferograms, counts_v1)

    opa_range = v1[2][0]
    pump_axis = v1[3][opa_range]
    frequency_domain_data_v1 = v1[0]

    ir_2d = frequency_domain_data_v1[:, :, opa_range, 0]
    ir_2d_txt = generate_legacy_data_format(ir_2d, ir_delays[:,0], probe_axis, pump_axis)
    
    ir_2d_plus_viper = frequency_domain_data_v1[:, :, opa_range, 1]
    ir_2d_plus_viper_txt = generate_legacy_data_format(ir_2d_plus_viper, ir_delays[:, 0], probe_axis, pump_axis)
    
    viper_v1 = frequency_domain_data_v1[:, :, opa_range, 1] - frequency_domain_data_v1[:, :, opa_range, 0]
    viper_v1_txt = generate_legacy_data_format(viper_v1, ir_delays[:, 0], probe_axis, pump_axis)
    
    np.savetxt(os.path.join(path, "ir_2d_v1" + window_function + "_.txt"), ir_2d_txt)
    np.savetxt(os.path.join(path,"ir_2d_plus_viper_v1" + window_function + "_.txt"), ir_2d_plus_viper_txt)
    np.savetxt(os.path.join(path,"viper_v1" + window_function + "_.txt"), viper_v1_txt)

def variant2(data, counts, window_function=""):
    # ----- Variant 2
    # Calculate VIPER in time domain 
    # then fourier transform then average
    time_domain_absorption = - np.log10(data[:,:,:-1])
    time_domain_viper = time_domain_absorption.take(1, axis=-1) - time_domain_absorption.take(0, axis=-1)
    
    # Average interferogram
    interferograms = np.average(d[:,:,-1], weights=counts[:,:,-1], axis=-1)
    counts_v2 = counts.sum(axis=-1)
    v2 = generate_frequency_domain_data(time_domain_viper, interferograms, counts_v2)

    opa_range = v2[2]
    pump_axis = v2[3][opa_range]
    viper_v2 = v2[0]

    # Average frequency domain scans
    viper_v2_avg = np.average(viper_v2, axis=0)

    # Generate legacy data format (this time it is not possible to generate
    # 2D-IR and 2D-IR + VIPER)
    viper_v2_txt = generate_legacy_data_format(viper_v2_avg[:, :,opa_range], ir_delays[:, 0], probe_axis, pump_axis)
    np.savetxt(os.path.join(path,"viper_v2_" + window_function + "_.txt"), viper_v2_txt)

def variant3_possibility0(data, counts, window_function=""):
    # ------ Variant 3
    # --- Possibility 0
    # Calculate VIPER signal for each scan then average
    # in time domain then fourier transform
    
    time_domain_absorption = - np.log10(d[:,:,:-1])
    time_domain_viper = time_domain_absorption.take(1, axis=-1) - time_domain_absorption.take(0, axis=-1)
    interferograms = np.average(data[:,:,-1], axis=-1, weights=counts[:,:,-1])
    c_v3 = counts.sum(axis=-1)
    time_domain_viper_avg = np.average(time_domain_viper, axis=0, weights=c_v3[:,:,:-1])
    interferograms_avg = np.average(interferograms, axis=0, weights=c_v3[:,:,-1])
    counts_v3 = c_v3.sum(axis=0)
    v3_p0 = generate_frequency_domain_data(time_domain_viper_avg, interferograms_avg, counts_v3)

    opa_range = v3_p0[2]
    pump_axis = v3_p0[3][opa_range]
    viper_v3_p0 = v3_p0[0]

    # Generate legacy data format (this time it is not possible to generate
    # 2D-IR and 2D-IR + VIPER)
    viper_v3_p0_txt = generate_legacy_data_format(viper_v3_p0[:, :, opa_range], ir_delays[:, 0], probe_axis, pump_axis)
    np.savetxt(os.path.join(path,"viper_v3_p0_" + window_function + "_.txt"), viper_v3_p0_txt)

def variant3_possibility1(data, counts, window_function=""):
    # ------ Variant 3
    # --- Possibility 1
    # Average scans then calculate VIPER
    # in time domain then perform fourier transform
    time_domain_avg = np.average(data, axis=0, weights=counts)
    time_domain_absorption = - np.log10(time_domain_avg[:,:,:-1])
    time_domain_viper = time_domain_absorption.take(1, axis=-1) - time_domain_absorption.take(0, axis=-1)
    counts_scan_avg = counts.sum(axis=0)
    print(data.shape[1])
    print(counts_scan_avg.shape[1])
    interferograms = np.average(time_domain_avg[:,-1], axis=-1, weights=counts_scan_avg[:,-1])
    counts_v3_p1 = counts_scan_avg.sum(axis=-1)
    v3_p1 = generate_frequency_domain_data(time_domain_viper[:,:-1], interferograms, counts_v3_p1)

    opa_range = v3_p1[2]
    pump_axis = v3_p1[3][opa_range]
    viper_v3_p1 = v3_p1[0]

    # Generate legacy data format (this time it is not possible to generate
    # 2D-IR and 2D-IR + VIPER)
    viper_v3_p1_txt = generate_legacy_data_format(viper_v3_p1[:, :, opa_range], ir_delays[:, 0], probe_axis, pump_axis)
    np.savetxt(os.path.join(path,"viper_v3_p1_" + window_function + "_.txt"), viper_v3_p1_txt)


# %%
if __name__ == "__main__":
    path = r"C:\Users\H-Lab\Documents\data_analysis\20201016_ITx_FTVIPER_186_000"
    # Load data set
    d, w, c, ir_delays, vis_delays, probe_axis = load_data_set(path)
    print(d.shape)
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

    for apo_func in apodization_functions:
        variant0_possibility0(d, c, apo_func)
        variant0_possibility1(d, c, apo_func)
        variant1(d,c, apo_func)
        variant2(d, c, apo_func)
        variant3_possibility0(d, c, apo_func)
        variant3_possibility1(d, c, apo_func)