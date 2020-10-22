import numpy as np
from numpy import ndarray

from matplotlib import pyplot as plt

import data_processing as dp

from save_data import SaveData
import os

save_path = r"C:\Users\H-Lab\Documents\data_analysis"
save_name = "20200916_ITx_FPVIPER_183_001_raw_data_eval"

raw_data_path = r"C:\Users\H-Lab\Documents\data_analysis\20200916_ITx_FPVIPER_183_001"

n_pump_pixels = 16
n_delay = 18
n_scans = 12
n_interleaves = 16

samples_to_acquire = 500
n_channels = 66

pixel_idx = np.arange(64)
probe_pixel_idx = np.arange(32)
ref_pixel_idx = np.arange(32, 64)

ir_chopper_idx = 64
ir_chopper_voltage_level = [0.04 / 2]

vis_chopper_idx = 65
vis_chopper_voltage_level = [5 / 2]

saver = SaveData(save_path, save_name, "Grüner Powerranger", np.arange(n_delay), pump_pixels=np.arange(n_pump_pixels))

raw_data_shape = (n_channels, samples_to_acquire)
# Preallocating
d = np.zeros((n_interleaves, probe_pixel_idx.size, 2, 2))
w = np.zeros(d.shape)
c = np.zeros(d.shape)
s2s = np.zeros((n_interleaves, probe_pixel_idx.size))

filename = os.path.basename(raw_data_path)
background = np.load(os.path.join(raw_data_path, "background_"+filename+".npy"))
for pump_pixel in range(n_pump_pixels):
    print("pump idx", pump_pixel)
    pump_folder = "pump_pixel{}".format(str(pump_pixel).zfill(3))
    pump_str = "p{}_".format(str(pump_pixel).zfill(3))
    for delay in range(n_delay):
        print("delay idx", delay)
        delay_folder =  "delay{}".format(str(delay).zfill(3))
        delay_str = "d{}_".format(str(delay).zfill(3))
        for scan in range(n_scans):
            scan_str = "s{}_".format(str(scan).zfill(6))
            for interleave in range(n_interleaves):
                interleave_str = "intlv{}_".format(str(interleave).zfill(3))
                f_path = os.path.join(raw_data_path, "raw_data", pump_folder, delay_folder)

                file_name = scan_str + pump_str + delay_str + interleave_str + filename + "_raw.npy"
                raw_data = np.load(os.path.join(f_path, file_name))

                background_corrected_data = raw_data[pixel_idx] - background[pixel_idx, np.newaxis]
                
                #* Do not linearise for our purposes
                intensities = background_corrected_data

                # Calculate transmission/ relative intensity
                # (probe intensity / reference intensity)
                transmission = intensities[probe_pixel_idx] / intensities[ref_pixel_idx]

                # Get the corresponding chopper state for each shot
                # -IR Chopper-
                ir_chopper_states = np.digitize(raw_data[ir_chopper_idx], ir_chopper_voltage_level)
                # -UV/VIS Chopper-
                vis_chopper_states = np.digitize(raw_data[vis_chopper_idx], vis_chopper_voltage_level)

                # Stack chopper states into one array
                # for sort data function 
                # This implicitly decides upon the 
                # order of dimensionality of the data array:
                # The last axis of data is the vis chopper
                # distinction while the second to 
                # last axis corresponds to the
                # ir chopper
                # (Always make sure that the n_possible_states
                # array lists its values in the same order!)
                states = np.vstack((ir_chopper_states, vis_chopper_states))

                d[interleave], w[interleave], c[interleave], statistics = dp.sort_data(
                                                                    transmission,
                                                                    states,
                                                                    np.array((2,2))
                                                                    )

                _, _, s2s[interleave], _ = dp.shot_to_shot_viper(
                                                            transmission,
                                                            states[1, 0], # 0th sample of UV/VIS Chopper
                                                            states[0, :2] # 0th and 1st sample of IR Chopper
                                                            )

                if interleave == n_interleaves - 1:
                    saver.save_scan(d, scan, delay_idx=delay, pump_idx=pump_pixel)
                    saver.save_counts(c, scan, delay_idx=delay, pump_idx=pump_pixel)
                    saver.save_weights(w, scan, delay_idx=delay, pump_idx=pump_pixel)
                    #*-------------
                    saver.save_s2s_std(s2s, scan, delay_idx=delay, pump_idx=pump_pixel)
                    #*-------------

                