import os
import sys
from pathlib import Path
import glob

import numpy as np
from numpy import ndarray

from datetime import date, datetime

import logging

# Set up logger
logger = logging.getLogger(__name__)

class SaveData():
    """
    SaveData class is used for creating directories
    and saving measurement data, figures etc. which 
    result from the experiments

    SaveData class is used to make it easier to save
    data from the given experiments programs. It can
    be passed as an object through processes and used
    to save certain data which is generated during the 
    measurements.

    Args:
        path (str): Path to where the data should be saved.
        file_name (str): Name of the file.
        username (str): Name of user who is measuring data.
        delays (ndarray, optional): Delays from the delay file.
            shape: 1D. Defaults to None, when there are no delays.
        pump_pixels (ndarray, optional): Pump pixels from the pump pixels file.
            shape: 1D. Defaults to None, when there are no pump pixels.
        raw_data (bool, optional): Bool value which specifies whether raw data
            from adc should be saved. Defaults to False. 
    """
    def __init__(self,
                path: str,
                file_name: str,
                username: str,
                delays: ndarray=None,
                pump_pixels: ndarray=None,
                raw_data: bool=False
                ):

        self.today = date.today()
        self.raw_data = raw_data
        self.path = path
        self.__dir_count = 0
        self.delays = delays
        self.pump_pixels = pump_pixels
        # Prepend date and append count
        self.file_name = self.today.strftime("%Y%m%d_") + file_name + "_" + str(self.__dir_count).zfill(3)
        self.username = username
        # Create total path
        self.main_path = os.path.join(self.path, "experimental_data", self.username, self.file_name)
        
        # Create directories 
        logger.info("Creating directories")
        self.create_initial_path()

        # Create a folder to save the figures in
        self.fig_path = os.path.join(self.main_path,"figures")
        Path(self.fig_path).mkdir(parents=True, exist_ok=True)

        # Create a folder to save the scans. Beforehand this was called
        # "temp files". The convention is changed now.
        self.scan_path = os.path.join(self.main_path,"scans")
        Path(self.scan_path).mkdir(parents=True, exist_ok=True)

        # Create a folder to save the averaged data.
        self.avg_path = os.path.join(self.main_path,"averaged_data")
        Path(self.avg_path).mkdir(parents=True, exist_ok=True)

        # If pump pixels were specified in pump pixel file
        # sub-folder-structure has to be created
        if type(self.pump_pixels) == ndarray:
            self.create_pump_dirs(self.scan_path)
            # We assume that when we have pump pixels
            # we also have delay dirs
            for pump_dir in os.listdir(self.scan_path):
                pump_dir = os.path.join(self.scan_path, pump_dir)
                self.create_delay_dirs(pump_dir)
        else:
            # If any delays are defined within a delay file. A
            # sub-folder-structure has to be created.
            if type(self.delays) == ndarray:
                self.create_delay_dirs(self.scan_path)
            
        # If rawdata should be saved create a folder for raw data
        if self.raw_data:
            self.raw_data_path = os.path.join(self.main_path,"raw_data")
            Path(self.raw_data_path).mkdir(parents=True, exist_ok=True)
            if type(self.pump_pixels) == ndarray:
                self.create_pump_dirs(self.raw_data_path)
                # We assume that when we have pump pixels
                # we also have delay dirs
                for pump_dir in os.listdir(self.raw_data_path):
                    pump_dir = os.path.join(self.raw_data_path, pump_dir)
                    self.create_delay_dirs(pump_dir)

            elif type(self.delays) == ndarray:
                self.create_delay_dirs(self.raw_data_path)
        
    def create_initial_path(self):
        """
        Directories in specified path are created. If the directory already
        exists, a new directory with the same name but appended numbers will be
        created.
        """
        if os.path.isdir(self.main_path):
            self.__dir_count += 1
            self.main_path = self.main_path[:-3] + str(self.__dir_count).zfill(3)
            self.file_name = self.file_name[:-3] + str(self.__dir_count).zfill(3)
            self.create_initial_path()
        else:
            Path(self.main_path).mkdir(parents=True, exist_ok=True)
            logger.info("Created directory {}".format(self.main_path))

    def create_delay_dirs(self, path):
        # Since we collect data at different delays we
        # create a folder for each delay in the delay file seperately
        for delay in range(self.delays.size):
            delay_path = os.path.join(path, "delay" + str(delay).zfill(3))
            Path(delay_path).mkdir()
        logger.info("Created a directory for each delay in {}".format(path))
    
    def create_pump_dirs(self, path):
        # Since we sometimes (Fabry-Perot) collect data at different
        # pump frequencies we create a folder for each pump pixel in the
        # pump pixel file seperately
        for pump_pixel in range(self.pump_pixels.size):
            pump_path = os.path.join(path, "pump_pixel" + str(pump_pixel).zfill(3))
            Path(pump_path).mkdir()
        logger.info("Created a directory for each pump pixel in {}".format(path))

    def save_scan(self, data: ndarray, scan_idx: int, delay_idx: int=None, pump_idx: int=None, **kwargs):
        """
        Saves data from single scan to binary file.

        Args:
            data (ndarray): Data which should be saved as binary.
            scan_idx (int): Index of scan.
            delay_idx (int, optional): Index of delay. Defaults to None.
            pump_idx (int, optional): Index of pump pixel / frequency. Defaults to None.
            kwargs (int, optional): Can be used to specify arbitrary index which will be appended
                to name of file. The name/key of the index will preprendend to the value of the index. 
                The value of the index will be padded with a maximum of 3 zeros.
        """
        # Sometimes no delays/pump pixels etc. are specified. 
        # Therefore it is necessary to differentiate. 
        file_prefix = ""
        path = self.scan_path
        if type(self.pump_pixels) == ndarray:
            file_prefix = file_prefix + "p" + str(pump_idx).zfill(3) + "_" 
            path = os.path.join(path, "pump_pixel" + str(pump_idx).zfill(3))
        
        if type(self.delays) == ndarray:
            file_prefix = file_prefix + "d" + str(delay_idx).zfill(3) + "_"
            path = os.path.join(path, "delay" + str(delay_idx).zfill(3))
        
        # Add kwargs keywords to file prefix
        for k, v in kwargs.items():
            file_prefix = file_prefix + k + str(v).zfill(3) + "_"

        file_name = "s" + str(scan_idx).zfill(6) +  "_" + file_prefix + self.file_name
        path = os.path.join(path, file_name)
            
        np.save(path, data)
        logger.info("Saved data of scan {} of delay {} of pump {} to {}".format(scan_idx, delay_idx, pump_idx, path))
    
    def save_weights(self, data: ndarray, scan_idx: int, delay_idx: int=None, pump_idx: int=None, **kwargs):
        """
        Saves data from single scan to binary file.
        In this case the data is intended to be
        the inverse variance of the transmission
        for each state. Any other data that should
        be saved as weights can be used too.

        Args:
            data (ndarray): Data which should be saved as binary.
            scan_idx (int): Index of scan.
            delay_idx (int, optional): Index of delay. Defaults to None.
            pump_idx (int, optional): Index of pump pixel / frequency. Defaults to None.
            kwargs (int, optional): Can be used to specify arbitrary index which will be appended
                to name of file. The name/key of the index will preprendend to the value of the index. 
                The value of the index will be padded with a maximum of 3 zeros.
        """
        # Sometimes no delays/pump pixels etc. are specified. 
        # Therefore it is necessary to differentiate. 
        file_prefix = ""
        path = self.scan_path
        if type(self.pump_pixels) == ndarray:
            file_prefix = file_prefix + "p" + str(pump_idx).zfill(3) + "_" 
            path = os.path.join(path, "pump_pixel" + str(pump_idx).zfill(3))
        
        if type(self.delays) == ndarray:
            file_prefix = file_prefix + "d" + str(delay_idx).zfill(3) + "_"
            path = os.path.join(path, "delay" + str(delay_idx).zfill(3))
        
        # Add kwargs keywords to file prefix
        for k, v in kwargs.items():
            file_prefix = file_prefix + k + str(v).zfill(3) + "_"
        
        file_prefix = file_prefix + "weights_"
        file_name = "s" + str(scan_idx).zfill(6) +  "_" + file_prefix + self.file_name
        path = os.path.join(path, file_name)

        np.save(path, data)
        logger.info("Saved weights of scan {} of delay {} of pump {} to {}".format(scan_idx, delay_idx, pump_idx, path))
        
    def save_s2s_std(self, data: ndarray, scan_idx: int, delay_idx: int=None, pump_idx: int=None, **kwargs):
        """
        Saves data from single scan to binary file.
        In this case the data is intended to be
        the standard deviation of the shot-to-shot
        difference signal.

        Note:
            For correct weighting this data then needs
            to be transformed into the inverse variances
            of the shot-to-shot difference signals.
            For this square data and take inverse.

        Args:
            data (ndarray): Data which should be saved as binary.
            scan_idx (int): Index of scan.
            delay_idx (int, optional): Index of delay. Defaults to None.
            pump_idx (int, optional): Index of pump pixel / frequency. Defaults to None.
            kwargs (int, optional): Can be used to specify arbitrary index which will be appended
                to name of file. The name/key of the index will preprendend to the value of the index. 
                The value of the index will be padded with a maximum of 3 zeros.
        """
        # Sometimes no delays/pump pixels etc. are specified. 
        # Therefore it is necessary to differentiate. 
        file_prefix = ""
        path = self.scan_path
        if type(self.pump_pixels) == ndarray:
            file_prefix = file_prefix + "p" + str(pump_idx).zfill(3) + "_" 
            path = os.path.join(path, "pump_pixel" + str(pump_idx).zfill(3))
        
        if type(self.delays) == ndarray:
            file_prefix = file_prefix + "d" + str(delay_idx).zfill(3) + "_"
            path = os.path.join(path, "delay" + str(delay_idx).zfill(3))
        
        # Add kwargs keywords to file prefix
        for k, v in kwargs.items():
            file_prefix = file_prefix + k + str(v).zfill(3) + "_"
        
        file_prefix = file_prefix + "s2s_std_"
        file_name = "s" + str(scan_idx).zfill(6) +  "_" + file_prefix + self.file_name
        path = os.path.join(path, file_name)

        np.save(path, data)
        logger.info("Saved s2s standard deviation data of scan {} of delay {} of pump {} to {}".format(scan_idx, delay_idx, pump_idx, path))
        
    def save_counts(self, data: ndarray, scan_idx: int, delay_idx: int=None, pump_idx: int=None, **kwargs):
        """
        Saves data from single scan to binary file.
        In this case the data is intended to be
        the counts (number of samples acquired in
        each state).

        Args:
            data (ndarray): Data which should be saved as binary.
            scan_idx (int): Index of scan.
            delay_idx (int, optional): Index of delay. Defaults to None.
            pump_idx (int, optional): Index of pump pixel / frequency. Defaults to None.
            kwargs (int, optional): Can be used to specify arbitrary index which will be appended
                to name of file. The name/key of the index will preprendend to the value of the index. 
                The value of the index will be padded with a maximum of 3 zeros.
        """
        # Sometimes no delays/pump pixels etc. are specified. 
        # Therefore it is necessary to differentiate. 
        file_prefix = ""
        path = self.scan_path
        if type(self.pump_pixels) == ndarray:
            file_prefix = file_prefix + "p" + str(pump_idx).zfill(3) + "_" 
            path = os.path.join(path, "pump_pixel" + str(pump_idx).zfill(3))
        
        if type(self.delays) == ndarray:
            file_prefix = file_prefix + "d" + str(delay_idx).zfill(3) + "_"
            path = os.path.join(path, "delay" + str(delay_idx).zfill(3))
        
        # Add kwargs keywords to file prefix
        for k, v in kwargs.items():
            file_prefix = file_prefix + k + str(v).zfill(3) + "_"
        
        file_prefix = file_prefix + "counts_"
        file_name = "s" + str(scan_idx).zfill(6) +  "_" + file_prefix + self.file_name
        path = os.path.join(path, file_name)

        np.save(path, data)
        logger.info("Saved count data of scan {} of delay {} of pump {} to {}".format(scan_idx, delay_idx, pump_idx, path))

    def save_raw_data(self, data: ndarray, scan_idx: int, delay_idx: int=None, pump_idx: int=None, **kwargs):
        """
        Saves raw data from single scan to binary file.

        Args:
            data (ndarray): Data which should be saved as binary.
            scan_idx (int): Index of scan.
            delay_idx (int, optional): Index of delay. Defaults to None.
            pump_idx (int, optional): Index of pump pixel / frequency. Defaults to None.
            kwargs (int, optional): Can be used to specify arbitrary index which will be appended
                to name of file. The name/key of the index will preprendend to the value of the index. 
                The value of the index will be padded with a maximum of 3 zeros.
        """
        if self.raw_data:
            # Sometimes no delays/pump pixels etc. are specified. 
            # Therefore it is necessary to differentiate. 
            file_prefix = ""
            path = self.raw_data_path
            if type(self.pump_pixels) == ndarray:
                file_prefix = file_prefix + "p" + str(pump_idx).zfill(3) + "_" 
                path = os.path.join(path, "pump_pixel" + str(pump_idx).zfill(3))
            
            if type(self.delays) == ndarray:
                file_prefix = file_prefix + "d" + str(delay_idx).zfill(3) + "_"
                path = os.path.join(path, "delay" + str(delay_idx).zfill(3))
            
            # Add kwargs keywords to file prefix
            for k, v in kwargs.items():
                file_prefix = file_prefix + k + str(v).zfill(3) + "_"

            file_name = "s" + str(scan_idx).zfill(6) +  "_" + file_prefix + self.file_name + "_raw"
            path = os.path.join(path, file_name)
            np.save(path, data)

            logger.info("Saved raw data of scan {} of delay {} of pump {} to {}".format(scan_idx, delay_idx, pump_idx, path))
        else:
            logger.warning("Save raw data was called although SaveData instance was not specified to save raw data. Doing nothing.")
    
    def save_pump_spectrum(self, data: ndarray, scan_idx: int, pump_idx: int):
        """
        Saves data from single scan to binary file.
        In this case the data is intended to be
        the pump spectrum of a Fabry Perot tuned
        to a given pixel.

        Because the delays are looped within each
        pump frequency the pump spectrum only needs 
        to be saved once for all delays within one
        scan. They are directly saved into each
        pump pixel folder. If the order of looping
        within the experiment ever changes this needs
        to be reconfigured.

        Args:
            data (ndarray): Data which should be saved as binary.
            scan_idx (int): Index of scan.
            pump_idx (int): Index of pump pixel / frequency.
        """
        if type(self.pump_pixels) == ndarray:
            path = self.scan_path
            
            file_prefix = "p" + str(pump_idx).zfill(3) + "_" 
            path = os.path.join(path, "pump_pixel" + str(pump_idx).zfill(3))

            file_prefix = file_prefix + "pump_spectrum_"

            file_name = "s" + str(scan_idx).zfill(6) +  "_" + file_prefix + self.file_name
            
            path = os.path.join(path, file_name)
            np.save(path, data)

            logger.info("Saved pump spectrum of scan {} of pump {} to {}".format(scan_idx, pump_idx, path))
        else:
            logger.warning("Save pump spectrum was called although SaveData instance was not specified to pump pixels. Doing nothing.")
            
            
    def save_avg(self, data: ndarray, delay_idx: int=None, pump_idx: int=None, **kwargs):
        """
        Saves averaged data to binary file

        Args:
            data (ndarray): Data which should be saved as binary.
            delay_idx (int, optional): Index of delay. Defaults to None.
        """
        # Sometimes no delays/pump pixels etc. are specified. 
        # Therefore it is necessary to differentiate. 
        file_prefix = ""
        path = self.avg_path
        if type(self.pump_pixels) == ndarray:
            file_prefix = file_prefix + "p" + str(pump_idx).zfill(3) + "_" 
        
        if type(self.delays) == ndarray:
            file_prefix = file_prefix + "d" + str(delay_idx).zfill(3) + "_"
            
        # Add kwargs keywords to file prefix
        for k, v in kwargs.items():
            file_prefix = file_prefix + k + str(v).zfill(3) + "_"

        file_name =  file_prefix + self.file_name + "_averaged"
        path = os.path.join(path, file_name)

        np.save(path, data)
        logger.info("Saved total/averaged data of delay {} of pump {} to {}".format(delay_idx, pump_idx, path))

    def save_readme(self):
        pass
    
    def save_logfile(self):
        pass
    
    def save_other(self, data: ndarray, name: str):
        """
        Save numpy array (in binary) to main folder with 
        name specified in name variable.

        This can be used to save probe wavenumber axis etc. 

        Args:
            data (ndarray): Array that should be saved in main folder
                of the folder structure.
            name (str): Name that will be prepended to the general file name.
        """
        file_name = name + "_" + self.file_name
        path = os.path.join(self.main_path, file_name)
        np.save(path, data)
        logger.info("Saved data to {}".format(path))
        
    def save_figures(self, scan_idx: int):
        """
        Returns path to save figure for a given scan.


        Args:
            scan_idx (int): Index of scan.

        Returns:
            [path]: Path including filename for the figure of a given scan.
        """
        file_name = "s" + str(scan_idx).zfill(6) + "_" + self.file_name
        path = os.path.join(self.fig_path, file_name)
        return path

class Background():
    """
    Class that handels saving and loading of
    background / dark noise data on detector.

    It enabled saving background data in the
    specified directory and loading the most recent
    file from the directory. If the directory
    does not exist it will be created.

    Args:
        path (str): Path to folder where background
            data should be saved and loaded.
    """
    def __init__(self, path: str):
        self.path = path
        # Create directory if it does not exist
        if not os.path.isdir(self.path):
            Path(self.path).mkdir(parents=True, exist_ok=True)

    def save_background(self, background_data: ndarray):
        """
        Saves background to folder specified in path
        attribute and prepends the current date and time
        to the filename.
        
        Args:
            background_data (ndarray): Raw, nonlinearized
                averaged data during time when detector was
                closed. shape: 1D e.g. (number of channels).
                We save all the data (also from channels
                that are not pixels) because this will
                make data processing easier in the routines. 
                
        """
        # Get current date and time
        now = datetime.now()
        # Format now to string
        file_name = now.strftime("%Y%m%d_%H_%M_%S_background")
        
        path = os.path.join(self.path, file_name)
        np.save(path, background_data)
        
        logger.info("Saved background data to {}".format(path))
    
    def load_background(self) -> ndarray:
        """
        Loads most recent background from file.

        Returns:
            ndarray: Raw, non linearized background for each
                pixel. shape: 1D e.g. (n_pixels)
        """

        # Get all data in directory
        list_of_backgrounds = glob.glob(os.path.join(self.path, '*.npy'))
        
        # Check if there is a background in the directory
        if not list_of_backgrounds:
            logger.error("No background data in {}! Aborting.".format(self.path))
            return False

        # Find most recent file
        latest_file = max(list_of_backgrounds, key=os.path.getctime)
        
        # Check if the background is from today
        if date.today().strftime("%Y%m%d") not in latest_file:
            logger.warning("""No background has been collected today. It
is recommended to block all light on detector and collect background.""")

        # Load data
        background = np.load(latest_file)
        logger.info("Returning {}".format(latest_file))
        
        return background

        
            
if __name__ == "__main__":
    path = r"C:\Users\H-Lab\Documents\testing_experiments"
    file_name = "JajaImportantDataForOurEyesOnly"
    username = "Grüner Powerranger"
    
    data = np.zeros((12, 25))

    # Test SaveData without delays and without pump pixel
    # t1 = SaveData(path, "no_delays_no_pixel", username)
    # t1.save_scan(data, scan_idx=0)
    # t1.save_scan(data, scan_idx=15)
    # t1.save_scan(data, scan_idx=16)
    # t1.save_scan(data, scan_idx=124)
    
    # # Test SaveData with delays and without pump pixel
    # delays = np.arange(25)
    # t2 = SaveData(path, "delays_no_pixel", username, delays=delays)
    # for delay in delays:
    #     t2.save_scan(data, scan_idx=0, delay_idx=delay)
    #     t2.save_scan(data, scan_idx=15, delay_idx=delay)
    #     t2.save_scan(data, scan_idx=16, delay_idx=delay)
    #     t2.save_scan(data, scan_idx=124, delay_idx=delay)
    
    # # Test SaveData with delays and with pump pixel
    # pump_pixels = np.arange(5)
    # delays = np.arange(25)
    # t3 = SaveData(path, "delays_pixel", username, delays=delays, pump_pixels=pump_pixels)
    # for pump_pixel in pump_pixels:
    #     for delay in delays:
    #         t3.save_raw(data, scan_idx=0, delay_idx=delay, pump_idx=pump_pixel)
    #         t3.save_scan(data, scan_idx=15, delay_idx=delay, pump_idx=pump_pixel)
    #         t3.save_scan(data, scan_idx=16, delay_idx=delay, pump_idx=pump_pixel)
    #         t3.save_scan(data, scan_idx=124, delay_idx=delay, pump_idx=pump_pixel)
    #         t3.save_s2s_std(data, scan_idx=3333, delay_idx=delay, pump_idx=pump_pixel)
    
    # Test SaveData with delays and with pump pixel raw data saving
    pump_pixels = np.arange(5)
    delays = np.arange(25)
    t3 = SaveData(path, "delays_pixel", username, delays=delays, pump_pixels=pump_pixels, raw_data=True)
    for pump_pixel in pump_pixels:
        for delay in delays:
            t3.save_counts(data, scan_idx=55, delay_idx=delay, pump_idx=pump_pixel, verraeter=31)
            t3.save_raw_data(data, scan_idx=0, delay_idx=delay, pump_idx=pump_pixel, auge_idx=12, judas_idx=999, chivato = 31)
            t3.save_raw_data(data, scan_idx=15, delay_idx=delay, pump_idx=pump_pixel, judas_idx=999)
            t3.save_raw_data(data, scan_idx=16, delay_idx=delay, pump_idx=pump_pixel, chivato = 31)
            t3.save_raw_data(data, scan_idx=124, delay_idx=delay, pump_idx=pump_pixel)
            t3.save_avg(data, 298, 55, judas=1324)
            t3.save_pump_spectrum(data, 666, pump_idx=pump_pixel)