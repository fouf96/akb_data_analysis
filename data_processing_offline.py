"""
The data_processing script contains all methods which, as the name
suggests, involve data processing. This is important to decouple the
hardware classes, GUI elements and other layer of the software
architecture from each other. The data_processing module does not only
contain independent methods but also separate classes.

**Overview:**
    
    **Independent Methods:**
        
        1.  Utility Methods:

            * convert_fs_to_mm
            * convert_mm_to_fs
        
        2.  Phase Cycling Methods:

            * calculate_interleave_array
            * get_wobbler_states

        3.  Methods for Time Domain Experiments:

            * find_wavegen_params
            * find_t_zero
            * calculate_counter_values
            * process_ft2dir_data
            * apodization_function
            * process_interferogram
            * calculate_frequency_axis
            * find_opa_range
            * calculate_phase_slope
            * find_zerobin

        4.  Methods for Data Handling:

            * sort_data
            * shot_to_shot_signal
            * shot_to_shot_viper

        5.  Visualization Methods for Plotting:
            * generate_img_data
            * scale_img
            * generate_contour_lines
            * update_contour_lines
        

    **Classes:**

        1.  PixelResponseLinearization:

            * _linearize_cubicfraction
            * _linearize_cubic
            * _linearize_one

        2.  ChopperStateFinder:

            * get_chopper_states
"""
import logging
import numpy as np
from numpy import ndarray
# from hardware_properties import HardwareProperties
# from analog_digital_converter import AnalogDigitalConverter

# Scipy modules
from scipy.fft import rfft, fft, next_fast_len
from scipy.stats import linregress
from scipy.signal import find_peaks, peak_widths
from scipy.signal.windows import get_window

# Pandas
import pandas as pd

# SymPy for find_wavegen_params()
from sympy import divisors

# Modules needed to accomodate
# 2D plots with pyqtgraph
from scipy.interpolate import RectBivariateSpline
import pyqtgraph as pg
from PyQt5 import QtCore

# this is important! Otherwise the axis order is reversed w.r.t. numpy
# arrays!
pg.setConfigOptions(imageAxisOrder="row-major")


import json

# Set up logger
logger = logging.getLogger(__name__)


def convert_fs_to_mm(femtoseconds: float):
    """
    Converts femtoseconds to millimeters.

    Args:
        femtoseconds (float): Value in femtoseconds.

    Returns:
        float: Value in millimeters.
    """
    speed_of_light = 299792458  # in m/s
    return femtoseconds * 1e-12 * speed_of_light


def convert_mm_to_fs(mm: float):
    """
    Converts femtoseconds to millimeters.

    Args:
        mm (float): Value in millimeters.

    Returns:
        float: Value in femtoseconds.
    """
    speed_of_light = 299792458  # in m/s
    return (mm * 1e12) / speed_of_light


def calculate_interleave_array(
    interleaves: int, pump_wavelength: float, delay: float
) -> ndarray:
    """
    Generate interleave delay positions.

    Interleaves are positions of the delay stage within one wavelength
    of the pump frequency. We use the interleaves to suppress
    interference on the MCT detector caused by scattering or similar
    phenomena. When moving the delay stage a specific delay, we have to
    additionally move the delay stage in small steps around this delay.
    These small steps are what we call interleaves and because they are
    within one cycle of the wavelength this effectively changes the
    phase of the pump pulse. This change in phase leads to the
    interference being cancelled out when averaging the interleaves.

    Args:
        interleaves (int): Number of interleaves to measure.
        pump_wavelength (float): Wavelength of the pump pulse in nanometers.
        delay (float): Population delay that is being targeted in femtoseconds.

    Returns:
        ndarray: absolute interleave positions (delay + respective interleave)
            in femtoseconds. 
                
                * shape: 1D (interleaves)

    """
    speed_of_light = 299792458 * 1e-6  # speed of light in nm/fs = 1E9/1E15 = 1E-6
    # Period of central wavelength of pump pulse in fs
    pump_waveperiod = pump_wavelength / speed_of_light
    # Generate positions in fs that need to be targeted relative to
    # delay position We create positions from 0 to end of period where
    # the end point is excluded because including it would imply
    # measuring one interleave twice.
    relative_interleave_pos = np.linspace(
        start=0, stop=pump_waveperiod, num=interleaves, endpoint=False
    )
    # Add delay to get absolute positions to target with PiStage.move()
    interlave_pos = relative_interleave_pos + delay
    return interlave_pos


from sympy import divisors


def find_wavegen_params(frequency, no_of_solutions=1, ylimit=4095, force=2, xmin=1000):
    """
    Finds optimum wavegen parameters for a desired frequency.

    The function returns a list of tuples of integers (x,y) such that
    the given frequency is optimally approximated by:
    
    .. math::

        f &= \\frac{1}{200 \\cdot 10^{-6} \\cdot x \\cdot y} \\\\
        \\rightarrow f &= \\frac{5000}{x \\cdot y}

    |   x must be between 2 and 4096 (inclusive) and should be as large as possible.
    |   y must be greater than 0. (should be limited at least to x_max)

    Args:
        frequency (float): value in Hertz
        no_of_solutions (int): number of solutions (default 1)
        ylimit (int): maximum y value (default 4095)
        force (int): force a divisor that x _must_ have
        xmin (int): lower limit of possible x

    Returns:
        [(x,y),...]: list of pairs of integers
    """

    solutions = []
    # target is the ideal *integer* product of x and y to get as close
    # to the desired frequency as possible
    target = int(5000 / (force * frequency) + 0.5)
    if target > 4096 * ylimit:
        return solutions
    # the basic idea is to find factors x and y for the target itself -
    # and further candidates in an increasing interval around target,
    # alternating below and above - that fit the restrictions. The order
    # of the "signs" makes sure we start on the correct side of 5000/f,
    # above or below, so the best approximation is really found first.
    signs = [-1, 1] if target > 5000 / (force * frequency) else [1, -1]
    for dist in range(target):  # checking ever further away from target
        for sign in signs:  # and consider candidates above and below
            approx_candidate = target + dist * sign
            for y in divisors(approx_candidate):
                x = approx_candidate // y
                if x < xmin // force:
                    break
                if x > 4096 // force:
                    continue
                # it helps to limit y in order to avoid technically
                # correct, but ridiculous solutions. This can happen for
                # very small frequencies where the best solution would
                # be a small x, with y being some absurdly large prime
                # number
                if y > ylimit:
                    break
                # print(target, dist, sign, approx_candidate, y, x,
                # frequency-5000/(approx_candidate))
                solutions.append((force * x, y))
                if len(solutions) == no_of_solutions:
                    return solutions
                break  # to avoid use multiple solutions like (1000,1) -> (500,2), (250,4)
            if dist == 0:
                break  # in this case don't jump in both directions
                # can only happen at the start of search
    return solutions


class PixelResponseLinearization:
    """
    Provides functionality to linearize the MCT pixel response to light
    intensity.

    This class offers functionality to linearize the response of the MCT
    detector (and its preamplifiers etc.) to changing light intensity.

    For object instantiation a *JSON* file with the fit parameters has
    to be specified. The top level of the *JSON* file has two entries,
    one 'type' (str) describing the type of linearization, that is, the
    fit function, and one dict under the name 'parameters', containing
    either a) the specific linearization parameters for each pixel (dict
    is large), or one set of parameters to be used for all pixels.
    During loading, the two cases are handled automatically depending on
    the size of the 'parameters' dict. See below for an example on how
    an entry of the pixel linearization parameter dict looks like

    .. code-block::

        {
            "0": {
                "name": "Probe (bottom) pixel 0",
                "a": [
                    9.7706e-16,
                    "float",
                    "unitless",
                    "fit parameter a from equation Intensity=b*ADC^3+a*ADC+c"
                ],
                "b": [
                    1.3441e-05,
                    "float",
                    "unitless",
                    "fit parameter b from equation Intensity=b*ADC^3+a*ADC+c"
                ],
                "c": [
                    0.005075,
                    "float",
                    "unitless",
                    "fit parameter c from equation Intensity=b*ADC^3+a*ADC+c"
                ]
            },
            ...
        }

    The object method `linearize` is then set up according to the 'type'
    specified in the JSON file by picking one of the preimplemented
    functions.

    Args:
        path (path): Path to the json file that specifies the fit paramters.
            The code will automatically detect whether one set of parameters
            for all pixels or one set of parameters was provided for each pixel.

    Attributes:
        a,b,c (ndarray): Fit parameters. The specific meaning depends on the
            fit function, and there could be more than three. They are there for
            the use of the fit function only and should not be accessed directly.
                
                * shape: (number_of_mct_pixels) or (1) if only one set of parameters is
                  specified for all pixels.

    Warning:
        The fit parameters are generally only valid for the given settings of the
        pre-amplifiers and the ADC for which the calibration was done. If,
        for example, the gain is changed a new calibration measurement needs
        to be done.

    References:

        Datasheets and Manuals/Pixel correction/pixel_linearisation.pdf
    """

    def __init__(self, path):
        # Read json file that contains pixel linearisation data
        with open(path) as json_file:
            content = json.load(json_file)

        if "type" not in content:
            logger.error(
                "Error reading linearization parameters. \
                Probably the JSON file has the old format, \
                and fit type information is missing."
            )
            assert "type" in content
        else:
            pixel_linearisation_fit_parameters = content["parameters"]

        # determine the proper fit function and assign the method
        # _linearize_one is for testing purposes
        if content["type"] == "cubic":
            self.linearize = self._linearize_cubic
        elif content["type"] == "cubicfraction":
            self.linearize = self._linearize_cubicfraction
        elif content["type"] == "one":
            self.linearize = self._linearize_one
        else:
            logger.error("Unknown linearization type: " + content["type"])
            assert content["type"] in ["cubic", "cubicfraction", "one"]

        # Preallocate arrays for fit paramters
        #! if later a function with more parameters is used
        #! this may have to be moved into the respective
        #! function assignment slot above
        if len(pixel_linearisation_fit_parameters) == 1:
            self.a = np.zeros(len(pixel_linearisation_fit_parameters))
            self.b = np.zeros(len(pixel_linearisation_fit_parameters))
            self.c = np.zeros(len(pixel_linearisation_fit_parameters))
        else:
            self.a = np.zeros((len(pixel_linearisation_fit_parameters), 1))
            self.b = np.zeros((len(pixel_linearisation_fit_parameters), 1))
            self.c = np.zeros((len(pixel_linearisation_fit_parameters), 1))

        # Write fit parameters into arrays. The neat thing is that this
        # works for both: If we have one set of paramters for all pixels
        # and also if we have a set of linearisation paramters for all
        # pixels. If we have only one value for a,b,c then the
        # attributes will just be float values (not arrays). If we have
        # fit parameters for all pixels then we get arrays and the
        # linearisation equations do not change! Naice.
        for pixel, values in pixel_linearisation_fit_parameters.items():
            self.a[int(pixel)] = values["a"][0]
            self.b[int(pixel)] = values["b"][0]
            self.c[int(pixel)] = values["c"][0]

    def _linearize_cubicfraction(self, data: ndarray):
        """
        Linearises the pixel response of data set.

        It is assumed that the relation between the light intensity and
        the measured voltage at the ADC is of the form:
            
            .. math::

                I(U) = \\frac{a \\cdot U^{3} + b \\cdot U}{c \\cdot U + 1}
                
        where I is the light intensity and U is the voltage measured by
        the ADC. The parameters a,b,c are fit constants that were
        obtained from a calibration measurement. These parameters can
        either be specified separately for each pixel or summarized in
        an average fit parameter for all pixels.

        Args:
            data (ndarray): Array containing the measured voltages for the MCT
                detector. 

                    * shape: (number_of_pixels, samples_to_acquire)

        Returns:
            ndarray: Array containing the intensities, obtained from the
                provided fit parameters and the MCT voltages.
        """
        # print(data)
        linearized = (self.a * np.float_power(data, 3) + self.b * data) / (
            self.c * data + 1
        )
        # print(linearized)
        return linearized

    def _linearize_cubic(self, data: ndarray):
        """
        Linearises the pixel response of data set.

        It is assumed that the
        relation between the light intensity and the measured voltage at the ADC is
        cubic: 
            
            .. math::

                I(U) = b \\cdot U^{3} + a \\cdot U + c

        where I is the light intensity and U is the voltage measured by
        the ADC. The parameters a,b,c are fit constants that were
        obtained from a calibration measurement. These parameters can
        either be specified separately for each pixel or summarized in
        an average fit parameter for all pixels.

        Args:
            data (ndarray): Array containing the measured voltages for the MCT
                detector. 

                    * shape: (number_of_pixels, samples_to_acquire)

        Returns:
            ndarray: Array containing the intensities, obtained from the
                provided fit parameters and the MCT voltages.
        """
        # print(data)
        linearized = self.b * np.float_power(data, 3) + self.a * data + self.c
        # print(linearized)
        return linearized

    def _linearize_one(self, data: ndarray):
        """Fake linearization for testing purposes

        Returns a matching list of ones for any input data.

        Args:
            data (ndarray): Array containing the measured voltages for the MCT
                detector. 

                    * shape: (number_of_pixels, samples_to_acquire)

        Returns:
            ndarray: array of the same shape as data, but filled with one's
        """
        return data * 0 + 1


def get_wobbler_states(
    wobbler_adc_data: ndarray, laser_freq: float, wobbler_freq: float = 250
) -> ndarray:
    """
    Assigns each state/position of the wobbler a number 0,1,2,3...

    The number 0 indicated that the Wobbler-Voltage was at the maximum
    value. This method assumes that values oscillate perfectly, with no
    skipping.

    Note:
        This will only work if the wobbler frequency is 1/4 of the laser
        repetition rate.

    Args:
        wobbler_adc_data (ndarray): Array containing ADC values of the Wobbler.
            It technically does not matter if these values come directly from
            the reference coil, or run through the an additional arduino.
                
                * shape: (adc.samples_to_acquire), other 1D shapes should work fine too.

        laser_freq (float): laser repition rate in Hz
        wobbler_freq (float, optional): Resonance frequency of the wobbler in Hz.
            Defaults to 250 Hz.

    Returns:
        ndarray: Array containing values 0,1,2,3..
            Where 0 corresponds to the maximum Wobbler-Voltage. 

                * shape: (wobbler_adc_data.size)
    """
    # For a 1 kHz laser and a 250 Hz Wobbler we would observe 4
    # repeating positions of the wobbler. For 3 kHz laser and 250 Hz
    # Wobbler we would observe 12 different repeating positions of the
    # wobbler.
    number_of_states = int(laser_freq / wobbler_freq)
    # Return repeating values of 0,1,2,3.. s.t. the value 0 corresponds
    # to the "high" of the wobbler.
    return (
        np.arange(wobbler_adc_data.size)
        + np.argmax(wobbler_adc_data[0:number_of_states])
    ) % number_of_states


class ChopperStateFinder:
    """
    Provides functionality to identify which combination of chopper
    states was present for an input channel that mixes (adds up) the
    voltage level of a combination of choppers.

    The class first loads all necessary information from a json file
    that specifies the voltage range and name of all choppers. It then
    creates an array containing all possible combinations of chopper
    states. This array is then used to calculate bin reference values
    for the *numpy.digitize* function.

    Note:

        At the moment the class only provides functionality to digitize the different chopper
        states. Functionality to identify which choppers had a **HIGH** signal is fairly easy
        by using the *self.deconvolution_matrix* indexed with the return of the *np.digitize* function.
        It is not clear if this is needed at the moment thats why it is left out.

    Args:
        path (str): Path to the json file that contains the chopper configuration for the experiment.

    Attributes:
        chopper_names (list): Names of the choppers, ordered such that the chopper with the highest
            voltage value is listed first. This corresponds to the order of the columns of deconvolution
            matrix.
        number_of_choppers (int): Number of choppers specified/characterised in json file. Corresponds to
            the number of choppers needed for the experiment.
        deconvolution_matrix (ndarray): Array containing all possible combinations of chopper states.
             
                * shape: (2^number_of_choppers, number_of_choppers). The 0th column represents the chopper with the
                  highest voltage. This coincides with the chopper_names list 
                * E.g.: the 1st entry of chopper_names corresponds to the chopper 
                  represented by the 1st column of deconvolution matrix.

        bin_reference_values (ndarray): Half way points between to adjacent voltage states. This is used by the
            np.digitize function.
    """

    def __init__(self, path):

        # Read json file that contains chopper information to dictionary
        with open(path) as json_file:
            chopper_config = json.load(json_file)
        # Create list that will hold HIGH voltage values for each
        # chopper
        chopper_reference_values = []
        # Create dictionary so we can identify which voltage value
        # belongs to which chopper
        chopper_voltage_to_name = {}
        # Iterate over all choppers in json file
        for chopper in chopper_config.keys():
            # Iterate over all properties of the chopper
            for key, value in chopper_config[chopper].items():
                # Extract voltage information
                if key == "voltage":
                    # We are interested in the 0 entry of "value"
                    # (voltage)
                    chopper_reference_values.append(value[0])
                    # Assign voltage value a chopper name with
                    # dictionary
                    chopper_voltage_to_name[value[0]] = chopper

        # Sort chopper reference values such that lowest values is
        # furthest to the right This is necessary because lowest value
        # should correspond to least significant digit
        chopper_reference_values.sort(reverse=True)
        chopper_reference_values = np.array(chopper_reference_values)
        # Create list such that chopper names are sorted in the same way
        # as their voltage values.
        self.chopper_names = [
            chopper_voltage_to_name[key] for key in chopper_reference_values
        ]

        # Count number of choppers in setup
        self.number_of_choppers = chopper_reference_values.size

        # If there is only 1 chopper in the setup there is no need to
        # run the rest of the algorithm
        if self.number_of_choppers == 1:
            self.bin_reference_values = chopper_reference_values / 2
            return

        # In the next step we create an array that should contain all
        # possible combinations of high/low states of the choppers.
        # Where each column represents a chopper and the right most
        # column represents the chopper with the lowest voltage value.
        # Each row corresponds to a different combination of states. We
        # can then use matrix multiplication with the chopper reference
        # values / voltages to calculate all possible observable
        # signals.

        # There are 2^number_of_choppers different combinations of
        # states.
        number_of_states = 2 ** self.number_of_choppers
        # We can represent each of these states as the binary
        # representation of a number from 0 to 2^number_of_choppers - 1
        # where each digit corresponds to the state of one specific
        # chopper: Preallocate array
        convolution_matrix = np.zeros(
            (number_of_states, self.number_of_choppers), dtype="int8"
        )
        # ---------- Easier to read but slower version of code -------------
        # The following code creates and equivalent version of
        # convolution matrix, but slower. for number in
        # range(number_of_states): # Get the binary representation of
        # the number as a string of length width. binary =
        # np.binary_repr(number, width=self.number_of_choppers) #
        # Convert string list of strings, which is then converted to
        # numpy integer array convolution_matrix[number] =
        # np.asarray(list(binary), dtype=int)
        # ---------- Fast version of code START ---------
        h = 1
        for i in range(self.number_of_choppers):
            convolution_matrix[h : 2 * h, :] = convolution_matrix[:h, :]
            convolution_matrix[h : 2 * h, self.number_of_choppers - i - 1] = 1
            h *= 2
        # ---------- Fast version of code END ---------

        # Using matrix multiplication we can now calculate  possible
        # voltage convolutions/states
        chopper_voltage_states = np.matmul(convolution_matrix, chopper_reference_values)
        # We compute the bin reference values for the numpy.digitize
        # function by calculating the mid points between all adjacent
        # voltage levels.
        self.bin_reference_values = (
            chopper_voltage_states[:-1] + np.diff(chopper_voltage_states) / 2
        )
        # Luckily, we can also use the convolution matrix as
        # deconvolution matrix, to generate an array telling us which
        # chopper in HIGH by calling deconvolution_matrix[bin_index]
        self.deconvolution_matrix = convolution_matrix

    def get_chopper_states(self, chopper_adc_data: ndarray) -> ndarray:
        """
        Digitizes voltage data of convoluted chopper reference signal into integers that can be
        used to sort the adc data.

        Args:
            chopper_adc_data (ndarray): Analog voltage data from chopper input channel of adc.
                
                    * shape: 1D
                    * E.g.:(adc.samples_to_acquire)

        Returns:
            ndarray: Array that represents each respective voltage state as an integer.
                Indexing *self.deconvolution_matrix* with this array will return the respective
                state for each chopper for each data point.
                
                    * shape: 1D
                    * E.g.: (adc.samples_to_acquire)
        """
        return np.digitize(chopper_adc_data, self.bin_reference_values)


# ---- Find t zero
def find_t_zero(signal, delays):
    pass


# ---- Process time domain data


def calculate_counter_values(
    data: ndarray, r2r_indices: ndarray, bin_reference_values: ndarray
) -> ndarray:
    """
    Determines the interferometer position from the (counter) values
    that the ADC collected from the R-2R-Network.

    The counter electronics from Zurich outputs the position via USB (on
    command) and also on a 16 line parallel port for every laser
    trigger. The results in a 16 digit binary number that is then
    converted into 4 digit hexadecimal number. Which is output on a 4
    line parallel port. (Interferometer counter box BNC R-2R 1-4, BNC
    R-2R 5-8 etc.). This data is recorded by the ADC. To obtain the
    actual counter values we need to decode the analog voltages that the
    ADC recorded. This is achieved by binning the data from all four
    channels accordingly using an array containing reference values and
    the function np.digitize. These reference values have to be measured
    manually from the R-2R Network by applying 5V to different R2R
    inputs and recording the corresponding output voltages. These values
    have to be written in a .csv file (see existing files). The
    resulting hexadecimal values for each line are then modified to
    match their poisition in the hexadecimal tetrade using the function
    *np.left_shift*. BNC R-2R 1-4 is interpreted to be the channel
    containing the least significant digit (and so on). Finally, we get
    the counter position by summation of all four hexadecimal numbers.
    We now need to divide by 2 because apparently the circuit counts the
    zero crossings for both photodiodes. (Technically it only makes to
    count on one photodiode, because the actual resolution is determined
    by the HeNe Wavelength.) Afterwards we floor the values to obtain
    integers. Note this is not mentioned in the datasheet and apparently
    is a empirically determined phenomenon (to understand how to prove
    that the flooring and division by two is correct see the comments
    within the code).

    References:
        |   H-Lab reference values for R2R-Network.csv
        |   TimingScheme.pdf
        |   Counter für Interferometer; Anleitung; Manual; Universitaet Zuerich.pdf

    Args:
        data (ndarray): Complete data set from adc (including pixel data, wobbler, R2R etc.)
            The indices/ rows containing the information of the R2R are selected automatically.
            
                * shape: 2D 
                * E.g. (number of channels, samples to acquire)

        r2r_indices (ndarray): Indices of the rows in the raw data array from adc that
            correspond to the input channels connected to the R-2R networks.
            
                * shape: 1D 
                * E.g. (4) the counter is connected to 4 R-2R networks

        bin_reference_values (ndarray): Reference values for each of the R-2R
            networks. 

                * shape: 2D (4, 15) 4 rows for each of the R-2R Networks, 15 values for
                  the 16 levels of the R-2R.

    Note:
        This function was copied from the interferometer_counter module, because
        it is necessary to be able to call this function without a reference to the
        actual InterferometerCounter class. This is due to issues regarding pickling
        within multiprocesses.

    Returns:
        ndarray: array containing interferometer position in counts.
                * shape: (adc.samples_to_acquire)
    """
    # Compute corresponding hexadecimal levels (0-15) from R2R voltage.
    # The np.digitize function returns the bin number (given by
    # bin_reference_values) in which a measured voltage value belongs.
    inds = np.zeros(data[r2r_indices, :].shape, dtype="uint16")
    for row in range(4):
        inds[row, :] = np.digitize(
            data[r2r_indices[row], :], bin_reference_values[row, :]
        )

    # Now we have the corresponding digits for each tetrade but they are
    # not in the correct hexadecimal position: For example: The digit
    # from the R2R line 13-16 might be 0x6 (hexadecimal digit 6) but the
    # actual value it corresponds to is 0x6000 (0x indicates hexadecimal
    # number). So to shift all values to their correct position we shift
    # the bits a corresponding multiple of 4 to the left
    # (np.left_shift). We need to transpose inds because otherwise numpy
    # can not broadcast the two arrays (inds and [0,4,8,12]) together.
    # To obtain the actual counter values we now need to add all the
    # values from all lines for each point in time together. (E.g.: For
    # each point in time we have 4 values, i.e.: [0x1, 0xA0, 0x700,
    # 0xE000] we obtain the actual counter value by summing these
    # values: 0xE7A1 = 59297)

    counter_values = np.left_shift(inds.T, np.arange(0, 16, 4)).sum(axis=1)

    # We now need to divide by 2 because apparently the circuit counts
    # the zero crossings for both photodiodes. (Technically it only
    # makes sense to count on one photodiode, because the actual
    # resolution is determined by the HeNe Wavelength.) Afterwards we
    # floor the values to obtain integers. Note this is not mentioned in
    # the datasheet and apparently is a empirically determined
    # phenomenon copied from LabView. Update (20200925): We tried to
    # find out if the division by two is correct/appropriate. For this
    # we set the coherence time of the interferometer (for the
    # wavegeneration) to 4 ps we then plotted the resulting counter
    # values and looked at the maximum. The maximum was at approx 2300
    # (+/- 150) counts. A coherence time of 4 ps corresponds to approx
    # 1.45 mm light path difference (including the built in buffer of
    # the coherence time scan). To obtain the resolution of the counter
    # we divide the light path difference by the counts. This should
    # yield the HeNe Wavelength. 1.45 mm / 2300 = 630 nm. This is more
    # than close enough and means that the division by two is correct
    # and that the electronics does not do what it specified in the
    # manual (at least for the weird init that was copied from LabView).
    return np.uint16(counter_values / 2)


def process_ft2dir_data(
    data: ndarray,
    interferogram: ndarray,
    window_function: str = None,
    zero_pad_factor: int = 2,
):
    """
    Computes the phase corrected 2D-FTIR spectrum from the time domain
    data of the MCT (or more spefically the probe (pulse) absorption
    spectrum in the pump time domain) and applies a window function if
    specified.

    Step by Step Algorithm:
    
        1.  The interferogram data, collected from the Zurich counter electronics, is used to find the zerobin
            (the position of the interferometer from which we are interested in the data, which also corresponds to
            the position where the pump pulses perfectly, temporally overlap). Furthermore the interferogram and the
            fourier transformed interferogram (pump spectrum) are obtained together with the pump frequency axis and
            the necessary information to find the pump OPA pulse within the spectrum
        2.  The time domain data (absorption spectrum) is shortened such that it starts at the zerobin position
        3.  The time domain data (absorption spectrum) is offset corrected
        4.  The time domain data (absorption spectrum) is zeropadded for better interpolation. For better
            understanding of zeropadding see references
        5.  A window function apodization function is calculated and applied to the time domain data.
        6.  The time domain data is fourier transformed into the frequency domain
        7.  The phasing factor is calculated from the pump spectrum (which was obtained through the fourier
            transformed interferogram)
        8.  The frequency domain data is phase corrected

    See Also:
        * process_interferogram
        * find_zerobin
        * find_opa_range
        * apodization_function
        * calculate_frequency_axis

    References:
        |   Werner Herres and Joern Gronholz: Understanding FT·IR Data Processing Part 2

    Args:
        data (ndarray): Array which contains time domain data of probe pulse absorption spectrum.
            This can be calculated by applying lambert beers law
            (:math:`\\log_{10} \\frac{I_{0}}{I_{1}}`) on the linearized
            MCT data. 
            
                * shape: (number_of_pixel, interferometer positions)

        interferogram (ndarray): Array containing the interferogram which is used to obtain the pump frequency range
            and the zerobin (the position of the interferometer from which we are interested in the MCT data). 
            
                * shape: (interferometer positions)

        window_function (str, optional): Apodization function which is applied to the time domain data. See scipy.signal.windows.get_window
            Documentation for choices. For some of the windows functions additional parameters need to be provided.
            Self implemented choices: cos_square. Defaults to None.
        zero_pad_factor (int, optional): Factor with which we want to zeropad the time domain data. For the algorithm to work properly, a
            factor of at least 2 is necessary. Defaults to 2.

    Returns:
        tuple: Contains frequency domain data and interferogram information
            (ndarray, tuple)
        ndarray: spectrum_2d contains the frequency domain data from the
            mct pixels. 

                * shape: (number_of_pixels, zero padded time domain
                  data size)

        tuple: interferogram_information contains the zerobin, the
            interferogram and the fourier transformed interferogram as well
            as the information which is needed to plot the correct part of
            the data (in range of the pump OPA pulse). It also contains the
            pump frequency axis necessary for plotting.

    """

    # Get all the information we need out of the interferogram. Contains
    # zerobin, interferogram, fft_interferogram, frequency_axis,
    # opa_info (peak index, fwhm index range and index range of width)
    interferogram_information = process_interferogram(
        interferogram, zero_pad_factor=zero_pad_factor
    )

    # We are only interested in the data starting at the zerobin
    # (position where pump pulses perfectly, temporally overlap). In
    # front of the zerobin the delay between pump and probe pulses is
    # not clearly defined. (It varies for each interferometer position.)
    zerobin = interferogram_information[0]
    data = data[:, zerobin:]

    # Offset correction
    # The newaxis expression is used to increase the dimension of the
    # existing array by one more dimension, when used once
    data = data - np.average(data, axis=-1)[:, np.newaxis]

    # Checks which length the array should have for a more efficient FFT
    # As described in Understanding FT·IR Data Processing (Werner Herres and Joern Gronholz)
    # It is advised to at least zero_pad the time domain data with a factor of two of the
    # interferogram length. Thats why we multiply the size of our time domain with the zero
    # pad factor.
    efficient_length = next_fast_len(data.shape[-1] * zero_pad_factor)
    envelope = apodization_function(data.shape[-1])

    #! Julian phase corrects the time domain spectra instead of the
    #! frequency domain spectra. We believe that it mathematically makes
    #! no difference. But because rfft is faster, we are going to phase
    #! correct afterwards.
    # Compute Fourier Transform of data multiplied with envelope for each
    # row of the time domain data.
    spectrum_2d = rfft(data * envelope, n=efficient_length, axis=-1)

    # The phase needed for the correction is the "angle" of the entry of
    # the pump_spectrum maximum.
    fft_interferogram = interferogram_information[
        2
    ]  # Obtain the whole pump spectrum (fourier transformed interferogram)
    opa_info = interferogram_information[
        4
    ]  # Obtain the information where the pulse opa peak is
    #! Is this the correct phasing factor
    # ? Check whether this is the correct phasing factor

    phasing_factor = np.angle(
        fft_interferogram[opa_info[0]]
    )  # Obtain phase factor from pump opa maximum position
    spectrum_2d = -1 * np.real(
        spectrum_2d * np.exp(-1j * phasing_factor)
    )  # Phase correct the frequency domain data

    return spectrum_2d, interferogram_information


def apodization_function(data_size: int, window: str or tuple = None) -> ndarray:
    """
    Generate an envelope for time domain data to prevent artifacts like
    Leakage in FFT.

    This is essentially a wrapper around the *scipy.signal.windows* *get_window()*
    function. Additionally, the return of a cosine squared window to the existing
    windows. If no window is specified, an array filled with ones is returned.
    For some of the windows functions additional parameters need to be
    provided.
    
    E.g. for a Gaussian envelope the width (standard deviation) needs to be specified.
    In this case instead of a string, a Tuple needs to passed to window.

    Note:
        If you want to add additional windows not provided by the get window function
        you can implement the wanted functionality by adding more if clauses.

    References:

        |   Hamm, Zanni 2011 - Concepts and Methods of 2D Infrared Spectroscopy Section 9.5.7
        |   Werner Herres and Joern Gronholz: Understanding FT·IR Data Processing Part 2
        |   https://en.wikipedia.org/wiki/Window_function
        |   scipy.signal.windows.get_window

    Args:
        data_size (int): size/ length of the axis on which the fourier transform is performed.
            I.e.: If we have 64 Pixels and a coresponding data set which has the shape: (64, 1000)
            we want to perform the Fourier Transform along the last axis. And thus our data_size is
            1000.
        window (str or tuple, optional): Name of the window to use. See *scipy.signal.windows.get_window*
            Documentation for choices. For some of the windows functions additional parameters need to be provided.
            Self implemented choices: cos_square. Defaults to None (effectiveley resulting in an array
            full of ones).

    Returns:
        ndarray: envelope/ apodization function with which to multiply time domain data.
           
                * shape: (data:size)
    """
    if window is None:
        return np.ones(data_size)
    elif window == "cos_square":
        # Generate a window that has the shape of a cosine^2 starting at
        # 1 and stopping at 0. This is achieved by generating an evenly
        # spaced "x-Axis" going from 0 to pi/2 with data_size steps.
        window = np.cos(np.linspace(0, np.pi / 2, data_size)) ** 2
        return window
    else:
        window = get_window(window, data_size, fftbins=True)


def process_interferogram(interferogram: ndarray, zero_pad_factor: int = 2):
    """
    Obtain the zerobin, the pulse width of the opa and the frequency axis from an interferogram.

    Step by Step Algorithm:

        1.  The interferogram is offset corrected
        2.  Take the maximum of the interferogram as the initial guess
        3.  Zeropad interferogram to have an efficient length for Fourier Transform
        4.  Obtain the amplitude of the Fourier Transform of the interferogram
        5.  Determine full width half maximum (fwhm) indices of the OPA pump-spectrum as reference values where
            the phase is linear. This is important because the phase is only linear around the maximum and strongly fluctuates
            other wise
        6.  Calculate the phase for different points within fwhm and obtain its derivative/ slope by linear regression
        7.  The zerobin is the position where the slope of the phase is as close to zero as possible
        8.  Zeropad interferogram to match zero_pad_factor requirement
        9.  Compute Fourier Transform of zeropadded interferogram
        10. Get the OPA spectrum info: peak (maximum) index, indices of the pulse in the spectrum, indicies of
            of the fwhm

    Args:
        interferogram (ndarray): Interferogram data. Contains a voltage value for every bin measured by the pyro-detector.
            
                * shape: 1D 
                * E.g. (interferometer positions)

        zero_pad_factor (int, optional): Factor with which we want to zeropad the time domain data. For the algorithm to work properly, a
            factor of at least 2 is necessary. Defaults to 2.

    Returns:
        tuple: Contains information about the Zerobin and other OPA peak related information
            (int, ndarray, ndarray, ndarray, tuple)
        int: Zerobin
            an index representing the position of the interferometer, where the two pump pulses overlap (temporally).
        ndarray: Zero padded, offset corrected, rolled interferogram
            The zerobin is the 0th entry of the array and all values
            in front of the zerobin are shifted to the end of the array.

                * shape: 1D 
                * E.g.: (next_fast_len(interferogram.size*zero_pad_factor))

        ndarray: OPA pump-spectrum 
            obtained from the Fourier Transform of zeropadded interferogram.

                * shape: 1D 
                * E.g.: (zeropadded interferogram size/2)

        ndarray: Frequency axis (pump axis)

                * shape: 1D
                * E.g.: (zeropadded interferogram size/2)

        tuple: OPA pulse information

                * E.g. Indices/locations of the pump OPA pulse in the spectrum that was obtained
                  from the amplitude (absolute) of the FFT. This can be used to later get the phase at the location of the
                  zerobin and to plot the relevant part of the spectrum (OPA pump-pulse).

    References:

        |   Jan Helbing and Peter Hamm: Compact implementation of
            Fourier transform two-dimensional IR spectroscopy without phase
            ambiguity
        |   **For better understanding of Fourier Transformation see also:** 
        |   Werner Herres and Joern Gronholz: Understanding FT·IR Data Processing Part 1 - 3
    """
    # Correct offset of interferogram by subtracting average
    interferogram = interferogram - np.average(interferogram)
    # We need to save the original length of the interferogram, because
    # we zero pad twice in the algorithm
    interferogram_size = interferogram.size
    # ------- Preparation and Algorithm to find zerobin --------- Make
    # an initial guess for the zerobin. We know that it has to be in
    # proximity of the maximum of the interferogram Take the absolute of
    # the interferogram first, since the 'highest' amplitude could also
    # be negative.
    initial_zerobin_guess = np.argmax(np.abs(interferogram))
    # Checks which length the array should have for a more efficient FFT
    # (thats very handy)
    efficient_length_zerobin = next_fast_len(interferogram_size)
    # Pad zeros to ends of the interferogram array until it has the
    # "efficient length" for FFT. This generally increases FFT speed by
    # at least an order of magnitude.
    interferogram = np.pad(
        interferogram,
        (0, efficient_length_zerobin - interferogram.size),
        mode="constant",
        constant_values=0,
    )
    # Take the Fourier Transformation of the interferogram. We use rfft
    # because the interferogram exclusively contains real numbers. For a
    # sequence of real (as opposed to complex) numbers, the resulting
    # sequence of DFT coefficients, i. e. its spectrum, is mirror
    # symmetric around its central element. The rfft will only return
    # the first half of the spectrum and thus saves computational time.
    fft_interferogram = np.abs(rfft(interferogram))
    # Use find_opa_range to find the indices of the full width half
    # maximum (We are now in the frequency domain)
    _, _, opa_fwhm_range = find_opa_range(fft_interferogram)
    # Calculate frequency axis
    frequency_axis = calculate_frequency_axis(interferogram.size)
    # Find the zerobin
    zerobin = find_zerobin(
        interferogram, frequency_axis, initial_zerobin_guess, opa_fwhm_range
    )
    # ----- End of zerobin search -------

    # ------ Calculating results with zero pad factor ------
    # Increase size by zero_pad_factor to match the length of the zero
    # padded time domain data of the MCT detector We subtract the
    # zerobin position because the time domain data of the MCT is
    # truncated at the zerobin
    efficient_length = next_fast_len((interferogram_size - zerobin) * zero_pad_factor)
    # Pad zeros to ends of the interferogram array until it has the
    # "efficient length" for FFT.
    interferogram = np.pad(
        interferogram,
        (0, efficient_length - interferogram.size),
        mode="constant",
        constant_values=0,
    )
    # Shift the interferogram so that the zerobin is the 0th position.
    # The data prior/ in front of the zerobin data is shifted to the
    # back
    interferogram = np.roll(interferogram, -zerobin)
    # Take the Fourier Transformation of the interferogram. We use rfft
    # because the interferogram exclusively contains real numbers. For a
    # sequence of real (as opposed to complex) numbers, the resulting
    # sequence of DFT coefficients, i. e. its spectrum, is mirror
    # symmetric around its central element. The rfft will only return
    # the first half of the spectrum and thus saves computational time.
    fft_interferogram = rfft(interferogram)
    # Use find peaks function to get the peak of the OPA spectrum (We
    # are now in the frequency domain)
    opa_info = find_opa_range(np.abs(fft_interferogram))
    # Calculate new frequency axis from zero padded interferogram size
    frequency_axis = calculate_frequency_axis(interferogram.size)

    return zerobin, interferogram, fft_interferogram, frequency_axis, opa_info


def calculate_frequency_axis(
    interferogram_size: int, he_ne_wavelength: float = 632.8
) -> ndarray:
    """
    Calculates the frequency axis (pump axis) for a given size of a time domain interferogram.

    Args:
        interferogram (int): Size of the interferogram data array.
        he_ne_wavelength (float, optional): Wavelength of the He-Ne-Laser in nanometers that is used to
            keep track of the position of the moving interferometer arm. Defaults to 632.8 nm.

    Returns:
        ndarray: Frequency axis (pump axis).

    Notes:
        The resolution/spacing of the frequency domain information and
        its corresponding axis it given by:

        .. math::

            \\Delta \\nu = \\frac{1}{N \\Delta x}
        
        where N is the number of bins that were traversed by the moving arm
        of the interferometer and :math:`\Delta x` is the distance between two adjacent bins.

    References:
        |   Werner Herres and Joern Gronholz: Understanding FT·IR Data Processing Part 1 (p.2 equation 4)
    """
    # Calculate the total movement range which is possible with the stepsize of the He-Ne Laser for the given
    # interferogram
    total_movement_range = (
        he_ne_wavelength * 1e-7 * interferogram_size
    )  # interferometer path length in cm
    # For each step of the interferogram obtain the wavenumber.
    frequency_axis = np.arange(interferogram_size) / total_movement_range
    return frequency_axis


def find_opa_range(spectrum: ndarray):
    """
    Find characteristics of OPA pulse: center position (index), indices
    of peak, and indices of full width half maximum (fwhm).

    The algorithm first truncates the spectrum s.t. it starts at the
    first index. The 0th index is ignored because the Fourier Transform
    yields the sum of all values of the interferogram. This can lead to
    an "unnatural" peak at the 0th position that we are not interested
    in. The *scipy.signal.find_peaks* function is used to find all peaks
    that are higher than 80% of the maximum of the spectrum. Then the
    *scipy.signal.peak_widths* function is used to find the width and
    indices of our peak at 50% height (fwhm) and at 5% height (which we
    consider to be full width). Then arrays containing all indices
    within the range of fwhm and full width are generated.

    Note:
        When zero padding the time domain data alot, we sometimes
        observed that the *find_peaks* function finds several peaks
        instead of just one. These peaks are in close proximity to each
        other and probably are some kind of artifact. In this case, we
        choose the peak that returns the greater peak width.

    Args:
        spectrum (ndarray): OPA pump spectrum (generally obtained
            from Fourier Transform).
            
               * shape: (interferogram.size) size of the zeropadded interferogram

    Returns:
        tuple:
            Contains information about OPA pulse.
            center position (index), indices of peak,
            and indices of full width half maximum (fwhm).

        int: index of the of the OPA pulse maximum

        ndarray: indices locating the full width of the OPA pulse in spectrum. 
            
               * shape: 1D, depends on spectral width of the OPA pulse.

        ndarray: indices locating the fwhm of the OPA pulse in spectrum. 
            
               * shape: 1D, depends on spectral width of the OPA pulse.
    """
    # Since the rfft has the sum of all frequencies as its 0th value, we
    # cut that data point out This enables us to find the opa peak with
    # the find_peak function. If we leave this spike in, find_peaks does
    # not work anymore. Later we correct the index by adding +1 to the
    # opa_pulse_width and fwhm etc. This is necessary because we need
    # the accurate indices for plotting lateron.
    spectrum = spectrum[1:]

    # Find OPA peak. Note that the spectrum contains artifacts like
    # higher modes, thats why we require to only find peaks that are
    # higher than 80% of the maximum.
    opa_peak, _ = find_peaks(
        spectrum, height=0.8 * spectrum.max()
    )  # 0.8 to only return peaks that are higher than 80% of maximum

    # Obtain the full width half maximum of the pump OPA which is
    # necessary to find range where the phase is linear
    opa_pulse_fwhm = list(
        peak_widths(spectrum, opa_peak, rel_height=0.5)
    )  # We convert to list in case we need to overwrite values (see if clause)

    # Obtain full width at 5% height of peak to display full frequency
    # range of OPA.
    opa_pulse_width = list(
        peak_widths(spectrum, opa_peak, rel_height=0.95)
    )  # We convert to list in case we need to overwrite values (see if clause)

    # We encountered that sometimes due to a large zero padding factor
    # that two peaks, instead of one peak, are found which are very
    # close to each other. We fix this by choosing the peak that has the
    # broader width.
    if len(opa_peak) > 1:
        logger.warning(
            "The peak finding algorithm detected two peaks. We are trying to fix this, by choosing the peak with the broader width."
        )
        correct_peak = np.array(opa_pulse_fwhm[3] - opa_pulse_fwhm[2]).argmax()
        opa_peak = opa_peak[correct_peak]
        opa_pulse_fwhm[2] = opa_pulse_fwhm[2][correct_peak]
        opa_pulse_fwhm[3] = opa_pulse_fwhm[3][correct_peak]
        opa_pulse_width[2] = opa_pulse_width[2][correct_peak]
        opa_pulse_width[3] = opa_pulse_width[3][correct_peak]

    # Generate array containing all indices within OPA pulse width Add
    # one to the indices to correct for the fact that we left the 0th
    # value of the fourier transformed interferogram out.
    opa_pulse_indices = np.arange(int(opa_pulse_width[2]), int(opa_pulse_width[3])) + 1
    # Generate an array containing the indices of the values between the
    # fwhm's (where the OPA spectrum lies)
    opa_fwhm_range = np.arange(int(opa_pulse_fwhm[2]), int(opa_pulse_fwhm[3])) + 1

    return opa_peak, opa_pulse_indices, opa_fwhm_range


def calculate_phase_slope(
    interferogram: ndarray,
    frequency_axis: ndarray,
    zerobin_guess: int,
    opa_fwhm_range: ndarray,
):
    """
    Calculate the phase and its slope for a given zerobin guess of the
    interferogram.

    The data points in the interferogram before the zerobin guess are
    shifted to the end of the array prior to performing the Fourier
    Transform. The interferogram is now in the frequency domain and the
    phase can be calculated. A linear regression of the phase in between
    the FWHM of the OPA peak is performed to obtain the slope of the
    phase. To obtain the zerobin the phase should be as close to zero as
    possible.

    Args:
        interferogram (ndarray): Interferogram data. Contains a voltage for every bin measured by the pyro-detector.

                * shape: 1D 
                * E.g.: (interferogram.size)

        frequency_axis (ndarray): Frequency axis (pump axis). 

                * shape: 1D 
                * E.g.: (interferogram.size/2)

        zerobin_guess (int): Guess for the zerobin as index of the interferogram.
        opa_fwhm_range (ndarray): Indices for all data points in between the FWHM of the pump OPA pulse. 
            
               * shape: 1D

    Returns:
        float: Slope of the phase (from linear regression).
    """
    # Shift the interferogram so that the zerobin is the 0th position.
    # The data prior/ in front of the zerobin data is shifted to the
    # back
    interferogram = np.roll(interferogram, -zerobin_guess)
    # Fourier Transform We use rfft because the interferogram
    # exclusively contains real numbers. For a sequence of real (as
    # opposed to complex) numbers, the resulting sequence of DFT
    # coefficients, i. e. its spectrum, is mirror symmetric around its
    # central element. The rfft will only return the first half of the
    # spectrum and saves thus saves computational time.
    fft_interferogram = rfft(interferogram)
    # Calculate the Phase of the interferogram
    phase = np.unwrap(np.angle(fft_interferogram))
    # Make linear regression of the phase values in range of the OPA
    # spectrum
    regression_data = linregress(
        frequency_axis[opa_fwhm_range].flatten(), phase[opa_fwhm_range].flatten()
    )
    logger.info(
        "Regression for zerobin guess: {} yields: {}".format(
            zerobin_guess, regression_data
        )
    )
    return regression_data[0]


def find_zerobin(
    interferogram: ndarray,
    frequency_axis: ndarray,
    zerobin_guess: int,
    opa_fwhm_range: ndarray,
    slope=None,
):
    """
    Determines the zerobin.

    Calculates the slope of the phase and searches the zerobin by
    comparing the slopes. The zerobin guess is incremented by +1 if the
    slope is negative and by -1 if it is positive (see References). The
    algorithm checks whether the slope had a sign change to determine
    when the zerobin was found. Of the two bins where the sign change
    occured, the one which has a phase slope that is closer to 0 is
    chosen as zerobin.

    Args:
        interferogram (ndarray): Interferogram data. Contains a voltage for every bin measured by the pyro-detector.

                * shape: 1D 
                * E.g. (interferogram.size)

        frequency_axis (ndarray): Frequency axis (pump axis). 

                * shape: 1D 
                * E.g. (interferogram.size/2)

        zerobin_guess (int): Guess for the zerobin. Index of the interferogram.
        opa_fwhm_range (ndarray): Indices for all data points in between the FWHM of the pump OPA pulse. 
        
               * shape: 1D

    References:
        |   Jan Helbing and Peter Hamm: Compact implementation of Fourier transform two-dimensional IR spectroscopy without phase ambiguity
    """
    # Calculate the slope for a given zerobin guess if there was no slope that was provided.
    if slope is None:
        slope = calculate_phase_slope(
            interferogram, frequency_axis, zerobin_guess, opa_fwhm_range
        )
    # If the phase has a negative slope, the zerobin guess was chosen too small. Increment it by +1
    if slope < 0:
        new_zerobin_guess = zerobin_guess + 1
        logger.info(
            "Incrementing zerobin guess by one because slope ({}) is negative.".format(
                slope
            )
        )
    # If the phase has a positive slope, the zerobin guess was chosen too large. Increment it by -1
    elif slope > 0:
        new_zerobin_guess = zerobin_guess - 1
        logger.info(
            "Decrementing zerobin guess by one because slope ({}) is negative.".format(
                slope
            )
        )
    # If the slope is zero (which is unlikely since the world is not perfect) then the current guess is the
    # correct zerobin
    else:
        return zerobin_guess

    # Calculate the new slope after incrementing or decrementing
    new_slope = calculate_phase_slope(
        interferogram, frequency_axis, new_zerobin_guess, opa_fwhm_range
    )

    # Since we only have discrete positions/ bins we will probably not find the point where the phase is perfectly zero
    # Therefore, we check whether the slope changed its' sign after the incrementation (First if clause).
    # Then we have to pick the slope which is closer to zero (inner if clause)
    if (new_slope > 0 and slope < 0) or (new_slope < 0 and slope > 0):
        logger.info(
            """Detected a sign change of the slope.
 Now figuring out which slope is closer to zero. slope: {}, new_slope: {}""".format(
                slope, new_slope
            )
        )
        if abs(new_slope) < abs(slope):
            logger.info(
                "new slope is closer to zero. Returning zerobin: {}".format(
                    zerobin_guess
                )
            )
            return new_zerobin_guess
        else:
            logger.info(
                "old slope is closer to zero. Returning zerobin: {}".format(
                    zerobin_guess
                )
            )
            return zerobin_guess
    # If sign has not changed do another iteration
    else:
        return find_zerobin(
            interferogram,
            frequency_axis,
            new_zerobin_guess,
            opa_fwhm_range,
            slope=new_slope,
        )


# ---- Sort data into states
def sort_data(
    data: ndarray, states: ndarray, number_of_possible_states: ndarray
) -> tuple:
    """
    For each state, average the associated spectral data and return a
    (multidimensional) array holding those averaged data. Additionally,
    calculate base statistical information for each state and for each
    pixel. Also returns weights, which are the inverse of the variance,
    to use when averaging data from separate acquisitions (see
    references).

    This function takes a 2D array, containing an unsorted series of
    spectral data (spectral referring to the fact that the data is
    collected from the spectrometer + MCT).

    Each spectrum is labelled with the system's state variables (e.g.
    wobbler state, chopper state, polarizer state) via the associated
    states array. For each shot, the states array records the state of
    one or more state variables.

    The function then sorts and groups the data with identical states
    and averages them - these are repeat measurements. The individual
    component states (wobbler, chopper, etc.) serve as indices into this
    multidimensional array of results. _Crucially, this relies on the
    states being representable as integers that can serve as array
    indices.

    What is achieved here: we want to average the datapoints /
    lasershots that belong to the same state of the experiment. The ADC
    collects the data that we want to sort and average, and labels them
    also with the information in which state the system was at each
    recorded shot. In the simplest case the states array is 1D,
    representing just one discriminating factor, i.e. wobbler position.
    Our wobbler can be in four different positions, thus we would have a
    1D state 1D array, containing the wobbler state for each shot (in
    this case, a number out of 0, 1, 2 or 3). We now can average all
    laser shots with wobbler state 0 separately from wobbler state 1
    etc.

    Now imagine we have a wobbler and an interferometer in our setup. So
    we effectively have a 2D state array, with two numbers per laser
    shot. One row contains the position of the interferometer and the
    other row contains the wobbler state for each laser shot. Now we
    only want to average data at the same interferometer position and
    the same wobbler state. So we need to find the indices of all
    columns in the state array that are identical and then average the
    data at those indices. Each additional state variable would add
    another row to this 2D state array.

    This function represents a generalized method to achieve this
    grouping and averaging. By passing a data array and a state array
    that have both the same number of columns (i. e. shots), we can
    group corresponding data columns together.

    Args:
        data (ndarray): For frequency-domain data use normalized intensity (transmission).
            This can be calculated from the linearized ADC data by dividing the
            probe pixel by the reference pixels.
            For pump time-domain and probe frequency-domain data
            (non-normalized) intensity or ADC counts/voltage were
            used in LabView.

                * shape: 2D 
                * E.g.: (number_of_pixels_per_row, samples_to_acquire) or
                  (number_of_pixels, samples_to_acquire) respectively.

                Note:
                    Although the algorithm is able to sort the raw, non-normalized
                    data, one should always input transmission values for pure
                    frequency domain data. The reason for this is explained in:
                    
                    Brazard, J., Bizimana, L. A., & Turner, D. B. (2015). *Accurate*
                    *convergence of transient-absorption spectra using pulsed lasers.*
                    Review of Scientific Instruments, 86(5), 053106. Section 2 Part D.
                    
                Note:
                    When comparing the sorting of non-normalized intensities
                    to sorting normalized intensities/transmission for time-domain
                    pump and frequency-domain probe experiments (namely FT-2D-IR)
                    for the same raw data set there was no significant difference.
                    The difference between the two was at least two orders of magnitude
                    smaller than the resulting frequency-domain difference absorption
                    spectrum. There seems to be no reason against sorting
                    shot-to-shot normalized intensities (transmission). We believe
                    this should also generally lead to better data quality.
                        
                    **Caveat:** The raw data set this was tested on only contained
                    negative delays. We recommend investigating this in
                    more depth.

        states (ndarray): 2D array containing different parts of the state
            information for each laser shot in terms of integers:
            
                1.  first row: interferometer positions (0-65535),
                2.  second_row: chopper states (0-7),
                3.  third_row: wobbler states (0-3).
                
                * shape: 1D or 2D
                * E.g.: (number_of_state_monitors, samples_to_acquire)
            
                Note:
                    The state information needs to start at 0 and must not exceed
                    the corresponding value specified in number_of_possible_states.
                    I.e.: To achieve this for the interferometer position one
                    can subtract the minimum so that the lowest count is 0 and
                    one can obtain the values for number_of_possible_states by taking
                    the maximum and adding 1.

        number_of_possible_states (ndarray): For each row in states the number
            of possible states that this state monitor reaches.
                
                * shape: 1D (number_of_state_monitors)
                * E.g.: np.array([65536, 8, 4]) (taking the example from states docstring).

    Returns:
        tuple: Contains information for each state.
            Contains sorted and averaged data for each state, 
            weights, counts and statistical data

        ndarray: sorted and averaged data.
                * shape: Multidimensional e.g. (number_of_pixels_per_row, number of interferometer positions,
                  number of chopper states, number of wobbler states)

        ndarray: weights (inverse variance) for each state to use when averaging two
            different scans/acquisitions.

                * shape: Multidimensional, same size as sorted_data 
                * I.e. (number_of_pixels_per_row, number of interferometer positions, number of chopper states,
                  number of wobbler states)

        ndarray: counter: how often a given state was found in the unsorted data. This can be
            used for trouble shooting to see if all states were 'hit'.
                
                * shape: Multidimensional - one dimension less than the averaged data.
                * E.g.: (number of interferometer positions, number of chopper states, number of wobbler states)
            
        tuple of pd.DataFrame: base statistical data: 
            |   **(variance, mean_state_std, mean_state_std_std)**
            
                1.  **variance:** the variance for all states and all pixels/transmission values
                2.  **mean_state_std:** mean standard deviation of all states for all pixels, this indicates
                    how much the laser intensity fluctuated during the acquisition for a given pixel.
                    Use this to visualize the fluctuations of the laser.
                3.  **std_state_std:** standard deviation of standard deviation of all states for all pixels,
                    this indicates how much the fluctuation of the laser intensity varied with the
                    states. Use this as errorbars when plotting the fluctuations of the laser.

    Note:
        The computational time needed for this algorithm to run depends
        largely on the size of the data set (samples_to_acquire) and
        also on the total number of possible states. It is recommended
        to reduce this number as far as possible. This is, for instance,
        very relevant for the interferometer positions since we only
        move the interferometer in a small range. If the counter of the
        interferometer has been reset appropriately the maximum count we
        are going to record will be approximately 4000. So the function
        will run a lot faster if we provide number_of_possible_states
        with 4001 (yes, 4000+1) instead of 65536. Also make sure you
        have enough RAM in the computer. We encountered a sudden
        (unproportional) increase in runtime when increasing the size of
        the data. We could trace this back to the RAM that python needs.

    References:
        |   Brazard, J., Bizimana, L. A., & Turner, D. B. (2015). Accurate convergence of transient-absorption spectra
            using pulsed lasers. Review of Scientific Instruments, 86(5), 053106.
        |   https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Variance_weights
    """
    #! Doesn't it make more sense to average out the repeated points
    #! directly? Idea to reduce computation time: generally reduce
    #! number_of_possible_states (looking at you signalsize) ... this
    #! could be done be subtracting interferometer_positions.min() from
    #! interferometer_positions. The number of possible states is then
    #! given by interferometer_positions.max() One should generally note
    #! that the speed of algorithm severely depends on the
    #! number_of_possible states and the number of total data points.
    # Load data into pandas Dataframe to be able to sort them. We need to
    # use pandas because numpy does not natively support group by
    # operations.
    df = pd.DataFrame(data.T, columns=[str(i) for i in range(data.shape[0])])

    # If we have state vectors we need convert them to a scalar
    # information because pandas only supports 2D arrays.
    if number_of_possible_states.size > 1:
        # Calculate the total number of different states that setup can
        # "produce" We need this to convert our state vector into a
        # state scalar.
        total_number_of_states = number_of_possible_states.prod()
        # Convert state "vectors" for each shot into a scalar using a
        # bijective map This is achieved using ravel_multi_index. For an
        # in depth explanation see:
        # https://stackoverflow.com/questions/38674027/find-the-row-indexes-of-several-values-in-a-numpy-array
        # The nice thing about this is when
        # .reshape(number_of_possible_states) is used on an array that
        # contains data sorted/indexed by states_1d, the order will be
        # such that we can index with the (multidimensional) states
        # array. Have a look at the end of this function. We set mode to
        # "clip" to handle indices that are out of range. An index that
        # is above the range will be clipped to the highest 1d-index. A
        # negative index would be clipped to 0. This also implies that
        # when number_of_possible_states is not specified properly data
        # is essentially lost.
        states_1d = np.ravel_multi_index(states, number_of_possible_states, mode="clip")
    # If the state is already a scalar information there is no need to
    # convert it.
    else:
        states_1d = states
        total_number_of_states = number_of_possible_states

    # Add states_1d to data frame
    df["state"] = states_1d
    # Group data by state, we can then use nice methods like .mean() and
    # .std() Turning off sorting of states makes group by method faster,
    # but it makes the sorting into the numpy array slower. this is the
    # slow and memory intensive step
    grouped_data = df.groupby("state", sort=True)

    # Count how often a given state was observed. We use np.bincount
    # instead of the grouped_data.count() because it is faster.
    # (grouped_data.count() counts for each pixel, but the count for
    # each pixel in a given state is identical so pandas is effectively
    # doing too much work.)
    counts = np.bincount(states_1d, minlength=total_number_of_states)

    # Calculate the average for each pixel for each state
    averaged_data = grouped_data.mean()

    # ---- Calculate the variance for each state to use as weights.
    # Using the inverse of the variance for each state as weights to
    # average data of different acquisitions yields the maximum
    # likelihood estimation of the mean for independently and normally
    # distributed variables. This implies that the average that we
    # calculate with these weights has highest probability to be the
    # 'correct' mean (expected value).
    # See: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Variance_weights
    # In VB6 ddof (degrees of freedom) was chosen to be 0 which means
    # that we divide by the number of samples. We believe that the
    # correct (unbiased) sample variance should be computed with ddof=1
    # which equates to dividing by number of samples - 1. Obviously it
    # is not possible to calculate the variance of one value. In this
    # case the pandas method returns NaN which we will later on replace
    # with np.inf. This results in the weights being 1/inf = 0 if a
    # state was only measured once. Make sure to record every state
    # several times!
    variance = grouped_data.var(ddof=1)

    # ---- Get statistical information
    # Calculate the mean standard deviation over all states for each
    # transmission/normalized intensity, we do this to later visualize
    # the fluctuations of the laser. Note that these values are only
    # used for visualisation purposes and are not needed for the actual
    # processing. We take the sqrt of the variance because it is much
    # faster than calling grouped_data.std() and it yielded exactly the
    # same results in our testing.
    std = variance.pow(1 / 2)
    mean_state_std = std.mean()
    # Calculate the standard deviation of the standard deviation to use
    # as errorbars on our plot for the laser fluctuations. That way we
    # can see how much the flucations differ over the states.
    std_state_std = variance.std(ddof=1)

    # Create tuple holding all statistical information
    statistics = (variance, mean_state_std, std_state_std)

    # ---- Sort data from pandas DataFrame into numpy ndarray Get the
    # order of the 1d states. This is required, because not forcibly all
    # states will have been hit/sampled, during the acquistion. So we
    # might have "empty rows" in our numpy array.
    indices_1d = averaged_data.index.to_numpy()

    # Pre allocate 2D numpy array in which to write data.
    sorted_data = np.zeros((data.shape[0], number_of_possible_states.prod()))
    # Pre allocate 2D numpy array in which to write weights.
    weights = np.zeros((data.shape[0], number_of_possible_states.prod()))

    # -- Sort data, weights into numpy array
    sorted_data[:, indices_1d] = averaged_data.to_numpy().T

    # Replace NaNs of variance array with np.inf then convert to numpy
    # array and calculate inverse.
    weights[:, indices_1d] = 1 / variance.fillna(np.inf).to_numpy().T
    # Quick note: Why do we not convert the variance data frame into a
    # numpy array? Because there is no plausible way to denote variances
    # that could not be calculated.

    # --- Reshape 2D arrays to have correct dimensions.
    # We only need to reshape sorted data and weights into a
    # multidimensional array if we have more than one state information
    # (state vector instead of state scalar).
    if number_of_possible_states.size > 1:
        sorted_data = sorted_data.reshape(
            (sorted_data.shape[0], *number_of_possible_states)
        )
        weights = weights.reshape((weights.shape[0], *number_of_possible_states))
        counts = counts.reshape(number_of_possible_states)

    return sorted_data, weights, counts, statistics


# Calculate shot to shot signals
def shot_to_shot_signal(transmission: ndarray, chopper_state: int = None):
    """
    Calculates the difference absorption spectra for the adjacent
    samples (shots) in the data and extracts statistical data.

    Args:
        transmission (ndarray): transmisson or relative intensity
            (probe/ref array) for each pixel from which difference
            signal should be calculated.

                * shape: 2D 
                * E.g. (number of pixels per row, samples to acquire)

        chopper_state (int, optional): 1 if the 0th sample (column) in
            the array corresponds to data from a pumped state.

            0 if the
            0th sample (column) in the array corresponds to data from an
            unpumped state. For data where chopper was not running
            but a "pseudo signal" should be calculated this argument
            does not need to be specified. Defaults to None.

    Returns:
        tuple: Tuple with data from difference absorption spectra
            Containing Averaged shot-to-shot signal, amplitude,
            standard deviation and average standard deviation of signal

            ndarray: Averaged shot-to-shot signal 
                * shape: 1D (number of pixels per row)

            float: Amplitude
                Amplitude of the the averaged shot-to-shot
                signal calculated by subtracting the minimum from the maximum.

            ndarray: Standard deviation of shot-to-shot signal 
                * shape: 1D (number of pixels per row)

            float: Average standard deviation of shot-to-shot signal
                standard deviation of shot-to-shot signal averaged over all
                pixels. This is what was referred to as mean noise.
    """
    # Calculate the absorption for every laser shot Because the
    # shot-to-shot difference signal can only be calculated for an even
    # number of samples truncate array if it has uneven size in the
    # first axis
    if transmission.shape[1] % 2:
        absorption = -np.log10(transmission[:, :-1])
    else:
        absorption = -np.log10(transmission)

    # Calculate the difference signal (absorption) between adjacent
    # pumped and unpumped shots
    signal = absorption[:, 1::2] - absorption[:, ::2]
    # Flip sign of signal if the 0th shot instead of the first shot was
    # pumped
    if chopper_state == 1:
        signal *= -1

    # Calculate standard deviation of shot to shot signal
    std_signal = np.nanstd(signal, axis=1, ddof=1)

    # Calculate the average standard deviation of the signal over all
    # pixels (this is what was used the be referred to as "mean noise")
    avg_std_signal = np.nanmean(std_signal)  # mean noise

    # Calculate the average of the shot to shot signal
    avg_signal = np.nanmean(signal, axis=1)
    # From this calculate the average amplitude of the shot to shot
    # signal
    amplitude = np.nanmax(avg_signal) - np.nanmin(avg_signal)

    return avg_signal, amplitude, std_signal, avg_std_signal


def shot_to_shot_viper(
    transmission: ndarray, vis_chopper_state: int, ir_chopper_state: ndarray
):
    """
    Calculates the VIPER difference absorption spectra for 4 adjacent
    samples (shots) in the data and extracts statistical data.

    Args:
        transmission (ndarray): transmission or relative intensity
            (probe/ref array) for each pixel from which difference
            signal should be calculated.
                
                * shape: 2D 
                * E.g. (number of pixels per row, samples to acquire)

        vis_chopper_state (int):
            1 if the 0th sample (column) in
            the array corresponds to data from a UV/VIS pumped state.

            0 if the 0th sample (column) in the array corresponds to
            data from an UV/VIS unpumped state. This assumes that the
            UV/VIS Chopper runs at half of the laser repitition rate.

        ir_chopper_state (ndarray): 
            * **[1, 1]** if the 0th and 1st sample (column)
              in the array corresponds to data from two IR pumped states.
            * **[0, 1]** if the 0th sample (column) corresponds to an IR unpumped
              state while the 1st sample (column) corresponds to an IR pumped state.
            * **[1, 0]** if the 0th sample (column) corresponds to an IR pumped state
              while the 1st sample (column) corresponds to an IR unpumped state.
            * **[0, 0]** if the 0th and 1st sample (column) in the array
              corresponds to data from two IR unpumped states.

    Returns:
        tuple:
            Data from VIPER difference absorption spectrum.
            Containing Averaged shot-to-shot VIPER signal, amplitude,
            standard deviation and average standard deviation of VIPER signal
 
        averaged shot-to-shot VIPER signal (ndarray):
            * shape: 1D (number of pixels per row)

        amplitude (float):
            Amplitude of the the averaged shot-to-shot
            VIPER signal calculated by subtracting the minimum from
            the maximum.
 
        standard deviation of shot-to-shot VIPER signal (ndarray):
            * shape: 1D (number of pixels per row)

        average standard deviation of shot-to-shot VIPER signal (float):
            standard deviation of shot-to-shot signal averaged over all
            pixels. This is what was reffered to as mean noise.
    """
    # Calculate the absorption for every laser shot Because the
    # shot-to-shot difference signal of VIPER can only be calculated for
    # a number of samples that is divisible by 4 truncate array if it
    # has the wrong size in the first axis
    upperlimit = (transmission.shape[1] // 4) * 4
    # a = absorption (orthwise lines will be too long)
    a = -np.log10(transmission[:, :upperlimit])

    # There are 8 different initial chopper states we can observe We can
    # view the chopper states that are passed here as binary numbers and
    # transform them into decimal format. This transformation is unique.
    state = 4 * vis_chopper_state + 2 * ir_chopper_state[0] + ir_chopper_state[1]

    # Calculate the difference signal using 4 adjacent shots
    # Beginners notes:
    # (IR off, UV off): Background (B) = [:, 0, 0]
    # (IR off, UV on): TRIR + Background (T) = [:, 0, 1]
    # (IR on, UV off): IR pump/IR probe + Background (I) = [:, 1, 0]
    # (IR on, UV on): VIPER + TRIR + IR pump/IR probe + Background (V) = [:, 1, 1]
    if state == 0:
        # state = 0 (0b000) implies:
        # uv chopper:   0    1    0    1
        # ir chopper:   0    0    1    1
        # information:  B    T    I    V
        signal = a[:, 3::4] - a[:, 2::4] - a[:, 1::4] + a[:, 0::4]
    elif state == 1:
        # state = 1 (0b001) implies:
        # uv chopper:   0    1    0    1
        # ir chopper:   0    1    1    0
        # information:  B    V    I    T
        signal = a[:, 1::4] - a[:, 2::4] - a[:, 3::4] + a[:, 0::4]
    elif state == 2:
        # state = 2 (0b010) implies:
        # uv chopper:   0    1    0    1
        # ir chopper:   1    0    0    1
        # information:  I    T    B    V
        signal = a[:, 3::4] - a[:, 0::4] - a[:, 1::4] + a[:, 2::4]
    elif state == 3:
        # state = 2 (0b011) implies:
        # uv chopper:   0    1    0    1
        # ir chopper:   1    1    0    0
        # information:  I    V    B    T
        signal = a[:, 1::4] - a[:, 0::4] - a[:, 3::4] + a[:, 2::4]
    elif state == 4:
        # state = 4 (0b100) implies:
        # uv chopper:   1    0    1    0
        # ir chopper:   0    0    1    1
        # information:  T    B    V    I
        signal = a[:, 2::4] - a[:, 3::4] - a[:, 0::4] + a[:, 1::4]
    elif state == 5:
        # state = 5 (0b101) implies:
        # uv chopper:   1    0    1    0
        # ir chopper:   0    1    1    0
        # information:  T    I    V    B
        signal = a[:, 2::4] - a[:, 1::4] - a[:, 0::4] + a[:, 3::4]
    elif state == 6:
        # state = 6 (0b110) implies:
        # uv chopper:   1    0    1    0
        # ir chopper:   1    0    0    1
        # information:  V    B    T    I
        signal = a[:, 0::4] - a[:, 3::4] - a[:, 2::4] + a[:, 1::4]
    elif state == 7:
        # state = 7 (0b111) implies:
        # uv chopper:   1    0    1    0
        # ir chopper:   1    1    0    0
        # information:  V    I    T    B
        signal = a[:, 0::4] - a[:, 1::4] - a[:, 2::4] + a[:, 3::4]

    # Calculate standard deviation of shot to shot signal
    std_signal = np.nanstd(signal, axis=1, ddof=1)

    # Calculate the average standard deviation of the signal over all
    # pixels (this is what was used the be referred to as "mean noise")
    avg_std_signal = np.nanmean(std_signal)  # mean noise

    # Calculate the average of the shot to shot signal
    avg_signal = np.nanmean(signal, axis=1)
    # From this calculate the average amplitude of the shot to shot
    # signal
    amplitude = np.nanmax(avg_signal) - np.nanmin(avg_signal)

    return avg_signal, amplitude, std_signal, avg_std_signal


# --------------- Visualisation processing for pyqtGraph --------------------
def generate_img_data(x_axis: ndarray, y_axis: ndarray, data: ndarray):
    """
    Generates data for heatmap/imshow plot in pyqtgraph.

    The returned data set is linearly interpolated to
    have equal spacing between "real" data points.
    Because pyqtgraph is not able to plot on unevenly
    spaced axes. (In matplotlib there exists a function
    called pcolormesh, just FYI. May later come also to
    pyqtgraph, see references.)

    Args:
        x_axis (ndarray): x-axis points (columns) of the data set
        y_axis (ndarray): y-axis points (rows) of the data set
        data (ndarray): data set that will be transformed into
            data with respectively equally spaced x- and y-axis
            data points using linear interpolation.
                
                * shape: 2D 
                * E.g.: (pixels, delays)

    References:
        |   https://github.com/pyqtgraph/pyqtgraph/issues/1262
        |   https://github.com/pyqtgraph/pyqtgraph/pull/1273

    Returns:
        ndarray: interpolated data set with x- and y-axis
            that are equally spaced respectively.
                
                * shape: 2D 
                * E.g.: (> pixels, > delays)
    """
    # Calculate image size (pixels in each dimension)
    # then generate the corresponding (interpolated) axes
    # x-axis
    x_size = int(
        round((x_axis[-1] - x_axis[0]) / np.diff(x_axis).min())
    )  # x_axis needs to be monotonically increasing
    # Limit the resolution
    x_size = min(1080, x_size)
    interpol_x_axis = np.linspace(x_axis[0], x_axis[-1], x_size)
    # y-axis
    y_size = int(
        round((y_axis[-1] - y_axis[0]) / np.diff(y_axis).min())
    )  # y_axis needs to be monotonically increasing
    # Limit the resolution
    y_size = min(1080, y_size)
    interpol_y_axis = np.linspace(y_axis[0], y_axis[-1], y_size)

    # build the (linear) interpolation function
    # that will be used to generate equally spaced image data
    lin_interpol = RectBivariateSpline(y_axis, x_axis, data, kx=1, ky=1)

    # Generate the equally spaced image data set (from interpolation function)
    interpol_data = lin_interpol(interpol_y_axis, interpol_x_axis)
    return interpol_data


def scale_img(x_axis: ndarray, y_axis: ndarray, data: ndarray, img: pg.ImageItem):
    """
    Makes it such that the axes of the image are correctly displayed.

    This is achieved by translating / moving the image to the point of the
    0 th entries of the x- and y-axis and then scaling it accordingly.

    Args:
        x_axis (ndarray): x-axis points (columns) of the data set
            (it does not matter whether this is the interpolated
            or non-interpolated axis)
        y_axis (ndarray): y-axis points (rows) of the data set
            (it does not matter whether this is the interpolated
            or non-interpolated axis)
        data (ndarray): interpolated data set that was used
            to generate the ImageItem
            
                * shape: 2D

        img (pg.ImageItem): ImageItem that is supposed to be adjusted.
    """
    img.translate(x_axis[0], y_axis[0])
    img.scale(
        (x_axis[-1] - x_axis[0]) / data.shape[1],
        (y_axis[-1] - y_axis[0]) / data.shape[0],
    )


def generate_contour_lines(data: ndarray, img: pg.ImageItem, contour_levels: int = 10):
    """
    Generates contour lines for given data.

    |   Sets positive height contour lines to black solid line.
    |   Sets negative height contour lines to black dashed line.

    Args:
        data (ndarray): Data from which to generate contour lines.
             
                * shape: 2D

        img (pg.ImageItem): ImageItem to which contour lines will be
            "attached"/overlayed.
        contour_levels (int, optional): Number of contour levels to generate.
            Defaults to 10.

    Returns:
        list: List holding references to contourline objects
            (IsocurveItem). Pass this to the update_contour_lines
            function to update them.
    """
    # Set positive level contour to solid line
    pos_pen = pg.mkPen(color="k", width=1, style=QtCore.Qt.SolidLine)
    # Set negative level contour to dashed line
    neg_pen = pg.mkPen(color="k", width=1, style=QtCore.Qt.DashLine)

    # Preallocate list hold IsocurveItem objects
    contour_lines = []
    # Select height / levels at which to display contour lines
    levels = np.linspace(data.min(), data.max(), contour_levels)
    for i in range(len(levels)):
        # Get height/ level for current contour line
        v = levels[i]
        # Make contour line solid/ dashed
        if v >= 0:
            pen = pos_pen
        else:
            pen = neg_pen

        # generate isocurve
        c = pg.IsocurveItem(level=v, pen=pen)
        # Make sure contour lines are displayed
        # over heatmap
        c.setParentItem(img)
        c.setZValue(10)
        #
        c.setData(data)
        # Add contour line to list
        contour_lines.append(c)

    return contour_lines


def update_contour_lines(data: ndarray, contour_lines: list):
    """
    Update contour lines given the new data set.

    Args:
        data (ndarray): Data from which to generate contour lines.
            
                * shape: 2D

        contour_lines (list): List of IsocurveItem references.
    """
    # Set positive level contour to solid line
    pos_pen = pg.mkPen(color="k", width=1, style=QtCore.Qt.SolidLine)
    # Set negative level contour to dashed line
    neg_pen = pg.mkPen(color="k", width=1, style=QtCore.Qt.DashLine)

    # Get the number of contour levels
    contour_levels = len(contour_lines)
    # Select height / levels at which to display contour lines
    levels = np.linspace(data.min(), data.max(), contour_levels)
    for i, c in enumerate(contour_lines):
        # Get height/ level for current contour line
        v = levels[i]

        # Update data
        c.setData(data, level=v)

        # Update pen
        # Make contour line solid/ dashed
        if v >= 0:
            pen = pos_pen
        else:
            pen = neg_pen

        c.setPen(pen)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # prl = PixelResponseLinearization("/Users/arthun/Documents/Uni/Masterarbeit/Messsoftware/akb_software/hardware_config_files/h-lab_averaged_pixel_linearization_fit_parameters.json")
    # adc_counts = np.arange(0xFFFF)
    # test = prl.linearize(adc_counts)

    # plt.plot(adc_counts, test)
    # plt.grid()
    # plt.show()

    # Test sort data in 2 chopper szenario
    roll1 = np.random.randint(4)
    roll2 = np.random.randint(4)
    n_spectra = 10
    pixels = 32
    absorption = np.tile(np.tile(np.arange(1, 5), n_spectra), pixels).reshape(
        pixels, 4 * n_spectra
    )
    ir_chop = np.tile(np.roll(np.array([0, 0, 1, 1]), roll1), n_spectra)
    uv_chop = np.tile(np.roll(np.array([1, 0, 1, 0]), roll2), n_spectra)

    n_states = np.array([2, 2])
    states = np.vstack((ir_chop, uv_chop))

    d, w, c, s = sort_data(absorption, states, n_states)
