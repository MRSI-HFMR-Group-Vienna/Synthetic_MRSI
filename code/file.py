## just for type checking, to solve circular imports ##
from __future__ import annotations
from cupyx.scipy.ndimage import zoom as zoom_gpu
import spectral_spatial_simulation
from spatial_simulation import ParameterMap, ParameterVolume
from typing_extensions import Self
from tools import JsonConverter, UnitTools, JupyterPlotManager, SpaceEstimator, ArrayTools

# For loading / processing MRS data:
from fsl_mrs.utils.mrs_io import jmrui_io, read_basis
from spec2nii import jmrui

from configurator import Configurator
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from tools import deprecated, ArrayTools, PathTools, DictionaryTools
from scipy.io import loadmat
from prettyconsole import Console
from pathlib import Path
import nibabel as nib
import numpy as np
import cupy as cp
import pint
import h5py
import json
import sys
import os
from typing import TYPE_CHECKING

from interface import WorkingSource, Plot
from spectral_spatial_simulation import FID as SpectralSpatialFID

# Just for type annotations
if TYPE_CHECKING:
    from sampling_simulation import CoilSensitivityVolume as SamplingCoilSensitivityVolume


# For enabling to use units
u = pint.UnitRegistry()

class BasisSet:

    def __init__(self, path: str, data_precision: int = 32, extension:str = None, verbose: bool = True):
        """
        The constructor. It checks if path to one folder or direct to one file is given.

        (!) None: If path is to folder and files with mixed extensions are inside then error message will raise.
                  Then, the solution is to define the 'extension' here (when creating these instance)!

        :param path: the path to folder or direct to one file
        :param data_precision: the precision, e.g. 32 yields float32 and complex64
        :param extension: Only important if path to one folder is given
        :param verbose: just if output to the console should be printed
        """

        # Here the configurator is not used. It is better to stay generic with the path.
        # Check whether path to one file is directly given of to folder containing multiple files
        self.path = PathTools.collect_files(path=path, extension=extension, verbose=False)

        if self.path is None:
            Console.printf("error", f"Could not find any files or mixed files are available. "
                                    f"If files with mixed extensions in the folder try to specify 'extension' here.")
            return

        self.data_precision = data_precision


        # The data precision is later handled to tools.ArrayTools.to_precision
        # -> Only affects the moment the time vector and signal
        # -> if precision 32 is given then complex yields precision 64 in total since "two arrays of signal with real and imaginary"
        self.data_precision: int = data_precision

        # Just if any output should be printed to the console
        self.verbose = verbose

        # All at the moment supported file types
        self.file_extensions = [".mrui", ".txt", ".m", ".basis"]

    def load(self, subset_names: list[str] = None) -> dict:
        """
        To automatically decide based on the file type which method should be called. The called methods
        are returning standardised output.

        :return: dictionary with parameters, signal, time
        """

        # Get first file type (extension) and check if supported
        path = self.path
        suffix = path[0].suffix.lower() # get suffix of fist file (and if only one file, then just one in the list)

        if suffix not in self.file_extensions:
            Console.printf("error", f"Invalid file extension: {suffix}. Only possible: {self.file_extensions}")
            return

        # Case 1: jmrui derivate
        if suffix in [".mrui", ".txt", ".m"]:
            data: dict = self._load_jmrui_file(suffix=suffix, verbose=self.verbose)

        # Case 2: basis set
        elif suffix in [".basis"]:
            data: dict = self._load_lcmodel_basis_set(verbose=self.verbose)

        else: # TODO: Maybe will never be reached!
            Console.printf("error", f"Invalid file extension: {suffix}. Only possible: {self.file_extensions}")
            return

        # Check if subset of the loaded data (specific metabolites by name) are requested!
        if subset_names is not None:
            data = self.get_subset_by_name(data=data, compound_name=subset_names, verbose=self.verbose)

        Console.printf("success", f"Loaded FID signals of {len(data['name'])} chemical compounds! \n"
                                  f"  names: {data['name']}", mute=not self.verbose)
        return data

    def write(self):
        # TODO: Write to NIFTI MRS?
        raise NotImplementedError

    def get_subset_by_name(self, data: dict, compound_name: str | list[str], verbose: bool=False) -> dict:
        """
        To get a subset of the data dictionary based on one or multiple compound names.

        :param verbose: if True the print ot console
        :param data: the dictionary holding the data of all chemical compounds and parameters
        :param compound_name: one compound name as string or a list of compound names
        :return: dictionary with only subset
        """

        # Check if compound name is of type string
        if isinstance(compound_name, str):
            compound_name = [compound_name]

        # Give error if no names provided
        if len(compound_name) == 0:
            Console.printf("Error", "No compound name(s) provided", verbose=verbose)
            return

        # Give error if any required key is missing in the dictionary
        required_keys = ['parameters', 'signal', 'time', 'name']
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            Console.printf("Error", f"Key(s) missing in data: {missing_keys}", verbose=verbose)
            return

        # Find which of the requested names exist and which do not:
        #  -> 'found' keeps the original order of compound_name
        #  -> 'missing' is just for the warning message
        found = [n for n in compound_name if n in data['name']]
        missing = [n for n in compound_name if n not in data['name']]

        # Give error if not a single requested name was found
        if len(found) == 0:
            Console.printf("Error", f"None of the requested name(s) found: {compound_name}", verbose=verbose)
            return

        # Just a warning if some names are missing but at least one is found
        if missing:
            Console.printf("Warning", f"Name(s) not found and skipped: {missing}", verbose=verbose)

        # Get all the indices of the found names and then build the subset:
        #  -> select the respective rows from the signal array,
        #  -> keep the names in the same order as found,
        #  -> copy parameters and time as they are shared across all signals
        indices = [data['name'].index(n) for n in found]
        subset = {
            'parameters': data['parameters'],
            'signal': data['signal'][indices, :],
            'time': data['time'],
            'name': [data['name'][i] for i in indices],
        }

        return subset

    def _check_jmrui_common_parameters(self, loaded_data, verbose=False):
        """
        To check if all jMRUI files have same parameters needed for one output.
        Files with different parameters are skipped.

        :param loaded_data: list with dicts containing parameters and signal
        :param verbose: if it should be printed to the chat or not.
        :return: list with filtered data, common dwell time, spectral frequency and nucleus
        """
        reference = loaded_data[0]
        checked_data = [reference]
        skipped_files = []

        for data in loaded_data[1:]:
            skip_reasons = []

            # Same signal size and dwell time create the same time vector
            if data['signal'].size != reference['signal'].size:
                skip_reasons.append("signal size")

            if not np.isclose(
                    data['dwell_time'].to(u.s).magnitude,
                    reference['dwell_time'].to(u.s).magnitude
            ):
                skip_reasons.append("dwell time")

            if not np.isclose(
                    data['spectral_frequency'].to(u.hertz).magnitude,
                    reference['spectral_frequency'].to(u.hertz).magnitude
            ):
                skip_reasons.append("spectral frequency")

            if data['nucleus'] != reference['nucleus']:
                skip_reasons.append("nucleus")

            if skip_reasons:
                skipped_files.append(f"  - {data['path'].name}: {', '.join(skip_reasons)}")
            else:
                checked_data.append(data)

        # Print collected skipped files
        if skipped_files:
            Console.printf(
                "warning",
                f"Skipped jMRUI files because following parameters do not match {reference['path'].name}:\n"
                + "\n".join(skipped_files),
                mute=not verbose
            )

        return checked_data, reference['dwell_time'], reference['spectral_frequency'], reference['nucleus']


    def _load_jmrui_file(self, suffix, verbose=False) -> dict:
        """
        All files are derived from jMRUI.
        For automatic handling the methods:
            * Reading .mrui file: self._read_jmrui_binary  (one metabolite per file)
            * Reading .txt file:  self._read_jmrui_txt     (one metabolite per file)
            * Reading .m file:    self._read_jmrui_m       (already multiple metabolites in one file)

        For .mrui and .txt each file holds one metabolite. This method calls the matching
        single-file reader for every file in self.path, then checks if all files share the
        same parameters (dwell time, spectral frequency, ...) and stacks them to one signal
        array of shape (metabolites, signal).

        The .m case is structurally different (multi-metabolite already in one file), so it
        is just delegated and does NOT run through the stacking pipeline here.

        :return: dictionary holding the data {parameters: (...), signal: (...), time: (...)}
        """

        # Case 1: .m file -> all metabolites already in one file, just delegate
        if suffix == ".m":
            return self._load_jmrui_m(verbose=verbose)

        # Case 2: .mrui / .txt -> pick the matching single-file reader
        if suffix == ".mrui":
            single_file_reader = self._load_jmrui_binary
        elif suffix == ".txt":
            single_file_reader = self._load_jmrui_txt
        else:
            Console.printf("error", f"Invalid file extension: {suffix}. Only possible: {self.file_extensions}")
            return

        # Read each file individually (one metabolite per file)
        # -> each returns an intermediate dict (path, signal, dwell_time, spectral_frequency, nucleus)
        loaded_data = []
        for path in self.path:
            loaded_data.append(single_file_reader(path=path))

        # Check if all files can be used as one metabolite list
        loaded_data, dwell_time, spectral_frequency, nucleus = self._check_jmrui_common_parameters(
            loaded_data=loaded_data,
            verbose=verbose
        )

        # Create the time vector
        time_vector = (np.arange(loaded_data[0]['signal'].size) * dwell_time).to(u.s)

        # Stack metabolites to desired shape: (metabolites, signal)
        signal = np.stack([data['signal'].magnitude for data in loaded_data], axis=0) * u.dimensionless
        names = [data['path'].stem for data in loaded_data]

        # Convert to desired data type
        signal_dtype_before = signal.dtype
        time_vector_dtype_before = time_vector.dtype

        signal = ArrayTools.to_precision(array=signal, precision=self.data_precision, verbose=False)
        time_vector = ArrayTools.to_precision(array=time_vector, precision=self.data_precision, verbose=False)
        #  Just for printing conversation

        w = max(len(str(signal_dtype_before)), len(str(time_vector_dtype_before)))
        Console.add_lines(f"      signal: {str(signal_dtype_before):<{w}} -> {signal.dtype}")
        Console.add_lines(f"      time:   {str(time_vector_dtype_before):<{w}} -> {time_vector.dtype}")
#        Console.printf("info", f"Changed the data type of the FID signal(s): \n"
#                               f"      signal: {str(signal_dtype_before):<{w}} -> {signal.dtype} \n"
#                               f"      time:   {str(time_vector_dtype_before):<{w}} -> {time_vector.dtype}")

        # (!) Uniform formatted output (as other similar methods here)
        return {
            'parameters': {
                'dwell_time': dwell_time,                   # in sec
                'spectral_frequency': spectral_frequency,   # in Hz
                'nucleus': nucleus,                         # jMRUI files do not define a reliable nucleus string
            },
            'signal': signal,                               # complex, dimensionless, shape: (metabolites, signal)
            'time': time_vector,                            # in sec
            'name': names                                   # based on file names (usually e.g., NAA.mrui / NAA.txt)
        }

    def _load_jmrui_txt(self, path):
        """
        To load metabolite FID signal and parameters from ONE jMRUI .txt-file.

        The multi-file handling (looping over self.path, checking common parameters,
        stacking signals, precision conversion) is done in self._read_jmrui_file.

        :param path: path to the single .txt file to read
        :return: intermediate dict (path, signal, dwell_time, spectral_frequency, nucleus)
        """

        # Get the signal and header data
        signal, header = jmrui_io.readjMRUItxt(path)
        signal = np.asarray(signal).squeeze() * u.dimensionless

        # Just complicate way to assume unit 'ms' and convert to 's' ;)
        dwell_time = (float(header['jmrui']['SamplingInterval']) * u.millisecond).to_base_units()
        # Assume Hz in txt file
        spectral_frequency = (float(header['jmrui']['TransmitterFrequency']) * u.hertz)
        # jMRUI files do not define a reliable nucleus string
        nucleus = None

        # (!) Intermediate output: gets combined with other files in self._read_jmrui_file (if multiple files)
        return {
            'path': path,
            'signal': signal,
            'dwell_time': dwell_time,
            'spectral_frequency': spectral_frequency,
            'nucleus': nucleus,
        }

    def _load_jmrui_binary(self, path):
        """
        To load metabolite FID signal and parameters from ONE jMRUI .mrui-file.

        The multi-file handling (looping over self.path, checking common parameters,
        stacking signals, precision conversion) is done in self._read_jmrui_file.

        :param path: path to the single .mrui file to read
        :return: intermediate dict (path, signal, dwell_time, spectral_frequency, nucleus)
        """

        signal, header, _ = jmrui.read_mrui(str(path))
        signal = np.asarray(signal).squeeze() * u.dimensionless

        # Assuming jMRUI-convention: sampling_interval in ms, transmitter_frequency in Hz
        dwell_time = (float(header['sampling_interval']) * u.millisecond).to_base_units()
        # Assume Hz in binary (mrui) file
        spectral_frequency = (float(header['transmitter_frequency']) * u.hertz)
        # jMRUI files do not define a reliable nucleus string
        nucleus = None

        # (!) Intermediate output: gets combined with other files in self._read_jmrui_file (if multiple files)
        return {
            'path': path,
            'signal': signal,
            'dwell_time': dwell_time,
            'spectral_frequency': spectral_frequency,
            'nucleus': nucleus,
        }

    def _load_jmrui_m(self, verbose=False):
        """
        To load metabolite FID signal and parameters from a jMRUI .m-file
        (MATLAB-style text export).

        :param verbose: if it should be printed to the chat or not.
        :return: dict with parameters, signal of shape (metabolites, samples), time vector
        """
        # Read the content of the .m file
        parameters: dict = {}
        with open(self.path[0], 'r') as file: # need self.path[0], the [0], since list per default!
            file_content = file.read()
            file_content = file_content.replace('{', '[').replace('}', ']')

        # Create a dictionary to store variable names and their values
        data_found = False  # for entering the mode to append the FID data
        amplitude_str = "["
        for line in file_content.splitlines():
            parts = line.split('=')
            if len(parts) == 2:
                var_name = parts[0].strip()
                var_value = parts[1].strip().rstrip(';')  # Remove trailing ';' if present
                parameters[var_name] = var_value

            if data_found:
                amplitude_str += line + ", "
                pass
            elif parts[0] == "DATA":
                data_found = True

        # For creating time vector (numpy array) from the parameters given in the file
        # Also, adding the other content. This includes the name of the chemical compounds.
        dim_values = parameters["DIM_VALUES"].replace("[", "").replace("]", "").replace("'", "").split(",")
        parameters["DIM_VALUES"] = []
        parameters["DIM_VALUES"].append(dim_values[0])
        parameters["DIM_VALUES"].append(dim_values[1])
        parameters["DIM_VALUES"].append(dim_values[2:])
        time_vector_raw = parameters["DIM_VALUES"][0].split(":")
        t_start, t_step, t_end = float(time_vector_raw[0]), float(time_vector_raw[1]), float(time_vector_raw[2])
        time = np.arange(t_start, t_end, t_step)
        time = time[0:len(time) - 1]
        dwell_time_raw = t_step  # -> step size in DIM_VALUES is the dwell time (in sec, jMRUI .m convention)

        # For transforming the signal values given in the file to a numpy array
        parameters['SIZE'] = eval(parameters['SIZE'])  # str to list
        amplitude_str = amplitude_str.replace('\t', ' ').replace('[', '').replace(']', '').replace(';',
                                                                                                   '')  # .split(",")
        amplitude_list_strings = amplitude_str.split(",")
        amplitude_list_strings = [[float(num) for num in string_element.split()] for string_element in
                                  amplitude_list_strings]
        amplitude_raw = np.asarray(amplitude_list_strings[0:len(amplitude_list_strings) - 2])

        # Combine real and imaginary parts to one complex FID
        # -> jMRUI DATA columns are [re(FID), im(FID), re(spec), im(spec)]; only first two needed
        signal = (amplitude_raw[:, 0] + 1j * amplitude_raw[:, 1]).astype(complex)

        # Split the whole signal into per-compound rows
        # -> SIZE[2] = number of compounds; DIM_VALUES[2] = matching names (same order)
        # -> resulting shape is (metabolites, samples), all sharing the same time axis
        signal = signal.reshape(int(parameters['SIZE'][2]), -1)
        compound_names = parameters['DIM_VALUES'][2]

        # Attach units to signal, dwell time and time vector
        signal = signal * u.dimensionless                           # a.u., shape (metabolites, samples)
        dwell_time = (dwell_time_raw * u.second).to_base_units()    # in sec
        time = time * u.second                                      # in sec

        # Assume Hz for TransmitterFrequency in .m file (if present)
        raw_freq = parameters.get('TransmitterFrequency')
        if raw_freq:
            spectral_frequency = float(raw_freq) * u.hertz
        else:
            Console.printf("warning",
                           f"TransmitterFrequency not defined in the header of {self.path[0].name}! Set it to 'None'",
                           mute=not verbose)
            spectral_frequency = None

        # Check if nucleus is in header defined
        nucleus = parameters.get('Nucleus')
        if not nucleus:
            Console.printf("warning",
                           f"Nucleus not defined in the header of {self.path[0].name}! Set it to 'None'",
                           mute=not verbose)
            nucleus = None

        # Convert to desired data type
        signal = ArrayTools.to_precision(array=signal, precision=self.data_precision, verbose=verbose)
        time = ArrayTools.to_precision(array=time, precision=self.data_precision, verbose=verbose)

        # (!) Uniform formatted output (as other similar methods here)
        return {
            'parameters': {
                'dwell_time': dwell_time,                   # in sec
                'spectral_frequency': spectral_frequency,   # in Hz, possibly None
                'nucleus': nucleus,                         # possibly None
            },
            'signal': signal,                               # complex pint quantity, shape (metabolites, samples)
            'time': time,                                   # in sec, shared across all compound rows
            'name': compound_names                          # list of compound names from DIM_VALUES[2]
        }


    def _load_lcmodel_basis_set(self, verbose=False):
        """
        To load metabolite FID signals and parameters from a .basis file (jMRUI).

        :param verbose: if it should be printed to the chat or not.
        :return: dict with parameters, signal, time vector
        """

        basis_set = read_basis(str(self.path[0]))
        signals = np.asarray(basis_set.original_basis_array).T * u.dimensionless # -> (number metabolites, number points)

        # Assuming BADELT (dwell time) in sec and HZPPPM (Hz per ppm) in MHz
        dwell_time = (float(basis_set.original_dwell) * u.second).to_base_units()
        # Create the time vector
        # -> shape is (number metabolites, number points); time axis runs along axis=1
        time_vector = np.arange(signals.shape[1]) * dwell_time
        # Assume MHz in jMRUI .basis file
        spectral_frequency = (float(basis_set.cf) * u.megahertz).to_base_units()

        # Try to get nucleus from the FSL-MRS attributes
        nucleus = getattr(basis_set, 'nucleus', None)
        if not nucleus:
            Console.printf("warning", f"Nucleus not defined in the header of {self.path[0].name}! Set it to 'None'", mute=not verbose)

        # Convert to desired data type
        signal = ArrayTools.to_precision(array=signals, precision=self.data_precision, verbose=verbose)
        time_vector = ArrayTools.to_precision(array=time_vector, precision=self.data_precision, verbose=verbose)

        # (!) Uniform formatted output (as other similar methods here)
        return {
            'parameters': {
                'dwell_time': dwell_time,                   # in sec
                'spectral_frequency': spectral_frequency,   # in Hz
                'nucleus': nucleus,                         # possibly None
            },
            'signal': signal,                               # complex, dimensionless, shape (number metabolites, number points)
            'time': time_vector,                            # in sec
            'name': list(basis_set.names),                  # list of metabolite names in same order as signal array
        }


# TODO: DELETE
###class JMRUI:
###    """
###    For reading the data from an m.-File generated from an jMRUI. TODO: Implement for the .mat file. Further, also rename JMRUI instead of JMRUI!
###    """
###
###    def __init__(self, path: str, signal_data_type: np.dtype = np.float64, mute: bool = False):
###        self.path = path
###        self.signal_data_type = signal_data_type # TODO: Has no effect!
###        self.mute = mute
###
###    def load_m_file(self) -> dict:
###        parameters: dict = {}
###
###        # Read the content of the .m file
###        with open(self.path, 'r') as file:
###            file_content = file.read()
###            file_content = file_content.replace('{', '[').replace('}', ']')
###
###        # Create a dictionary to store variable names and their values
###        data_found = False  # for entering the mode to append the FID data
###        amplitude = "["
###        for line in file_content.splitlines():
###            parts = line.split('=')
###            if len(parts) == 2:
###                var_name = parts[0].strip()
###                var_value = parts[1].strip().rstrip(';')  # Remove trailing ';' if present
###                parameters[var_name] = var_value
###
###            if data_found:
###                amplitude += line + ", "
###                pass
###            elif parts[0] == "DATA":
###                data_found = True
###
###        # For creating time vector (numpy array) from the parameters given in the file
###        # Also, adding the other content. This includes the name of the chemical compounds.
###        dim_values = parameters["DIM_VALUES"].replace("[", "").replace("]", "").replace("'", "").split(",")
###        parameters["DIM_VALUES"] = []
###        parameters["DIM_VALUES"].append(dim_values[0])
###        parameters["DIM_VALUES"].append(dim_values[1])
###        parameters["DIM_VALUES"].append(dim_values[2:])
###        time_vector = parameters["DIM_VALUES"][0].split(":")
###        time_vector_start, time_vector_stepsize, time_vector_end = float(time_vector[0]), float(time_vector[1]), float(time_vector[2])
###        time = np.arange(time_vector_start, time_vector_end, time_vector_stepsize)
###        time = time[0:len(time) - 1]
###
###        # For transforming the signal values given in the file to a numpy array
###        parameters['SIZE'] = eval(parameters['SIZE'])  # str to list
###        amplitude = amplitude.replace('\t', ' ').replace('[', '').replace(']', '').replace(';', '')  # .split(",")
###        amplitude_list_strings = amplitude.split(",")
###        amplitude_list_strings = [[float(num) for num in string_element.split()] for string_element in amplitude_list_strings]
###        amplitude = np.asarray(amplitude_list_strings[0:len(amplitude_list_strings) - 2])
###
###        # Short overview of a chosen data type for the FID signal, used space and precision (in digits)
###        Console.printf("info", f"Loaded FID signal as {self.signal_data_type} \n" +
###                       f" -> thus using space: {amplitude.nbytes / 1024} KB \n" +
###                       f" -> thus using digits: {np.finfo(amplitude.dtype).precision}",
###                       mute=self.mute)
###
###        return {"parameters": parameters,
###                "signal": amplitude,
###                "time": time}
###
###    def show_parameters(self) -> None:
###        """
###        Printing the successfully read parameters formatted to the console.
###
###        :return: None
###        """
###        # np.set_printoptions(20)  # for printing numpy numbers with 20 digits to the console
###        for key, value in self.parameters.items():
###            Console.add_lines(f"Key: {key}, Value: {value}")
###
###        Console.printf_collected_lines("success")
###        # np.set_printoptions() resetting numpy printing options


class NeuroImage:
    """
    For loading neuro images of the format "Neuroimaging Informatics Technology Initiative" (NIfTI).
    """

    def __init__(self, path: str):
        self.path: str = path  # Path to the file
        self.name: str = Path(self.path).name  # the name of the file
        self.nifti_object: nib.nifti1.Nifti1Image = None  # Just the raw loaded data with header

        self.header: nib.nifti1.Nifti1Header = None  # Header of the nifti image
        self.data: np.memmap = None  # Image data of the nifti image
        self.shape: tuple = ()  # Just the shape of the nifti image data

    def load_nii(self, data_type: np.dtype = np.float64, mute: bool = False, report_nan: bool =True):
        """
        The data will be loaded as "np.memmap", thus usage of a memory map. Since the data is just mapped, no
        full workload on the RAM because data is only loaded into memory on demand.
        For a more efficient way see: https://nipy.org/nibabel/images_and_memory.html

        :return: None
        """

        try:
            self.nifti_object = nib.load(self.path)
        except:
            Console.printf("error", f"Cannot load file: '{self.name}'")
            return

        self.header = self.nifti_object.header
        self.data = self.nifti_object.get_fdata(dtype=data_type)
        self.shape = self.nifti_object.shape

        Console.printf("success", f"Loaded file '{self.name}':"
                                  f"\n    Shape             -> {self.shape}"
                                  f"\n    Pixel dimensions: -> {self.header.get_zooms()}"
                                  f"\n    Values range:     -> [{round(np.min(self.data), 3)}, {round(np.max(self.data), 3)}]"
                                  f"\n    Data type:        -> {data_type}"
                                  f"\n    In memory cache?  -> {self.nifti_object.in_memory}",  # TODO what was exactly the purpose?
                       mute=mute)

        ArrayTools.check_nan(self.data, verbose=report_nan)

        return self




class Mask(Plot):
    """
    For loading a mask from a file (e.g., metabolic mask, lipid mask, B0 inhomogeneities, ...). It requires a
    available JSON configuration file. See :class: `Configurator`.
    """
    def __init__(self, configurator: Configurator):
        self.configurator = configurator
        self.mask = None
        self.mask_name = None


    def load(self, mask_name: str, data_type: np.dtype = np.float64) -> NeuroImage:
        """
        For loading a mask from a path. The necessary path is available in a JSON config file (path_file),
        the :class: `Configurator does handle it.

        :param configurator: Handles the paths, thus also for the masks.
        :param mask_name: The name of the mask. Needed to be available in the JSON file.
        :param data_type: The data type of the mask
        :return: An object of the class :class: `Neuroimage`. The variable 'data' of the object returns the mask data itself (numpy memmap).
        """

        # Load the paths from JSON file in the configurator object
        self.configurator.load()

        # Assign the current mask name to instance variable
        self.mask_name = mask_name

        # Then, check if Mask exists and load the path. Otherwise, warn the user and exit program.
        available_masks = list(self.configurator.data["mask"].keys())
        if mask_name not in available_masks:
            Console.printf("error", f"Mask '{mask_name}' not listed in {configurator.file_name}. Only listed: {available_masks}. Terminating the program!")
            sys.exit()

        # Finally, load the mask according to the path
        self.mask = NeuroImage(path=self.configurator.data["mask"][mask_name]["path"]).load_nii(data_type=data_type)

        Console.printf("success", f"Thus, loaded the '{mask_name}' mask.")

        return self.mask

    def plot_jupyter(self, cmap="gray"):
        """
        To plot the loaded volume in an interactive form for the jupyter notebook/lab.
        (!) Note: %matplotlib ipympl need be called once in the respective jupyter notebook.

        :return: Nothing
        """
        if self.mask is not None:
            JupyterPlotManager.volume_grid_viewer(vols=[self.mask.data], rows=1, cols=1, titles=self.mask_name, cmap=cmap)
        else:
            Console.printf("error", "No mask loaded yet, thus no plotting possible.")

    def plot(self, figsize=(10,5), cmap="gray", slices:str | tuple[int, int, int] ="central"):
        """
        To plot the loaded volume as cross-sectional view. None interactive. Works also with command line.

        :return: Nothing
        """
        zz, yy, xx = None, None, None

        if slices == "central":
            z, y, x = self.mask.data.shape
            zz, yy, xx = z//2, y//2, x//2
            Console.printf("info", f"Plotting central slices by default: ({zz},{yy},{xx})")
        else:
            if isinstance(slices, tuple):
                zz, yy, xx = slices[0] // 2, slices[1] // 2, slices[2] // 2
            else:
                Console.printf("error", f"Format required for plotting custom slices: tuple(int, int, int). But provided: {type(slices)}")


        fig, axs = plt.subplots(figsize=figsize, nrows=1, ncols=3)
        slice_1 = axs[0].imshow(self.mask.data[zz,:,:], cmap=cmap)
        axs[0].set_title("Z")
        slice_2 = axs[1].imshow(self.mask.data[:,yy,:], cmap=cmap)
        axs[1].set_title("Y")
        slice_3 = axs[2].imshow(self.mask.data[:,:,xx], cmap=cmap)
        axs[2].set_title("X")
        fig.suptitle(self.mask_name, fontsize=16)

        cbar = fig.colorbar(slice_1, ax=axs, location="right", fraction=0.03, pad=0.02)
        cbar.set_label("Value")

        plt.show()


class MetabolicAtlas:
    # TODO

    def __init__(self):
        # TODO
        pass

    def load(self):
        # TODO
        pass


class ParameterMaps(WorkingSource[ParameterVolume], Plot):
    """
    This class can be used at the moment for various purposes:

    1) To load data:
        -> from one nii file, yields dictionary of just values
        -> from one h5 file, yields dictionary (e.g., with keys 'imag', 'real')
        -> from multiple nii files in a respective dictionary (for metabolites)

    2) Transform to 'working map' from spatial metabolic distribution
        Therefore, create working map object with data from this class:
            -> map_type_name
            -> loaded maps
            -> loaded maps unit

    The class automatically figures out the filetype, at the moment based on '.nii', '.nii.gz', '.h5', '.hdf5'.
    """
    # TODO: Maybe program more flexible

    u = pint.UnitRegistry() # class variable, same for all instances

    def __init__(self, configurator: Configurator, map_type_name: str):
        self.configurator = configurator
        self.map_type_name = map_type_name  # e.g. B0, B1, metabolites
        self.loaded_maps: dict[str, np.memmap | h5py._hl.dataset.Dataset] | np.memmap | np.ndarray = None
        self.loaded_maps_unit = None
        self.file_type_allowed = ['.nii', '.nii.gz', '.h5', '.hdf5']


        self.main_path = Path(self.configurator.data["maps"][self.map_type_name]["path"])
        if self.main_path.is_file():
            # Case 1: the main_path points to just one file
            Console.printf("info", "Maps object: The provided path points to a file")
            name = self.main_path.name.lower()
            self.file_type = ".nii.gz" if name.endswith(".nii.gz") else self.main_path.suffix.lower()

        elif self.main_path.is_dir():
            # Case 2: the main_path points to a folder containing multiple files
            Console.printf("info", "Maps object: The provided path points to a folder")
            #  Check if multiple extensions available. If just same, and since "set" used, entries cannot be twice.
            files_extensions = {file.suffix.lower() for file in self.main_path.iterdir() if file.is_file()}
            if len(files_extensions) == 1:
                self.file_type = files_extensions.pop() # get from set the extension (only one is available here (e.g., {".nii})
            else:
                raise ValueError(f"Multiple file types found: {files_extensions}, folder operation not possible!")

        else:
            Console.printf("error", f"Maps object: The provided path does not exist or is neither a file nor a folder: {self.main_path} Exiting system.")
            sys.exit()

        if self.file_type not in self.file_type_allowed:
            Console.printf("error", f"Only possible to load formats: {self.file_type_allowed}. But it was given: {self.file_type}")
            sys.exit()

        self.configurator.load()


    def load_file(self) -> Self:
        """
        Load a single map from file.
        Supports both H5 and NIfTI (nii) formats.
        """
        self.loaded_maps = {}             # key is abbreviation like "Glu" for better matching
        self._load_and_assign_pint_unit() # load the associated unit defined in the json file

        if self.file_type == '.h5' or self.file_type == '.hdf5':
            """
            Case 1: To handle h5 files: Results in dictionary ->  self.loaded_maps[key] = values
            """
            Console.printf("info", f"Loading h5 file for map type {self.map_type_name}")
            with h5py.File(self.main_path, "r") as file:
                for key in file.keys():
                    item = file[key]

                    if isinstance(item, h5py.Dataset):
                        Console.add_lines(f" => Key: {key:<5} | Type: {'Dataset':<10} | Shape: {item.shape} | Data Type: {item.dtype}")
                    elif isinstance(item, h5py.Group):
                        Console.add_lines(f" => Key: {key:<5} | Type: {'Group':<10}")

                    Console.add_lines("     Attributes: ")
                    if len(item.attrs.items()) == 0:
                        Console.add_lines("      Not available")
                    else:
                        for attr_key, attr_val in item.attrs.items():
                            Console.add_lines(f"        Attribute key: {attr_key:<10} | value: {attr_val:<20}")

                    self.loaded_maps[key] = item

            Console.printf_collected_lines("success")

        elif self.file_type == '.nii' or self.file_type == '.nii.gz':
            """
            Case 2: To handle nii files: Results in dictionary -> self.loaded_maps[map_type_name] = values
            """
            Console.printf("info", f"Loading nii file for map type {self.map_type_name}")
            loaded_map = NeuroImage(path=self.main_path).load_nii(mute=True, report_nan=True).data

            # store as dict for consistent downstream API
            self.loaded_maps[self.map_type_name] = loaded_map
            Console.add_lines(
                f"Loaded nii map: {os.path.basename(self.main_path)} | Shape: {loaded_map.shape} | "
                f"Values range: [{round(np.min(loaded_map), 3)}, {round(np.max(loaded_map), 3)}] | "
                f"Unit: {self.loaded_maps_unit} | "
                f"Unique values: {len(np.unique(loaded_map))}"
            )
            Console.printf_collected_lines("success")
        else:
            Console.printf("error", f"Unsupported file type: {self.file_type}")
            sys.exit()

        return self

    def load_files_from_folder(self, working_name_and_file_name: dict[str, str]) -> Self:
        """
        (!) At the moment mainly for loading metabolites. As example use the dictionary to load:

        working_name_and_file_name = {  "Glu": "MetMap_Glu_con_map_TargetRes_HiRes.nii",
                                        "Gln": "MetMap_Gln_con_map_TargetRes_HiRes.nii",
                                        "m-Ins": "MetMap_Ins_con_map_TargetRes_HiRes.nii",
                                        "NAA": "MetMap_NAA_con_map_TargetRes_HiRes.nii",
                                        "Cr+PCr": "MetMap_Cr+PCr_con_map_TargetRes_HiRes.nii",
                                        "GPC+PCh": "MetMap_GPC+PCh_con_map_TargetRes_HiRes.nii" }

        :param working_name_and_file_name: a dictionary containing as key the desired 'working name, like Glu' and the corresponding real filename as value
        :return: the object of the whole class
        """
        Console.printf("warning", "Maps.load_files_from_folder ==> by standard nii is loaded. No h5 support yet!")
        Console.printf("warning", "The same unit is assumed for all the files in the given folder!")
        self.configurator.load()
        main_path = self.configurator.data["maps"][self.map_type_name]["path"]

        self._load_and_assign_pint_unit()
        self.loaded_maps: dict = {} # dict required for the operations afterward

        Console.add_lines("Loaded maps: ")
        for i, (working_name, file_name) in enumerate(working_name_and_file_name.items()):
            path_to_map = os.path.join(main_path, file_name)
            loaded_map = NeuroImage(path=path_to_map).load_nii(mute=True, report_nan=True).data
            self.loaded_maps[working_name] = loaded_map

            Console.add_lines(
                f"  {i}: working name: {working_name:.>10} | {'Shape:':<8} {loaded_map.shape} | "
                f"Values range: [{round(np.min(loaded_map), 3)}, {round(np.max(loaded_map), 3)}] "
                f"| Unit: {self.loaded_maps_unit} |"
            )
        Console.printf_collected_lines("success")


        return self

    def _load_and_assign_pint_unit(self):
        """
        For loading the unit string, which should be compliant with the library. If it fails, fallback to "dimensionless" via pint.

        :return:
        """
        unit_string = self.configurator.data["maps"][self.map_type_name]["unit"]

        # Try to convert provided string in the config file to pint unit, and assign "dimensionless" if it fails
        try:
            self.loaded_maps_unit = ParameterMaps.u.Unit(unit_string)
        except:
            Console.printf("error", f"Could not convert loaded unit '{unit_string}' to pint unit. Therefore, assigned 'dimensionless'!")


    def to_base_units(self, verbose=False):
        """
        To convert all loaded arrays to the base units.

        :param verbose: it true then it prints conversation results to the console.
        :return: Nothing
        """

        loaded_maps_unit_before = self.loaded_maps_unit
        loaded_maps_unit_after = None
        try:
            # CASE A: multiple arrays
            if isinstance(self.loaded_maps, dict):
                for i, (name, array) in enumerate(self.loaded_maps.items()):
                    self.loaded_maps[name], loaded_maps_unit_after = UnitTools.to_base(
                        values=array,
                        units=self.loaded_maps_unit,
                        return_separate=True,
                        verbose=verbose)

                self.loaded_maps_unit = loaded_maps_unit_after

            # CASE B: assuming single array (e.g. numpy array, numpy memmap, and so on)
            else:
                self.loaded_maps, self.loaded_maps_unit = UnitTools.to_base(
                    values=self.loaded_maps,
                    units=self.loaded_maps_unit,
                    return_separate=True,
                    verbose=verbose)

            Console.printf("success", f"Converted to base units: {loaded_maps_unit_before} -> {self.loaded_maps_unit}")
        except:
            Console.printf("error", "Could not convert to base units.")

    def plot(self, cmap="viridis"):
        fig, axs = plt.subplots(figsize=(15, 2), ncols=len(self.loaded_maps.items()), nrows=1, cmap=cmap)

        for i, (key, value) in enumerate(self.loaded_maps.items()):
            image = axs[i].imshow(value[:, :, 40])
            axs[i].set_title(key)
            axs[i].set_axis_off()

        fig.colorbar(image, ax=axs, location="right")

    def plot_jupyter(self, cmap="gray"):
        """
        To plot the loaded volume in an interactive form for the jupyter notebook/lab.
        (!) Note: %matplotlib ipympl need be called once in the respective jupyter notebook.

        :return: Nothing
        """
        if self.loaded_maps is not None:
            JupyterPlotManager.volume_grid_viewer(vols=self.loaded_maps.values(), rows=1, cols=len(self.loaded_maps.keys()), titles=self.loaded_maps.keys(), cmap=cmap)
        else:
            Console.printf("error", "No maps loaded yet, thus no plotting possible.")


    def to_working(self, data_type: str = None, verbose: bool = True) -> ParameterVolume:
        """
        This transforms the loaded maps directly to a 4D array of (metabolites, X, Y, Z)

        :return: a ParameterVolume (see module spatial_simulation)
        """

        # Create the Parameter Volume
        volume = ParameterVolume(maps_type=self.map_type_name)

        # Create for each metabolite a Parameter Map and add it to the Parameter Volume
        for metabolite, data in self.loaded_maps.items():
            parameter_map = ParameterMap(map_type=self.map_type_name, metabolite_name=metabolite, values=data, unit=self.loaded_maps_unit, affine=None)
            volume.add_map(map=parameter_map, verbose=False)

        # To create from all the 3D volumes inside the ParameterVolume actually one 4D volume
        volume.to_volume(verbose=verbose)

        # Change data type if given by the user
        if data_type is not None:
            volume.to_data_type(data_type)

        return volume



class FID(WorkingSource[SpectralSpatialFID]):
    """
    This class is for creating an FID containing several attributes. The FID signal and parameters can be
    either added from e.g., a simulation or loaded from a MATLAB file (.m). Moreover, it is able to convert
    the Real and Imaginary part in the file given as (Re, Im) -> Re + j*Im.
    """

    def __init__(self, configurator: Configurator):
        self.configurator = configurator
        self.parameters: dict = {}
        self.signal: np.ndarray = None
        self.time: np.ndarray = None
        self.names: list = []
        self.fid: spectral_spatial_simulation.FID = None

    def load(self, fid_key: str, compounds: list | str = None, data_precision: int = 32, verbose: bool = True):
        """
        For loading and splitting the FID according to the respective chemical compound (metabolites, lipids).
        Then, create on :class: `spectral_spatial_simulation.FID` for each chemical compound and store it into a list.
        Additional: since the complex signal is represented in two columns (one for real and one for imaginary),
        it has to be transformed to a complex signal.

        :param compounds: if desired enter a list of the names (or one string name) of the chemical compounds that should be loaded. Otherwise, all will be loaded!
        :param data_precision: the precision of the FID signal and time. If e.g., 32 then float32 for time and complex64 for signal
        :param verbose: if True then prints to the console
        :param fid_key: Name of the FID (e.g., 'metabolites', 'lipids'). For nested: 'abcd.txt.metabolites'
        :return: Nothing. Access the list loaded_fid of the object for the signals!
        """

        # (1) Load the data from the config file and get the required path
        self.configurator.load()
        path = self.configurator.get_data(key=["fid", fid_key, "path"])

        # (2) Load the basis set data
        basis_set = BasisSet(path=path, data_precision=data_precision, verbose=verbose)
        data = basis_set.load(subset_names=compounds)

        # (3) Extract the relevant data from the basis set data
        self.parameters, self.signal, self.time = data["parameters"], data["signal"], data["time"]

        # (4) Create for each signal an FID object and then merge it to one FID object
        Console.add_lines("Assigned FID parts:")
        self.names = data["name"]

        # (5) Check for loaded signal and time if NaNs are present!a
        if ArrayTools.check_nan(self.signal, verbose=False):
            Console.printf("warning", "Loaded signal of FID contains NaNs.")
        if ArrayTools.check_nan(self.time, verbose=False):
            Console.printf("warning", "loaded time vector of FID contains NaNs.")

        return self


    def to_working(self, verbose: bool = True) -> SpectralSpatialFID:
        """
        To transform the loaded data to an FID object in another module, since this class
        here is just for handling loading FID data from the files.
        The other FID object has various methods to manipulate the FID signal, e.g.,
        including interpolation, merging, renaming and so on.

        :param verbose: if True then print to the console.
        :return: The spectral spatial FID object.
        """

        # (1) To create an empty FID object
        fid = spectral_spatial_simulation.FID()

        # (2) Create and FID object for each signal and then add it together.
        #     The new FID class also checks for having the same time vector
        #     and unit.
        for column_number, name in enumerate(self.names):
            fid += spectral_spatial_simulation.FID(signal=self.signal[column_number], time=self.time, name=[name])
            if verbose:
                Console.add_lines(f"{column_number}. {name + ' ':-<30}-> shape: {self.signal[column_number].shape}")

        if verbose:
            Console.printf_collected_lines("success")

        return fid

    def show_parameters(self) -> None:
        """
        Printing the successfully read parameters formatted to the console.

        :return: None
        """
        # np.set_printoptions(20)  # for printing numpy numbers with 20 digits to the console
        for key, value in self.parameters.items():
            Console.add_lines(f"Key: {key}, Value: {value}")

        Console.printf_collected_lines("success")


class CoilSensitivityMaps(WorkingSource["SamplingCoilSensitivityVolume"], Plot):
    """
    For loading and interpolating the coil sensitivity maps. At the moment, only HDF5 files are supported.
    The maps can be interpolated on 'cpu' and desired 'gpu'. It is strongly recommended to use the 'gpu'
    regarding computational performance.
    An object of the configurator is necessary to get the path to the coil sensitivity maps.

    Note: Although at the moment only able to load a volume, it could be in future that it will all load
    some coil sensitivity maps that are separate and not in one 4D array, therefore still the naming
    CoilSensitivityMaps and not CoilSensitivityVolume!
    """

    def __init__(self, configurator: Configurator):
        self.configurator = configurator
        self.maps: np.ndarray | np.memmap = None
        self.coils: list[str] = None
        self.shape = None

    def load_h5py(self, keys: list = ["imag", "real"], dtype=np.complex64) -> None:
        """
        For loading coil sensitivity maps of the type HDF5. The default keys are 'real' and 'imag' for loading the complex values.
        If different keys are defined in the HDF5 it is necessary to explicitely define them with the argument 'keys'.

        :param dtype: desired data type. Since the data is complex ude numpy complex data types.
        :return: None.
        """

        # Load the HDF5 file via h5py
        path_maps = self.configurator.data["maps"]["coil_sensitivity"]["path"]
        h5py_file = h5py.File(path_maps, "r")
        h5py_file_keys_required = keys

        # If the default keys or by the user defined keys are not available in the HDF5 file:
        if not all(key in h5py_file.keys() for key in h5py_file_keys_required):
            Console.add_lines(f"Could not find keys {h5py_file_keys_required} in the HDF5 file")
            Console.add_lines(f" => Instead the following keys are available: {h5py_file.keys()}.")
            Console.add_lines(f" => Aborting the program.")
            Console.printf_collected_lines("error")
            sys.exit()

        # If the keys are existing in the HDF5 file, load them and convert to complex values:
        maps_real = h5py_file["real"][:]
        maps_imag = h5py_file["imag"][:]
        maps_complex = maps_real + 1j * maps_imag
        maps_complex = maps_complex.astype(dtype, copy=False)

        self.coils = [f"Coil {i+1}" for i in range(maps_complex.shape[0])]

        Console.add_lines("Coil Sensitivity Maps:")
        Console.add_lines(f" => Could find keys {h5py_file_keys_required} in the HDF5 file")
        Console.add_lines(f" => Loaded and converted maps to {maps_complex.dtype}")
        Console.add_lines(f" => Data shape: {maps_complex.shape}")
        Console.add_lines(f" => Space required: {SpaceEstimator.for_array(maps_complex, 'MiB'):.5g}")
        Console.printf_collected_lines("success")

        self.maps = maps_complex

    def to_working(self) -> SamplingCoilSensitivityVolume| None:
        """
        Converts the from file.CoilSensitivityMaps to sampling.CoilSensitivityVolume.

        :return: SamplingCoilSensitivityVolumein sampling
        """
        from sampling_simulation import CoilSensitivityVolume

        if self.maps is None:
            Console.printf("error", "Need to load coil sensitivity maps before converting to working volume!")
            return None
        else:
            coil_sensitivity_volume = CoilSensitivityVolume()
            coil_sensitivity_volume.volume: list[str] = self.maps
            coil_sensitivity_volume.coils: np.ndarray | np.memmap = [f"Coil {i+1}" for i in range(self.maps.shape[0])]
            Console.printf("success", "Transformed file.CoilSensitivityMaps => sampling.CoilSensitivityVolume")

            return coil_sensitivity_volume


    def plot_jupyter(self, cmap: str = "viridis"):
        """
        To plot the coild sensitivity maps to the notebooker lab. It should be interactive,

        :param cmap: color map. See matplotlib.
        :return: Nothing
        """

        cols = int(np.ceil(len(self.coils)/2))
        rows = 2

        if self.maps is not None:
            # self.maps has shape (coil, X,Y,Z) and list(self.maps) cerates list((X,Y,Z), (X,Y,Z), ...)
            JupyterPlotManager.volume_grid_viewer(vols=list(np.abs(self.maps)), rows=rows, cols=cols, titles=self.coils, cmap=cmap)
        else:
            Console.printf("error", "No coil sensitivity maps loaded yet, thus no plotting possible.")
        pass

    def plot(self, cmap: str = "gray", **kwargs):
        """
        Plot one slice per coil.
        """

        slice_position = kwargs.pop("slice_position", None)
        axis = kwargs.pop("axis", 3)  # choose a sensible default for your data
        ax_imshow = kwargs  # remaining kwargs forwarded to imshow

        # Validate axis
        if axis is None:
            axis = 3
            Console.add_lines(f"No axis to plot is specified. Choosing: {axis}")

        # Default slice position to middle
        if slice_position is None:
            slice_position = self.maps.shape[axis] // 2
            Console.add_lines(f"No slice to plot is specified. Choosing: {slice_position}")

        if axis is None or slice_position is None:
            Console.printf_collected_lines("warning")

        n_coils = self.maps.shape[0]
        ncols = max(1, n_coils // 2)
        nrows = 2 if n_coils > 1 else 1

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 2 * nrows))
        axs = np.atleast_1d(axs).ravel()

        for i, ax in enumerate(axs[:n_coils]):
            # Slice per coil
            coil_map = self.maps[i]
            chosen_slice = np.abs(coil_map.take(slice_position, axis=axis - 1))  # axis shifts because coil dim removed

            ax.imshow(chosen_slice, cmap=cmap, **ax_imshow)
            ax.axis("off")
            ax.set_title(f"coil {i}")

        # Hide unused axes if any
        for ax in axs[n_coils:]:
            ax.axis("off")

        plt.show()


class Trajectory:
    """
    To read the parameters and gradient values from JSON trajectories files.

    This requires files:
        * JSON Structure file
        * JSON Gradient file
    """

    # Class variable (thus not bound to an object)
    u = pint.UnitRegistry()  # for using units with the same Registry

    def __init__(self, configurator: Configurator):
        """
        Initialise the object with the configurator which manages the paths, thus also contains the
        paths to the respective trajectory file.

        :param configurator: Configurator object that manages the paths.
        """
        self.configurator = configurator.load()
        self.path_trajectories = configurator.data["path"]["trajectories"]
        self.path_simulation_parameters = configurator.data["path"]["simulation_parameters"]

    @staticmethod
    def get_pint_unit_registry() -> pint.UnitRegistry:
        """
        For using the same registry as used to load the trajectory parameters and data.
        Usefully for methods and functions that implement the file.Trajectory class.

        :return: pint.UnitRegistry(), often used just as "u." (e.g., u.mm for millimeter).
        """

        return Trajectory.u

    def load_cartesian(self, console_output : bool = False) -> dict:
        """
        To load the simulation parameters of the respective JSON file for being able to construct the cartesian trajectory.
        Also, renaming of the keys - if desired - can be done here.

        :return: Dictionary with simulation parameters to construct the cartesian trajectory
        """

        # Need specific parameters from the simulation parameters file
        with open(self.path_simulation_parameters, "r") as file_simulation_parameters:
            json_parameters_content = json.load(file_simulation_parameters)

        # For extracting only the desired parameters via keys from the loaded dict
        desired_parameters = [
            "MatrixSizeImageSpace",
            "AcquisitionVoxelSize",
            "MagneticFieldStrength",
            "SpectrometerFrequency"
        ]

        # Selecting only the desired parameters based on the defined keys before
        selected_parameters = {key: json_parameters_content[key] for key in desired_parameters if key in json_parameters_content}

        # Rename key that does not match BIDS (MatrixSizeImageSpace -> MatrixSize)
        selected_parameters["MatrixSize"] = selected_parameters.pop("MatrixSizeImageSpace")

        # To add units to the parameters
        u = self.get_pint_unit_registry() # get class variable u
        selected_parameters["AcquisitionVoxelSize"] = selected_parameters["AcquisitionVoxelSize"] * u.mm
        selected_parameters["MagneticFieldStrength"] = selected_parameters["MagneticFieldStrength"] * u.T
        selected_parameters["SpectrometerFrequency"] = selected_parameters["SpectrometerFrequency"] * u.MHz

        # Output to the console if True
        if console_output:
            Console.add_lines(f"Loaded cartesian parameters from file '{Path(self.path_simulation_parameters).name}':")
            for key, value in selected_parameters.items():
                Console.add_lines(f" -> {key:.<25}: {value}")

            Console.printf_collected_lines("success")

        # -> TODO: partition encoding * nuber slices (where to find?)
        # -> TODO: vector size? Where do I get from?
        # -> NumberReceiveCoilActiveElements (total channels measured previously)

        return selected_parameters

    def load_concentric_rings(self) -> dict:
        """
        To load the content of the respective JSON files to get the trajectory data itself and the according information.

        :return: Dictionary with the keys of the json files with the respective data. The file information is removed.
        """

        # Load all data from structure and gradient file
        with open(self.path_simulation_parameters, "r") as file_simulation_parameters, open(self.path_trajectories["crt"], "r") as file_gradients:
            json_parameters_content = json.load(file_simulation_parameters)
            json_gradients_content = json.load(file_gradients)

        # Merge whole content of both files
        json_merged_content = json_parameters_content | json_gradients_content
        json_merged_content.pop("FileInfo")

        # Transform the gradient values, represented as string list for each trajectory, to a list
        # of complex numpy arrays.
        trajectories_gradients_values: list = []
        for trajectory_gradient_values in json_merged_content["GradientValues"]:
            trajectories_gradients_values.append(JsonConverter.complex_string_list_to_numpy(trajectory_gradient_values))

        json_merged_content["GradientValues"] = trajectories_gradients_values

        # Just for printing loaded data (which is not too long) to the console.
        Console.add_lines(f"\nUsed files for concentric rings trajectory:")
        Console.add_lines(f" a) {Path(self.path_simulation_parameters)}")
        Console.add_lines(f" b) {self.path_trajectories['crt']}")
        Console.add_lines(f"\nLoaded parameters and data:")
        for key, value in json_merged_content.items():
            if len(str(value)) < 50: # convert each value to string and count length to omit too long data
                Console.add_lines(f" -> {key:.<40}: {value}")
            else:
                Console.add_lines(f" -> {key:.<40}: (!) Too long. Not displayed.")

        Console.printf_collected_lines("success")

        return json_merged_content




@deprecated(reason="This class only supports .mat files. The simulation uses .json files, thus use the class Trajectory.")
class CRTTrajectoriesMAT:
    """
    TODO: Only for CRT now, thus name it or mark it somehow!
    To read trajectory data from .mat files and prepare it by organising all metadata and trajectory data into a dictionary with lists. This process structures
    the data from the .mat file in a specific format suitable for further use.
    """

    def __init__(self, configurator: Configurator, trajectory_type: str):
        """
        Initialise the object with the configurator which manages the paths, thus also contains the
        paths to the respective trajectory file. The trajectory type is defined in the json file which
        the configurator handles.

        :param configurator: Instantiated configurator object that manages the paths.
        :param trajectory_type: E.g., 'crt'. However, this example might differ. Check the respective json file which the configurator uses.
        """
        self.configurator = configurator.load()
        self.trajectory_type = trajectory_type
        self.path = configurator.data["path"]["trajectories"][trajectory_type]
        self.data = None
        self.number_of_trajectories = None

        # Only important if def 'load_from_mat_file' is called.
        self.important_keys_mat_file: list = ["dMaxGradAmpl",
                                              "NumberOfBrakeRunPoints",
                                              "NumberOfLaunTrackPoints",
                                              "NumberOfLoopPoints",
                                              "NumberOfRewinderPoints",
                                              "NumberOfAngularInterleaves",
                                              "dGradValues"]

        ###self.important_keys_json_file: list = ["MaximumGradientAmplitude",  # TODO: in gradients file!
        ###                                       "GradientValues",            # TODO: in gradients file!
        ###                                       "NumberAngularInterleaves",  # TODO: in structure file!
        ###                                       "NumberRewinderAfterEachTrajectory",   # TODO: in structure file!
        ###                                       "NumberRewinderAfterAllTrajectories",  # TODO: in structure file! # (=break run points) TODO: Use already?
        ###                                       ]

        # NumberOfLoopPoints can be calculated -> + rename "number_measured_points"

        # "NumberOfBrakeRunPoints",    # TODO ?????????????????
        # "NumberOfLaunTrackPoints",   # TODO OK! -> measured points[0][0]-1
        # "NumberOfLoopPoints",        # TODO DELETE AND USE MeasuredPoints, thus maybe naming_ "number_measured_points": -> measured points[0][1] - measured points[0][0]+1
        # "NumberOfRewinderPoints",    # TODO -> Maybe not needed? (allways 0)
        # "NumberOfAngularInterleaves" # TODO -> Maybe not needed? (allways 0, but wrong in intermediate file!?)

        self.combined_trajectories: dict = None

    ###def _combine_trajectories_json_file(self) -> None:
    ###    # TODO: Under construction
    ###    pass
    ###
    ###def load_from_json_file(self):
    ###    # TODO: Under construction
    ###    # TODO: Docstring
    ###
    ###    from pprint import pprint
    ###    # open json file
    ###    with open(self.path, 'r') as openfile:
    ###        # Reading from json file
    ###        json_object = json.load(openfile)
    ###
    ###        for i in range(len(json_object)):
    ###            try:
    ###                json_object[f"trajectory {i + 1}"]["gradient values"] = JsonConverter.complex_string_list_to_numpy(json_object[f"trajectory {i + 1}"]["gradient values"])
    ###            except:
    ###                pass
    ###            finally:
    ###                trajectories_data = json_object
    ###
    ###    # get number of trajectories
    ###    # -> combine trajectories
    ###    # print loaded
    ###
    ###    pass

    def _combine_trajectories_mat_file(self) -> None:
        """
        Reformat the trajectory data that it result in one dictionary with the respective (defined) keys holding the
        related values. The dictionary values are stored in lists. Don't call this private method directly from an outside.

        :return: Nothing
        """

        # create a empty list
        self.combined_trajectories = {key: [] for key in self.important_keys_mat_file}  # (1) create pre-defined empty directory

        # use list comprehensions instead of nested for loops:
        [self.combined_trajectories[key].append(one_trajectory_data[key])  # (4) append to pre-defined empty dictionary           ^
         for one_trajectory_data in self.data["trajectory_pack"]  # (2) get one after another trajectory data            |
         for key in self.important_keys_mat_file]  # (3) get only items from important keys in dictionary |

    def load_from_mat_file(self, squeeze_trajectory_data=True) -> dict[list]:
        """
        To load the trajectory data from a dict and then combine the data (restructure data) that a dictionary holding only the necessary keys
        with the values in a respective list.
        For each trajectory, one list entry of the respective key is represented. E.g., for 32 trajectories each of the 32 list entries storing
        one numpy array with the trajectory data.

        :param squeeze_trajectory_data: bool if extra dimensions of size 1 should be removed. E.g., if true: shape 1,22,5,1 --> shape 22,5
        :return: dictionary with the trajectory parameters and values
        """
        # (1a) load the data from the .mat file
        #     Note: load trajectory data and squeeze by default. E.g. shape 1,10,5,1 --> 10,5
        #     Note: simplify_cells enables to get a dictionary with the right keys to the according values, not a tuple
        self.data = loadmat(self.path, squeeze_me=squeeze_trajectory_data, simplify_cells=True)
        # (1b) Also check if the right keys are provided in the matlab file. Missing values are listed in the error message.
        subset_keys = set(self.important_keys_mat_file)
        superset_keys = set(list(self.data["trajectory_pack"][0].keys()))
        if subset_keys.issubset(superset_keys):
            Console.printf("success", f"Found necessary keys in {Path(self.path).name} file")
        else:
            missing_keys = subset_keys - superset_keys
            Console.printf("error", f"Missing values in file '{Path(self.path).name}': {missing_keys} Aborting program!")
            sys.exit()
        # (2) get number of trajectories
        self.number_of_trajectories = len(self.data["trajectory_pack"])
        # (3) combine the data fom all trajectories
        self._combine_trajectories_mat_file()

        # (4) print info what has been loaded
        Console.add_lines(f"Loaded trajectory type: {self.trajectory_type}")
        Console.add_lines(f"Number of trajectories: {self.number_of_trajectories}")
        data_keys_formatted = '\n  => '.join(self.important_keys_mat_file)
        Console.add_lines(f"Used values from keys: \n  => {data_keys_formatted}")
        Console.add_lines(f"Loaded from: {self.data['__header__']}")
        Console.add_lines(f"Path: {self.path}")
        Console.printf_collected_lines(status="success")

        return self.combined_trajectories

    def print_loaded_data(self):
        """
        To print the loaded keys and corresponding values. This excludes the dGradValues due to huge amount of data.

        :return: Nothing
        """
        Console.add_lines(f"Loaded data:")
        for key in self.important_keys_mat_file:
            if key != "dGradValues":
                Console.add_lines(f"{key}: {self.combined_trajectories[key]}")
        Console.add_lines(f"dGradValues: Not displayed here!")
        Console.printf_collected_lines("info")


if __name__ == "__main__":
    configurator = Configurator(path_folder="/home/mschuster/projects/Synthetic_MRSI/config/OLD/", file_name="config_19082024.json")
    configurator.load()

    # trajectories = CRTTrajectoriesMAT(configurator=configurator, trajectory_type="crt")
    # trajectories.load_from_mat_file()
    # trajectories.load_from_json_file()
    # trajectories.print_loaded_data()

    configurator = Configurator(path_folder="/home/mschuster/projects/Synthetic_MRSI/config/", file_name="paths_25092024.json")
    trajectories = Trajectory(configurator=configurator)
    trajectories.load_concentric_rings()

    # coilSensitivityMaps = CoilSensitivityMaps(configurator=configurator)
    # coilSensitivityMaps.load_h5py()
    # coilSensitivityMaps.interpolate(target_size=(32, 112, 128, 80), compute_on_device='cpu')
