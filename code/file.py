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
from collections import Counter

from inputs import Configurator
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

from interface import WorkingSourceInterface, PlotInterface
from spectral_spatial_simulation import FID as SpectralSpatialFID

# Just for type annotations
if TYPE_CHECKING:
    from sampling_simulation import CoilSensitivityVolume as SamplingCoilSensitivityVolume


# For enabling to use units
#u = pint.UnitRegistry()
from units import u

class BasisSet:

    """
    To load data from basis sits of different file format. If possible use the FID class in this module, since it
    incorporates this Class and also has a to_working() method to transform to FID class to be able to work with
    (e.g., interpolation, summation, general manipulation, ...)

    """

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
            Console.printf("error", "No compound name(s) provided", verbose=verbose)
            return

        # Give error if any required key is missing in the dictionary
        required_keys = ['parameters', 'signal', 'time', 'name']
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            Console.printf("error", f"Key(s) missing in data: {missing_keys}", verbose=verbose)
            return

        # Find which of the requested names exist and which do not:
        #  -> 'found' keeps the original order of compound_name
        #  -> 'missing' is just for the warning message
        found = [n for n in compound_name if n in data['name']]
        missing = [n for n in compound_name if n not in data['name']]

        # Give error if not a single requested name was found
        if len(found) == 0:
            Console.printf("error", f"None of the requested name(s) found: {compound_name}", verbose=verbose)
            return

        # Just a warning if some names are missing but at least one is found
        if missing:
            Console.printf("warning", f"Name(s) not found and skipped: {missing}", mute=False)

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
        Console.add_lines(f"Changed the data type of the FID signal(s):")
        Console.add_lines(f"   signal: {str(signal_dtype_before):<{w}} -> {signal.dtype}")
        Console.add_lines(f"   time:   {str(time_vector_dtype_before):<{w}} -> {time_vector.dtype}")
        #Console.add_lines("\n")
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


class _MultiVolumeFile:
    """
    (!) Note: This is used in class: _Volume

    Helper class for files which can contain multiple arrays inside one file.

    Supported file types:
        * .h5 / .hdf5 : multiple datasets in one file
        * .npz        : multiple arrays in one zipped NumPy archive

    This class only unwraps one physical file into multiple array entries.
    It does not check the array shape and it does not stack the data.

    The final shape check, stacking, precision conversion and output formatting
    are handled by _Volume.
    """

    file_extensions = [".h5", ".hdf5", ".npz"]

    @staticmethod
    def load(path,
             suffix: str = None,
             wanted_raw: set[str] | None = None,
             verbose: bool = False) -> list[dict]:
        """
        To automatically decide based on the file type how the multi-volume file
        should be unwrapped.

        :param path: path to one .h5/.hdf5/.npz file
        :param suffix: optional suffix; if None, inferred from path
        :param wanted_raw: optional set of internal array/dataset names to keep
        :param verbose: if it should be printed to the console or not
        :return: list of array entries
        """

        if suffix is None:
            suffix = path.suffix.lower()

        if suffix not in _MultiVolumeFile.file_extensions:
            Console.printf("error",
                           f"Invalid multi-volume file extension: {suffix}. "
                           f"Only possible: {_MultiVolumeFile.file_extensions}",
                           mute=not verbose)
            return []

        if suffix == ".npz":
            return _MultiVolumeFile._load_npz(
                path=path,
                wanted_raw=wanted_raw,
                verbose=verbose,
            )

        elif suffix in [".h5", ".hdf5"]:
            return _MultiVolumeFile._load_h5(
                path=path,
                wanted_raw=wanted_raw,
                verbose=verbose,
            )

        else:  # TODO: Maybe will never be reached!
            Console.printf("error",
                           f"Invalid multi-volume file extension: {suffix}. "
                           f"Only possible: {_MultiVolumeFile.file_extensions}",
                           mute=not verbose)
            return

    @staticmethod
    def _load_npz(path,
                  wanted_raw: set[str] | None = None,
                  verbose: bool = False) -> list[dict]:
        """
        To unwrap one .npz file into multiple array entries.

        Each array inside the archive is treated as one entry.
        No spatial metadata is available in .npz by default.

        :param path: path to one .npz file
        :param wanted_raw: optional set of array names to keep
        :param verbose: if it should be printed to the console or not
        :return: list of array entries
        """

        loaded_data = []

        with np.load(str(path)) as archive:
            for name in archive.files:

                # Early skip: array not in requested subset
                if wanted_raw is not None and name not in wanted_raw:
                    continue

                array = archive[name]

                loaded_data.append({
                    'path': path,
                    'name': name,
                    'volume': array * u.dimensionless,
                    'voxel_size': None,
                    'affine': None,
                    'orientation': None,
                })

        return loaded_data

    @staticmethod
    def _load_h5(path,
                 wanted_raw: set[str] | None = None,
                 verbose: bool = False) -> list[dict]:
        """
        To unwrap one .h5/.hdf5 file into multiple array entries.

        Convention used here:
            * Each top-level Dataset in a file = one entry
            * The dataset key becomes the entry name
            * File-level attributes 'voxel_size' and 'affine' are used if present

        This method does not check the dimensionality of the datasets.

        :param path: path to one .h5/.hdf5 file
        :param wanted_raw: optional set of dataset names to keep
        :param verbose: if it should be printed to the console or not
        :return: list of array entries
        """

        loaded_data = []

        with h5py.File(path, 'r') as file:

            # File-level attributes for spatial info
            voxel_size = file.attrs.get('voxel_size', None)
            affine = file.attrs.get('affine', None)

            if voxel_size is not None:
                voxel_size = np.asarray(voxel_size) * u.mm
            else:
                Console.printf("warning",
                               f"voxel_size not in attributes of {path.name}, set to None",
                               mute=not verbose)

            if affine is not None:
                affine = np.asarray(affine)
                orientation = nib.aff2axcodes(affine)
            else:
                Console.printf("warning",
                               f"affine not in attributes of {path.name}, set to None",
                               mute=not verbose)
                orientation = None

            # Collect top-level datasets
            for name in file.keys():

                # Early skip: dataset not in requested subset
                if wanted_raw is not None and name not in wanted_raw:
                    continue

                item = file[name]

                # Ignore groups and other non-dataset objects
                if not isinstance(item, h5py.Dataset):
                    continue

                array = np.asarray(item)

                loaded_data.append({
                    'path': path,
                    'name': name,
                    'volume': array * u.dimensionless,
                    'voxel_size': voxel_size,
                    'affine': affine,
                    'orientation': orientation,
                })

        return loaded_data


class _Volume:
    """
    (!) Note: This is used in class: ParameterMaps

    To load volume data from various file formats and provide standardised output.

    Supported file types:
        * .nii / .nii.gz : NIfTI, one volume per file
        * .npy           : NumPy single array, one volume per file
        * .h5 / .hdf5    : HDF5, one or multiple volumes per file
        * .npz           : NumPy zipped archive, one or multiple volumes per file

    The output is uniform across all formats - see load() return signature.
    """

    def __init__(self,
                 path: str,
                 data_precision: int = 32,
                 extension: str = None,
                 verbose: bool = True):
        """
        The constructor. It checks if path to one folder or direct to one file is given.

        (!) Note: If path is to folder and files with mixed extensions are inside then error message will raise.
                  Then, the solution is to define the 'extension' here when creating this instance.

        :param path: the path to folder or direct to one file
        :param data_precision: the precision, e.g. 32 yields float32 and complex64
        :param extension: Only important if path to one folder is given
        :param verbose: just if output to the console should be printed
        """

        # All currently supported file types
        self.file_extensions = [".nii", ".nii.gz", ".h5", ".hdf5", ".npy", ".npz"]

        # Here the configurator is not used. It is better to stay generic with the path.
        # Check whether path to one file is directly given or to folder containing multiple files.
        # Special handling is needed for .nii.gz because pathlib's .suffix only sees ".gz".
        self.path = PathTools.collect_files(path=path, extension=extension, verbose=False)

        if self.path is None:
            Console.printf("error",
                           f"Could not find any files or mixed files are available. "
                           f"If files with mixed extensions in the folder try to specify 'extension' here.")
            raise FileNotFoundError(f"Could not collect volume files from path: {path}")

        # The data precision is later handled by tools.ArrayTools.to_precision
        # -> Only affects the moment the volume array is converted
        self.data_precision: int = data_precision

        # Just if any output should be printed to the console
        self.verbose = verbose

    def load(self,
             subset_names: str | list[str] = None,
             rename: dict[str, str] = None,
             subset_inside_file: bool = True) -> dict:
        """
        To automatically decide based on the file type which method should be called. The called methods
        are returning then a standardised output.

        The following example should show what is recommended:
            => If you want to get just a 'subset' of all files in a folder, pass the following:

                rename={
                    "MetMap_Glu_con_map_TargetRes_HiRes": "Glu",
                    "MetMap_Gln_con_map_TargetRes_HiRes": "Gln",
                    "MetMap_NAA_con_map_TargetRes_HiRes": "NAA",
                    "MetMap_Cr+PCr_con_map_TargetRes_HiRes": "Cr+PCr"
                    }

            => If you too just want get a subset of all files, then you now can use the 'renamed' names:

                    subset_names=["Glu", "Gln", "NAA", "Cr+PCr"] ... also possible to use less, but why then renaming first?

            => If you have a file .h5/.hdf5 or .npz, possible multiple maps are in one file, therefore also
                able to filter inside the file with

                    subset_inside_file=True


        :param subset_names: optional list of volume names to keep (post-rename names if 'rename' given)
        :param rename: optional dict {current_name: new_name} applied after loading
        :param subset_inside_file: if True, skip unwanted volumes already at read time for h5/npz
                                   (default True; set False to load everything then subset afterwards)
        :return: dictionary with parameters, data, name
        """

        # Get first file type (extension) and check if supported
        path = self.path
        suffix = PathTools.get_file_extension(path=path[0], possible_extensions=self.file_extensions)

        if suffix not in self.file_extensions:
            Console.printf("error", f"Invalid file extension: {suffix}. Only possible: {self.file_extensions}")
            return

        # Make sure subset_names is always a list internally
        if isinstance(subset_names, str):
            subset_names = [subset_names]

        # Build the set of raw pre-rename names to keep, if a subset was requested.
        # Needed for early file/dataset filtering inside the format loaders.
        wanted_raw = None
        if subset_names is not None:
            wanted_raw = self._resolve_subset_to_raw_names(
                subset_names=subset_names,
                rename=rename,
            )

        # Case 1: NIfTI - one volume per file
        if suffix in [".nii", ".nii.gz"]:
            loaded_data = self._load_nifti(
                verbose=self.verbose,
                wanted_raw=wanted_raw,
            )

        # Case 2: NumPy single array - one volume per file
        elif suffix == ".npy":
            loaded_data = self._load_npy(
                verbose=self.verbose,
                wanted_raw=wanted_raw,
            )

        # Case 3: HDF5 / NPZ - one or multiple volumes per file
        elif suffix in [".h5", ".hdf5", ".npz"]:
            loaded_data = self._load_multi_volume_file(
                suffix=suffix,
                verbose=self.verbose,
                wanted_raw=wanted_raw if subset_inside_file else None,
            )

        else:  # TODO: Maybe will never be reached!
            Console.printf("error", f"Invalid file extension: {suffix}. Only possible: {self.file_extensions}")
            return

        # Build final standard output from the volume entries
        data = self._build_output(
            loaded_data=loaded_data,
            verbose=self.verbose,
        )

        if data is None:
            return

        # Apply rename FIRST so that subset_names can refer to the new working names
        if rename is not None:
            data = self.rename_volumes(
                data=data,
                name_mapping=rename,
                verbose=self.verbose,
            )

        # Final subset call. After early filtering this is mostly idempotent,
        # but it still gives the warning message for requested names that were not found.
        if subset_names is not None:
            data = self.get_subset_by_name(
                data=data,
                volume_name=subset_names,
                verbose=self.verbose,
            )

        if data is None:
            return

        Console.printf("success",
                       f"Loaded {len(data['name'])} volume(s)! \n"
                       f"  names: {data['name']}",
                       mute=not self.verbose)

        return data

    def write(self):
        # TODO: Write to NIfTI?
        raise NotImplementedError

    def get_subset_by_name(self,
                           data: dict,
                           volume_name: str | list[str],
                           verbose: bool = False) -> dict:
        """
        To get a subset of the data dictionary based on one or multiple volume names.

        :param data: the dictionary holding the data of all volumes and parameters
        :param volume_name: one volume name as string or a list of volume names
        :param verbose: if True then print to console
        :return: dictionary with only subset
        """

        # Check if volume name is of type string
        if isinstance(volume_name, str):
            volume_name = [volume_name]

        # Give error if no names provided
        if len(volume_name) == 0:
            Console.printf("error", "No volume name(s) provided", verbose=verbose)
            return

        # Give error if any required key is missing in the dictionary
        required_keys = ['parameters', 'data', 'name']
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            Console.printf("error", f"Key(s) missing in data: {missing_keys}", verbose=verbose)
            return

        # Find which of the requested names exist and which do not
        found = [n for n in volume_name if n in data['name']]
        missing = [n for n in volume_name if n not in data['name']]

        # Give error if not a single requested name was found
        if len(found) == 0:
            Console.printf("error", f"None of the requested name(s) found: {volume_name}", verbose=verbose)
            return

        # Just a warning if some names are missing but at least one is found
        if missing:
            Console.printf("warning", f"Name(s) not found and skipped: {missing}", mute=False)

        # Get all the indices of the found names and then build the subset
        indices = [data['name'].index(n) for n in found]

        subset = {
            'parameters': data['parameters'],
            'data': data['data'][indices, ...],
            'name': [data['name'][i] for i in indices],
        }

        return subset


    def rename_volumes(self,
                       data: dict,
                       name_mapping: dict[str, str],
                       verbose: bool = False) -> dict:
        """
        To apply a renaming map to volume names. Useful when raw filenames are not the desired
        working names. As example:

        name_mapping = {  "MetMap_Glu_con_map_TargetRes_HiRes": "Glu",
                          "MetMap_Gln_con_map_TargetRes_HiRes": "Gln",
                          "MetMap_NAA_con_map_TargetRes_HiRes": "NAA" }

        Names not present in the mapping are left unchanged.

        :param data: the dictionary holding the loaded volumes
        :param name_mapping: dict {current_name: new_name}
        :param verbose: if True then print to console
        :return: data dict with updated 'name' list
        """

        renamed = []
        for current_name in data['name']:
            if current_name in name_mapping:
                renamed.append(name_mapping[current_name])
            else:
                renamed.append(current_name)

        # Just info about which mapping keys did not match any loaded volume
        unmatched_keys = [k for k in name_mapping.keys() if k not in data['name']]
        if unmatched_keys:
            Console.printf("warning",
                           f"Rename keys did not match any loaded volume: {unmatched_keys}",
                           mute=not verbose)

        data = dict(data)  # shallow copy so we do not mutate the input
        data['name'] = renamed

        return data

    def _load_nifti(self,
                    verbose: bool = False,
                    wanted_raw: set[str] | None = None) -> list[dict]:
        """
        To load NIfTI files. Each file is treated as one 3D volume.

        :param verbose: if it should be printed to the console or not
        :param wanted_raw: optional set of raw file stems to keep
        :return: list of volume entries
        """

        loaded_data = []

        for path in self.path:
            name = PathTools.get_file_stem(path=path, possible_extensions=self.file_extensions,)

            # Early skip: file not in requested subset
            if wanted_raw is not None and name not in wanted_raw:
                continue

            img = nib.load(str(path))
            volume = np.asarray(img.dataobj)

            # Force to 3D - if 4D with singleton last axis, squeeze; otherwise skip
            if volume.ndim == 4 and volume.shape[-1] == 1:
                volume = volume[..., 0]

            if volume.ndim != 3:
                Console.printf("warning",
                               f"NIfTI file {path.name} has shape {volume.shape}; "
                               f"only 3D volumes supported - skipping.",
                               mute=not verbose)
                continue

            # Affine and derived spatial parameters
            affine = np.asarray(img.affine)
            voxel_size = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0)) * u.mm
            orientation = nib.aff2axcodes(affine)

            loaded_data.append({
                'path': path,
                'name': name,
                'volume': volume * u.dimensionless,
                'voxel_size': voxel_size,
                'affine': affine,
                'orientation': orientation,
            })

        if len(loaded_data) == 0 and wanted_raw is not None:
            Console.printf("error",
                           f"No NIfTI files match the requested subset: {sorted(wanted_raw)}",
                           mute=not verbose)

        return loaded_data

    def _load_npy(self,
                  verbose: bool = False,
                  wanted_raw: set[str] | None = None) -> list[dict]:
        """
        To load .npy files. Each file is treated as one 3D volume.
        No spatial metadata is available in .npy by default.

        :param verbose: if it should be printed to the console or not
        :param wanted_raw: optional set of raw file stems to keep
        :return: list of volume entries
        """

        loaded_data = []

        for path in self.path:
            name = PathTools.get_file_stem(path=path, possible_extensions=self.file_extensions,)

            # Early skip: file not in requested subset
            if wanted_raw is not None and name not in wanted_raw:
                continue

            volume = np.load(str(path))

            if volume.ndim != 3:
                Console.printf("warning",
                               f".npy file {path.name} has shape {volume.shape}; "
                               f"only 3D arrays supported - skipping.",
                               mute=not verbose)
                continue

            loaded_data.append({
                'path': path,
                'name': name,
                'volume': volume * u.dimensionless,
                'voxel_size': None,
                'affine': None,
                'orientation': None,
            })

        if len(loaded_data) == 0 and wanted_raw is not None:
            Console.printf("error",
                           f"No .npy files match the requested subset: {sorted(wanted_raw)}",
                           mute=not verbose)

        return loaded_data

    def _load_multi_volume_file(self,
                                suffix: str,
                                verbose: bool = False,
                                wanted_raw: set[str] | None = None) -> list[dict]:
        """
        To load files which may contain multiple 3D volumes inside one physical file.

        This method makes .h5/.hdf5/.npz behave like multiple individual volume files:
        one internal array/dataset becomes one volume entry.

        :param suffix: file suffix
        :param verbose: if it should be printed to the console or not
        :param wanted_raw: optional set of internal array/dataset names to keep
        :return: list of volume entries
        """

        loaded_data = []

        for path in self.path:
            loaded_data.extend(
                _MultiVolumeFile.load(
                    path=path,
                    suffix=suffix,
                    wanted_raw=wanted_raw,
                    verbose=verbose,
                )
            )

        if len(loaded_data) == 0 and wanted_raw is not None:
            Console.printf("error",
                           f"No 3D arrays/datasets match the requested subset: {sorted(wanted_raw)}",
                           mute=not verbose)

        return loaded_data

    def _check_volume_common_parameters(self,
                                        loaded_data: list[dict],
                                        verbose: bool = False):
        """
        To check if all loaded volumes share the parameters needed for one stacked output:
        same shape, same voxel size, same affine. Mismatching volumes are skipped.

        :param loaded_data: list with volume entries
        :param verbose: if it should be printed to the console or not
        :return: list with filtered data, common voxel_size, affine, orientation
        """

        reference = loaded_data[0]
        checked_data = [reference]
        skipped_files = []

        for data in loaded_data[1:]:
            skip_reasons = []

            # Same 3D shape is required for stacking
            if data['volume'].shape != reference['volume'].shape:
                skip_reasons.append(f"shape {data['volume'].shape} vs {reference['volume'].shape}")

            # Same voxel size, if both are available
            if data['voxel_size'] is not None and reference['voxel_size'] is not None:
                if not np.allclose(
                        data['voxel_size'].to(u.mm).magnitude,
                        reference['voxel_size'].to(u.mm).magnitude
                ):
                    skip_reasons.append("voxel size")

            # Same affine, if both are available
            if data['affine'] is not None and reference['affine'] is not None:
                if not np.allclose(data['affine'], reference['affine']):
                    skip_reasons.append("affine")

            if skip_reasons:
                skipped_files.append(
                    f"  - {data['path'].name} / {data['name']}: {', '.join(skip_reasons)}"
                )
            else:
                checked_data.append(data)

        # Print collected skipped files
        if skipped_files:
            Console.printf(
                "warning",
                f"Skipped volumes because following parameters do not match "
                f"{reference['path'].name} / {reference['name']}:\n"
                + "\n".join(skipped_files),
                mute=not verbose
            )

        return (checked_data,
                reference['voxel_size'],
                reference['affine'],
                reference['orientation'])

    def _build_output(self,
                      loaded_data: list[dict],
                      verbose: bool = False) -> dict:
        """
        To build the final standardised output from a list of volume entries.

        :param loaded_data: list of volume entries
        :param verbose: if it should be printed to the console or not
        :return: dictionary with parameters, data, name
        """

        if loaded_data is None or len(loaded_data) == 0:
            Console.printf("error", "No usable volumes found.", mute=not verbose)
            return

        # Check if all volumes can be stacked together
        loaded_data, voxel_size, affine, orientation = self._check_volume_common_parameters(
            loaded_data=loaded_data,
            verbose=verbose,
        )

        # Stack volumes to desired shape: (volumes, x, y, z)
        volumes = np.stack([data['volume'].magnitude for data in loaded_data], axis=0) * u.dimensionless
        names = [data['name'] for data in loaded_data]

        # Warn on duplicate names
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        if duplicates:
            Console.printf("warning",
                           f"Duplicate volume names: {duplicates}. "
                           f"Subset operations will only pick the first occurrence.",
                           mute=not verbose)

        # Convert to desired data type
        volumes_dtype_before = volumes.dtype
        volumes = ArrayTools.to_precision(
            array=volumes,
            precision=self.data_precision,
            verbose=False,
        )
        Console.add_lines(f"      data: {str(volumes_dtype_before)} -> {volumes.dtype}")

        # Uniform formatted output
        return {
            'parameters': {
                'voxel_size': voxel_size,
                'affine': affine,
                'orientation': orientation,
                'unit': u.dimensionless,
                'nucleus': None,
            },
            'data': volumes,
            'name': names,
        }

    @staticmethod
    def _resolve_subset_to_raw_names(subset_names: list[str],
                                     rename: dict[str, str] | None) -> set[str]:
        """
        To translate the user-facing 'subset_names' after rename into the raw names
        as they appear on disk.

        Mapping rules:
            * No rename given:
                subset_names are the raw names directly.
            * Rename given:
                for each subset_name 's', take all raw names 'r' with rename[r] == s.
                Additionally include 's' itself if it is not a rename key.

        :param subset_names: list of user-facing volume names
        :param rename: optional dict {raw_name: working_name}
        :return: set of raw names to load
        """

        if rename is None:
            return set(subset_names)

        wanted_raw: set[str] = set()

        for sname in subset_names:

            # Raw names that rename to this subset name
            mapped_from = [raw_name for raw_name, working_name in rename.items()
                           if working_name == sname]

            if mapped_from:
                wanted_raw.update(mapped_from)

            # If the subset name itself is not a rename key, it would pass through unchanged
            if sname not in rename:
                wanted_raw.add(sname)

        return wanted_raw


class ParameterMaps(WorkingSourceInterface[ParameterVolume], PlotInterface):
    """
    This class can be used at the moment for various purposes:

    1) To load data:
        -> from one nii file, yields dictionary of just values
        -> from one h5 file, yields dictionary with the keys inside the file
        -> from multiple nii files in a respective dictionary

    2) Transform to 'working map' from spatial metabolic distribution
        Therefore, create working map object with data from this class:
            -> map_type_name
            -> loaded maps
            -> loaded maps unit

    The actual file loading is delegated to the _Volume class.
    """

    def __init__(self, configurator: Configurator, map_type_name: str):
        self.configurator = configurator
        self.configurator.load()

        self.map_type_name = map_type_name  # e.g. B0, B1, metabolites
        self.map_key = None
        self.main_path = None

        self.loaded_maps: dict[str, np.ndarray] = None
        self.loaded_maps_unit = None

        # Spatial information loaded by _Volume, if available
        self.affine = None
        self.voxel_size = None
        self.orientation = None

        # Needed for robust stem / extension handling, especially ".nii.gz"
        self.file_extensions = [".nii", ".nii.gz", ".h5", ".hdf5", ".npy", ".npz"]

    def load(self,
             map_key: str = None,
             compounds: str | list[str] = None,
             working_name_and_file_name: dict[str, str] = None,
             data_precision: int = 32,
             verbose: bool = True) -> Self:
        """
        It is possible to load just one volume of arbitrary shape or a set of volumes at once
        that have the same shape.

        Note: To load parameter maps using the generic _Volume class. Therefore, the loading is
        delegated to the file._Volume class.

        :param map_key: key in the configurator. If None, self.map_type_name is used
        :param compounds: optional working name or list of working names to keep
        :param working_name_and_file_name: optional dict {working_name: file_name}
                                           mostly for metabolite map folders
        :param data_precision: the precision, e.g. 32 yields float32
        :param verbose: if it should be printed to the console
        :return: the object of the whole class
        """

        # If no map_key is given, use the map_type_name also as config key
        if map_key is None:
            map_key = self.map_type_name

        self.map_key = map_key

        map_config = self._get_map_config(map_key=self.map_key)
        if map_config is None:
            return self

        self.main_path = Path(map_config["path"])
        self._load_and_assign_pint_unit(map_config=map_config)

        # Check if compounds is of type string
        if isinstance(compounds, str):
            compounds = [compounds]

        rename = None
        extension = None
        subset_names = compounds

        # Case 1: working names are given directly
        # working_name_and_file_name = {"Glu": "MetMap_Glu_con_map_TargetRes_HiRes.nii", ...}
        #
        # _Volume.rename expects the opposite:
        # {"MetMap_Glu_con_map_TargetRes_HiRes": "Glu", ...}
        if working_name_and_file_name is not None:
            rename = {}

            for working_name, file_name in working_name_and_file_name.items():
                raw_name = PathTools.get_file_stem(
                    path=file_name,
                    possible_extensions=self.file_extensions,
                )
                rename[raw_name] = working_name

            # If no compounds are explicitly requested, only load the maps given in the dictionary
            if subset_names is None:
                subset_names = list(working_name_and_file_name.keys())

            # If the folder contains mixed files, _Volume / PathTools can use this extension filter
            first_file_name = next(iter(working_name_and_file_name.values()))
            extension = PathTools.get_file_extension(
                path=first_file_name,
                possible_extensions=self.file_extensions,
            )

        # Load all maps through _Volume
        volume_data = _Volume(
            path=str(self.main_path),
            data_precision=data_precision,
            extension=extension,
            verbose=verbose,
        ).load(
            subset_names=subset_names,
            rename=rename,
        )

        if volume_data is None:
            Console.printf("error", f"Could not load parameter maps for {self.map_type_name}", mute=not verbose)
            return self

        # Store spatial information from _Volume
        self.affine = volume_data["parameters"].get("affine", None)
        self.voxel_size = volume_data["parameters"].get("voxel_size", None)
        self.orientation = volume_data["parameters"].get("orientation", None)

        # Convert _Volume output:
        #   data shape: (n_maps, ...)
        # to ParameterMaps output:
        #   self.loaded_maps = {name: map_array}
        values = volume_data["data"]

        # _Volume returns a pint Quantity, therefore use magnitude for raw numerical arrays
        if hasattr(values, "magnitude"):
            values = values.magnitude

        names = volume_data["name"]

        # For one single nii/npy file, use the map_type_name as dictionary key
        if self.main_path.is_file():
            file_extension = PathTools.get_file_extension(
                path=self.main_path,
                possible_extensions=self.file_extensions,
            )

            if file_extension in [".nii", ".nii.gz", ".npy"] and len(names) == 1 and rename is None:
                names = [self.map_type_name]

        self.loaded_maps = {}

        Console.add_lines("Loaded maps: ")

        for i, name in enumerate(names):
            loaded_map = values[i, ...]

            self.loaded_maps[name] = loaded_map

            Console.add_lines(
                f"  {i}: working name: {name:.>10} | {'Shape:':<8} {loaded_map.shape} | "
                f"Values range: [{round(np.nanmin(loaded_map), 3)}, {round(np.nanmax(loaded_map), 3)}] "
                f"| Unit: {self.loaded_maps_unit} |"
            )

        if verbose:
            Console.printf_collected_lines("success")

        return self

    def _get_map_config(self, map_key: str) -> dict:
        """
        To get the map configuration entry from the configurator.

        The map_key can either be a direct key in self.configurator.data
        or a nested key separated by dots.

        :param map_key: key to the map entry in the configurator
        :return: dictionary holding the map config
        """

        maps_config = self.configurator.data

        # Case 1: direct key, also works if the key itself contains dots
        if map_key in maps_config:
            return maps_config[map_key]

        # Case 2: nested key, separated by dots
        config = maps_config
        for key in map_key.split("."):
            if key not in config:
                Console.printf("error", f"Could not find map_key '{map_key}' in configurator.")
                return None

            config = config[key]

        return config

    def combine_to_complex(self, real_key: str = "real", imag_key: str = "imag", output_key: str = None, verbose: bool = True):
        """
        If complex data is loaded e.g., from a h5 file than it yields the following:

            self.loaded_maps["key_real"]  = real data (e.g., as float 32)
            self.loaded_maps["key_imag"]  = imag data (e.g., as float 32)

        This method created instead: self.loaded_maps["my_key"] = complex_data (e.g., as complex 64)

        (!) Note: Therefore the initial precision is automatically maintained.
        (!) Note: Not necessary to call if no complex data is loaded!

        :param real_key: The real key in the e.g., h5 file
        :param imag_key: The imag key in the e.g., h5 file
        :param output_key: The new key where the complex data should be stored
        :param verbose: If True then prints the conversation results to the console
        :return: The object itself
        """

        # 1) Check if already loaded some maps
        if self.loaded_maps is None:
            Console.printf("error", "No maps loaded yet. Call load() before combine_to_complex()")
            return self

        # 2) Then check if real and imag keys are available
        if real_key not in self.loaded_maps:
            Console.printf("error", f"Could not find real key '{real_key}' in loaded maps.")
            return self

        if imag_key not in self.loaded_maps:
            Console.printf("error", f"Could not find imag key '{imag_key}' in loaded maps.")
            return self

        # 3) If available, get the real and imag data (stored separately)
        real = self.loaded_maps[real_key]
        imag = self.loaded_maps[imag_key]

        # 4) Then check if the shapes of real and imag match!
        if real.shape != imag.shape:
            Console.printf("error", f"Cannot combine real and imaginary parts because shape differ:"
                                    f"{real_key:.<10}: {real.shape}\n"
                                    f"{imag_key:.<10}: {imag.shape}")
            return self

        # 5) Just check if numeric, but maybe don't need?
        if not np.issubdtype(real.dtype, np.number):
            Console.printf("error", f"real key '{real_key}' is not numeric: dtype={real.dtype}")
            return self

        if not np.issubdtype(imag.dtype, np.number):
            Console.printf("error", f"imag key '{imag_key}' is not numeric: dtype={imag.dtype}")
            return self

        # 6) Just take the name of the map type as key
        if output_key is None:
            output_key = self.map_type_name

        # 7) Check if desired new key not already exists
        if output_key in self.loaded_maps and output_key not in {real_key, imag_key}:
            Console.printf(
                "error",
                f"output key '{output_key}' already exists in loaded maps."
            )
            return self

        # 8) Transform the imag and real array to complex data type and remove old imag and real keys
        complex_data = real + 1j * imag
        del self.loaded_maps[real_key]
        del self.loaded_maps[imag_key]
        self.loaded_maps[output_key] = complex_data

        # 9) Print information about results to the console if desired
        if verbose:
            Console.printf(
                "success",
                f"Combined '{real_key}' and '{imag_key}' into complex map '{output_key}': \n"
                f" > New shape: {complex_data.shape} \n"
                f" > Data type: {complex_data.dtype} "
            )

        return self



    def _load_and_assign_pint_unit(self, map_config: dict):
        """
        For loading the unit string, which should be compliant with the library.
        If it fails, fallback to "dimensionless" via pint.

        :param map_config: dictionary holding the map config
        :return:
        """

        unit_string = map_config.get("unit", "dimensionless")

        # Try to convert provided string in the config file to pint unit, and assign "dimensionless" if it fails
        try:
            self.loaded_maps_unit = u.Unit(unit_string)
        except:
            Console.printf("error",
                           f"Could not convert loaded unit '{unit_string}' to pint unit. "
                           f"Therefore, assigned 'dimensionless'!")
            self.loaded_maps_unit = u.dimensionless

    def to_base_units(self, verbose=False):
        """
        To convert all loaded arrays to the base units.

        :param verbose: it true then it prints conversation results to the console.
        :return: Nothing
        """

        loaded_maps_unit_before = self.loaded_maps_unit
        loaded_maps_unit_after = None

        try:
            for name, array in self.loaded_maps.items():
                self.loaded_maps[name], loaded_maps_unit_after = UnitTools.to_base(
                    values=array,
                    units=self.loaded_maps_unit,
                    return_separate=True,
                    verbose=verbose,
                )

            self.loaded_maps_unit = loaded_maps_unit_after

            Console.printf("success", f"Converted to base units: {loaded_maps_unit_before} -> {self.loaded_maps_unit}")

        except:
            Console.printf("error", "Could not convert to base units.")

    def plot(self,
             cmap: str = "viridis",
             slice_index: int = None,
             complex_mode: str = "magnitude"):
        """
        To plot the loaded maps.

        If a loaded map is 3D, it is plotted directly at the given slice.
        If a loaded map is 4D, the first axis is interpreted as map/channel axis,
        and every channel is plotted separately.

        Complex-valued maps are converted before plotting.

        :param cmap: the matplotlib colormap
        :param slice_index: the slice index in z direction. If None, the central slice is used
        :param complex_mode: how complex-valued maps should be plotted.
                             Possible: "magnitude", "phase", "real", "imag"
        :return: matplotlib figure and axes
        """

        if self.loaded_maps is None:
            Console.printf("error", "No maps loaded yet, thus no plotting possible.")
            return

        plot_maps = {}

        # Prepare all maps before plotting
        for key, value in self.loaded_maps.items():

            # If complex, select what should be shown
            if np.iscomplexobj(value):

                if complex_mode == "magnitude":
                    value = np.abs(value)

                elif complex_mode == "phase":
                    value = np.angle(value)

                elif complex_mode == "real":
                    value = value.real

                elif complex_mode == "imag":
                    value = value.imag

                else:
                    Console.printf(
                        "error",
                        f"Unknown complex_mode '{complex_mode}'. "
                        f"Possible are: magnitude, phase, real, imag"
                    )
                    return

            # Case 1: one 3D map
            if value.ndim == 3:
                plot_maps[key] = value

            # Case 2: multiple 3D maps in one 4D array
            elif value.ndim == 4:
                for i in range(value.shape[0]):
                    plot_maps[f"{key}_{i}"] = value[i, ...]

            else:
                Console.printf(
                    "error",
                    f"Could not plot map '{key}' with shape {value.shape}. "
                    f"Only 3D maps (X, Y, Z) or 4D maps (C, X, Y, Z) are supported."
                )
                return

        number_maps = len(plot_maps.items())

        if number_maps == 0:
            Console.printf("error", "No maps available for plotting.")
            return

        fig, axs = plt.subplots(
            figsize=(3 * number_maps, 3),
            ncols=number_maps,
            nrows=1,
        )

        # Makes list if just one map, else already list
        axs = [axs] if number_maps == 1 else axs

        image = None

        for i, (key, value) in enumerate(plot_maps.items()):

            # Use central slice if no slice index is given
            z_index = value.shape[2] // 2 if slice_index is None else slice_index

            # Check if slice index is inside the volume
            if z_index < 0 or z_index >= value.shape[2]:
                Console.printf(
                    "error",
                    f"slice_index {z_index} is outside map '{key}' with z shape {value.shape[2]}."
                )
                return

            image = axs[i].imshow(value[:, :, z_index], cmap=cmap)
            axs[i].set_title(key)
            axs[i].set_axis_off()

        fig.colorbar(image, ax=axs, location="right")

        return fig, axs


###    def plot(self, cmap="viridis"):
###        number_maps = len(self.loaded_maps.items())
###        fig, axs = plt.subplots(figsize=(15, 2), ncols=number_maps, nrows=1)
###
###        # makes list if just one map, else already list:
###        axs = [axs] if number_maps == 1 else axs
###
###        for i, (key, value) in enumerate(self.loaded_maps.items()):
###            image = axs[i].imshow(value[:, :, 40], cmap=cmap)
###            axs[i].set_title(key)
###            axs[i].set_axis_off()
###
###        fig.colorbar(image, ax=axs, location="right")

    def plot_jupyter(self, cmap="gray"):
        """
        To plot the loaded volume in an interactive form for the jupyter notebook/lab.
        (!) Note: %matplotlib ipympl need be called once in the respective jupyter notebook.

        :return: Nothing
        """
        if self.loaded_maps is not None:
            JupyterPlotManager.volume_grid_viewer(
                vols=self.loaded_maps.values(),
                rows=1,
                cols=len(self.loaded_maps.keys()),
                titles=self.loaded_maps.keys(),
                cmap=cmap,
            )
        else:
            Console.printf("error", "No maps loaded yet, thus no plotting possible.")

    def to_working(self, data_type: str = None, verbose: bool = True) -> ParameterVolume:
        """
        This transforms the loaded maps directly to a 4D array of (metabolites, X, Y, Z)

        :return: a ParameterVolume (see module spatial_simulation)
        """

        # Create the Parameter _Volume
        volume = ParameterVolume(maps_type=self.map_type_name)

        # Create for each metabolite a Parameter Map and add it to the Parameter _Volume
        for metabolite, data in self.loaded_maps.items():
            parameter_map = ParameterMap(
                map_type=self.map_type_name,
                metabolite_name=metabolite,
                values=data,
                unit=self.loaded_maps_unit,
                affine=self.affine,
            )
            volume.add_map(map_object=parameter_map, verbose=False)

        # To create from all the 3D volumes inside the ParameterVolume actually one 4D volume
        volume.to_volume(verbose=verbose)

        # Change data type if given by the user
        if data_type is not None:
            volume.to_data_type(data_type)

        return volume


class FID(WorkingSourceInterface[SpectralSpatialFID]):
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
        Console.add_lines("\nAdded FID signal(s):")
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


class Trajectory:
    """
    To read the parameters and gradient values from JSON trajectories files.

    This requires files:
        * JSON Structure file
        * JSON Gradient file
    """

    # Class variable (thus not bound to an object)
    #u = pint.UnitRegistry()  # for using units with the same Registry

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



###################################### FROM HERE DEPRECATED CLASSES ##################################################

@deprecated(reason="Seems not to serve a purpose any more!")
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



# TODO: Use ParameterMaps instead!!! & REMOVE THIS IN FUTURE!
@deprecated(reason="Use ParameterMaps instead")
class Mask(PlotInterface):
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

@deprecated(reason="Maybe not necessary at the moment; more general class could handle this")
class MetabolicAtlas:
    # TODO

    def __init__(self):
        # TODO
        pass

    def load(self):
        # TODO
        pass


@deprecated(reason="Use ParameterMaps instead")
class CoilSensitivityMaps(WorkingSourceInterface["SamplingCoilSensitivityVolume"], PlotInterface):
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
        PlotInterface one slice per coil.
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
