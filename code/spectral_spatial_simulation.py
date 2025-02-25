from __future__ import annotations      #TODO: due to circular import. Maybe solve different!
from typing import TYPE_CHECKING        #TODO: due to circular import. Maybe solve different!

if TYPE_CHECKING:                       #TODO: due to circular import. Maybe solve different!
    from spatial_metabolic_distribution import MetabolicPropertyMap  #TODO: due to circular import. Maybe solve different!

from cupyx.scipy.interpolate import interpn as interpn_gpu
from scipy.interpolate import interpn as interpn_cpu
from dask.diagnostics import ProgressBar
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tools import CustomArray
from dask.array import Array
from printer import Console
import dask.array as da
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import numpy as np
# import numba
# from numba import cuda
import tools
import cupy
import time
import cupy as cp
import dask
import sys
import os


import xarray as xr


class FID:
    """
    The FID includes the basic attributes, including the signal and time vector, as
    well as the name of the chemical compound refereed to it. Further, the T2 value and
    the concentration.
    Also, it is possible to get the signal in various data types and thus, if
    necessary, decrease the memory load.
    """

    def __init__(self,
                 signal: np.ndarray = None,
                 time: np.ndarray = None,
                 name: list[str] = None,
                 signal_data_type: np.dtype = None,
                 sampling_period: float = None):
        """
        A checks if the shape of the time vector equals the signal vector is performed. If false then the program quits.
        Further, it is also possible to instantiate a class containing just "None".
        """

        # If FID is instantiated not empty
        if signal is not None and time is not None:
            # Check if the length of the time and signal vector equals
            if not signal.shape[-1] == time.shape[-1]:
                Console.printf("error",
                               f"Shape of signal and time vector does not match! Signal length is {signal.shape[-1]} while time length is {time.shape[-1]}. Terminating the program!")
                sys.exit()

        # Check if the user defined a preferred data type
        self.signal_data_type = signal_data_type
        if signal_data_type is not None:
            self.signal: np.ndarray = signal.astype(self.signal_data_type)
        else:
            self.signal: np.ndarray = signal  # signal vector

        self.time: np.ndarray = time  # time vector
        self.name: list = name  # name, e.g., of the respective metabolite
        self.concentration: float = None
        self.t2_value: np.ndarray = None  # TODO
        self.sampling_period = sampling_period  # it is just 1/(sampling frequency)

        self._iter_index = 0

    def __iter__(self):
        """
        For getting the iteration object.

        :return: returning the current object as the iteration object.
        """
        return self

    def __next__(self):
        """
        If more than one FID signal then it returns the next FID signal, corresponding to the respective metabolite, as new object.

        :return:
        """

        if self._iter_index >= self.signal.shape[0]:
            raise StopIteration
        else:
            fid = FID(signal=self.signal[self._iter_index, :],
                      time=self.time,
                      name=[self.name[self._iter_index]],
                      signal_data_type=self.signal_data_type,
                      sampling_period=self.sampling_period)
            self._iter_index += 1

        return fid

    def merge_signals(self, names: list[str], new_name: str, divisor: int):
        """
        TODO

        :param new_name:
        :param names:
        :param divisor:
        :return:
        """
        indices = []

        signal = np.zeros(self.signal.shape[1], dtype=self.signal.dtype) # empty array of fid length
        for name in names:
            signal += self.get_signal_by_name(name).signal
            index = self.name.index(name)
            indices.append(index)

        signal *= 1/divisor

        # delete OLD entries
        self.signal = np.delete(self.signal, indices, axis=0)
        self.name = [name for name in self.name if name not in names] # delete the OLD names by getting subset

        # insert new entries
        self.signal = np.vstack((self.signal, signal))
        self.name.append(new_name)
        Console.printf("success", f"Merged signals of {names} with factor 1/{divisor}. New name of signal: {new_name}")



    def get_partly_fid(self, names: list[str]):
        """
        This creates a fid with only containing the FID signals corresponding to the names. Thus, this FID represents a subset of the whole FID!

        :param names: names of all FIDs in the current FID
        :return: new FID object with only the desired FID signals
        """
        fid = FID()

        for name in names:
            fid += self.get_signal_by_name(name)

        return fid


    def get_name_abbreviation(self) -> list[str]:
        """
        Extracts the abbreviation of each given chemical compound name. It is necessary that the string, represending the name of the signal in the FID
        contains a abbreviated for somewhere in brackets. For example: Creatine (Cr)

        Example use case:
            Creatine (Cr)+Phosphocreatine (PCr) --> Cr+PCr

        No effect if no brackets available or already abbreviated:
            Cr+PCr --> Cr+PCr

        :return: list of strings containing the abbreviations
        """
        name_abbreviation = []
        for name in self.name:
            # Search for the indices of the brackets
            start_indices = [index for index, letter in enumerate(name) if letter == "("]
            end_indices = [index for index, letter in enumerate(name) if letter == ")"]

            # CASE 1: failed to find "(" and ")" or it is already abbreviated
            if (not start_indices) & (not end_indices):
                name_abbreviation.append(name)
            # CASE 2: start to abbreviate the given name
            else:
                abbreviation = ""
                for i, (start_index, end_index) in enumerate(zip(start_indices, end_indices)):
                    # extract "(" and ")" and the text in between (the abbreviation)
                    abbreviation += name[start_index + 1:end_index]
                    # if only one abbreviation don't add "+", also not if we reached the end!
                    if (i >= 0) and (i != len(start_indices)-1):
                        abbreviation += "+"

                # append abbreviation of one signal
                name_abbreviation.append(abbreviation)

        return name_abbreviation

    def get_signal_by_name(self, compound_name: str):
        """
        TODO

        :param compound_name:
        :return:
        """
        try:
            index = self.name.index(compound_name)
            signal = self.signal[index, :]
            return FID(signal=signal, name=[compound_name], time=self.time)
        except ValueError:
            Console.printf("error", f"Chemical compound not available: {compound_name}. Please check your spelling and possible whitespaces!")
            return

    def get_signal(self, signal_data_type: np.dtype, mute=True):
        """
        To get the signal with a certain precision. Useful to reduce the required space.

        :param signal_data_type:
        :param mute: By default, True. If False, then the precision, according to the data type, is printed to the console.
        :return: Amplitude of chosen data type and thus precision.
        """
        signal = self.signal.astype(dtype=signal_data_type)
        if not mute:
            Console.printf("info", f"Get signal of {self.name} with precision of {np.finfo(signal.dtype).precision} decimal places")

        return signal

    def show_signal_shape(self) -> None:
        """
        Print the shape of the FID signal to the console.

        :return: None
        """
        Console.printf("info", f"FID Signal shape: {self.signal.shape}")

    def sum_all_signals(self):
        """
        To sum up all signals.
        """
        self.signal = np.sum(self.signal, axis=0)  # To sum up all signals
        self.name = [" + ".join(self.name)]  # To put all compound names in one string (in a list)
        return self

    def get_spectrum(self) -> np.ndarray:
        """
        To get the spectrum of each
        """
        # TODO: just check if right axis is used. Should be fft to each row here?
        # TODO: Also think about the return type! Maybe just return magnitude?
        frequency = np.fft.fftfreq(self.time.size, self.sampling_period)
        magnitude = np.fft.fft(self.signal, axis=1)
        return {'frequency': frequency,
                'magnitude': magnitude}

    def change_signal_data_type(self, signal_data_type: np.dtype) -> None:
        """
        For changing the data type of the FID. Possible usecase: convert FID signals to lower bit signal, thus reduce required space.

        :param signal_data_type: Numpy data type
        :return: Nothing
        """
        self.signal = self.signal.astype(signal_data_type)

    def __add__(self, other):
        """
        For merging different FID signals. Add two together with just "+"
        :param other:
        :return:
        """

        # Case 1: The self object is empty (e.g., used as initial object) and the other object to sum has values.
        if self.signal is None and self.time is None and self.name is None:
            # The other object need to have not None attributes!
            if other.signal is None or other.time is None or other.name is None:
                Console.printf("error", f"Not possible to sum the two FID since the 'other' object includes None in one of this attributes: signal, time, name!")
                return
            # If the other object does have not None attributes: set attributes from other object
            self.signal = other.signal
            self.time = other.time
            self.name = other.name
            return self

        # Case 2: The self and other object both do not have None attributes and need to be summed
        if not np.array_equal(self.time, other.time):
            Console.printf("error", f"Not possible to sum the two FID since the time vectors are different! Vector 1: {self.time.shape}, Vector 2; {other.times.shape}")
            return
        if not self.signal.shape[-1] == other.signal.shape[-1]:
            Console.printf("error", "Not possible to sum the two FID since the length does not match!")
            return
        else:
            fid = FID(signal=self.signal, time=self.time, name=self.name)  # new fid object containing the information of this object
            fid.signal = np.vstack((self.signal, other.signal))  # vertical stack signals, thus add row vectors
            fid.name = self.name.copy() + other.name.copy()  # since lists not immutable, copy() is required!

            return fid

    def __str__(self):
        """
        Print to the console the name(s) of the chemical compounds in the FID and the signal shape.
        """
        Console.add_lines(f"FID contains of chemical compound(s):")

        for i, compound_name in enumerate(self.name):
            Console.add_lines(f"  {i}: {compound_name}")

        Console.add_lines(f"=> with signal shape {self.signal.shape}")
        Console.printf_collected_lines("info")
        return "\n"


class Model:
    """
    For creating a model that combines the spectral and spatial information. It combines the FIDs, metabolic property maps and mask.
    """

    def __init__(self,
                 block_size: tuple,
                 TE: float,
                 TR: float,
                 alpha: float,
                 path_cache: str = None,
                 data_type: str = "complex64",
                 compute_on_device: str = "cpu",
                 return_on_device: str = "cpu"):
        # Define a caching path for dask. Required if RAM is running out of memory.
        if path_cache is not None:
            if os.path.exists(path_cache):
                self.path_cache = path_cache
            else:
                Console.printf("error", f"Terminating the program. Path does not exist: {path_cache}")
                sys.exit()
            dask.config.set(temporary_directory=path_cache)

        # The block size used for dask to compute
        self.block_size = block_size

        # Parameters defined by the user
        self.TE = TE
        self.TR = TR
        self.alpha = alpha

        self.fid = FID()  # instantiate an empty FID to be able to sum it ;)
        self.metabolic_property_maps: dict[str, MetabolicPropertyMap] = {}
        self.mask = None

        # Define the device for computation and the target device (cuda, cpu)
        self.compute_on_device = compute_on_device
        self.return_on_device = return_on_device

        # Define the data type which should be used
        self.data_type = data_type

    def model_summary(self):
        # TODO: make more beautiful (programming style) ;)
        # TODO: add units
        Console.add_lines("Spectral-Spatial-Model Summary:")
        Console.add_lines(f" TE          ... {self.TE}")
        Console.add_lines(f" TR          ... {self.TR}")
        Console.add_lines(f" alpha       ... {self.alpha}")
        Console.add_lines(f" FID length  ... {len(self.fid.signal[0])}")
        Console.add_lines(f" Metabolites ... {len(self.metabolic_property_maps)}")
        Console.add_lines(f" Model shape ... {self.mask.shape}")
        Console.add_lines(f" Block size  ... {self.block_size} [t,x,y,z]")
        Console.add_lines(f" Compute on  ... {self.compute_on_device}")
        Console.add_lines(f" Return on   ... {self.return_on_device}")
        Console.add_lines(f" Cache path  ... {self.path_cache}")
        Console.printf_collected_lines("info")


    def add_fid(self, fid: FID) -> None:
        """
        Add a FID from the class `~FID`, which can contain multiple signals, to the Model. All further added FID will perform the
        implemented __add__ in the `~FID` class. Thus, the loaded_fid will be merged. Resulting in just one fid object containing all
        signals.

        Example usage 1:
         => Add FID of metabolites
         => Add FID of lipids
         => Add FID of water simulation
         => Add FID of macromolecules simulation
        Example usage 2:
         => Add FID metabolite 1
         => Add FID metabolite 2

        :param fid: fid from the class `~FID`
        :return: Nothing
        """
        try:
            self.fid + fid  # sum fid according to the __add__ implementation in the FID class
            Console.add_lines(f"Added the following FID signals to the spectral spatial model:")
            for i, name in enumerate(fid.name):
                Console.add_lines(f"{i}: {name}")
            Console.printf_collected_lines("success")
        except Exception as e:
            Console.printf("error", f"Error in adding compound '{fid.name} to the spectral spatial model. Exception: {e}")

    def add_mask(self, mask: np.ndarray) -> None:
        """
        For adding one mask to the model. It is just a numpy array with no further information so far.

        :param mask: Numerical values of the mask as numpy array
        :return: Nothing
        """
        self.mask = mask

    def add_metabolic_property_map(self, metabolic_property_map: MetabolicPropertyMap):
        """
        Map for scaling the FID at the respective position in the volume. One map is per metabolite.

        :param metabolic_property_map: Values to scale the FID at the respective position in the volume
        :return: Nothing
        """

        Console.printf("info", f"Added the following metabolic a property map to the spectral spatial model: {metabolic_property_map.chemical_compound_name}")
        self.metabolic_property_maps[metabolic_property_map.chemical_compound_name] = metabolic_property_map

    def add_metabolic_property_maps(self, metabolic_property_maps: dict[str, MetabolicPropertyMap]):
        """
        Multiple Maps for scaling the FID at the respective position in the volume. Each map is for one metabolite.

        :param metabolic_property_maps: A dictionary containing the name as str and the respective metabolic property map
        :return: Nothing
        """
        self.metabolic_property_maps.update(metabolic_property_maps)

        Console.add_lines("Adding the following metabolic property maps to the model:")
        for i, (names, _) in enumerate(metabolic_property_maps.items()):
            Console.add_lines(f"{i}: {names}")

        Console.printf_collected_lines("success")

    @staticmethod
    def _transform_T1(xp, volume, alpha, TR, T1):
        """
        TODO: make full docstring
        Transform the volume. Therefore, alpha, TR and T1 is used.
           alpha ... scalar value (either numpy or cupy)
           TR    ... scalar value (either numpy or cupy)
           T1    ... matrix       (either numpy or cupy)
           xp    ... either numpy or cupy -> using the whole imported library (np or cp is xp)

        It needs to be a static function, otherwise it seems dask cannot handle it properly with map_blocks.
        """
        return volume * xp.sin(xp.deg2rad(alpha)) * (1 - xp.exp(-TR / T1)) / (1 - (xp.cos(np.deg2rad(alpha)) * xp.exp(-TR / T1)))

    @staticmethod
    def _transform_T2(xp, volume, time_vector, TE, T2):
        """
        TODO: make full docstring
        Transform the volume. Therefore, a time vector, TE and T2 is used.
           time vector ... vector (either numpy or cupy)
           TE          ... scalar value (either numpy or cupy)
           T2          ... matrix       (either numpy or cupy)
           xp          ... either numpy or cupy -> using the whole imported library (np or cp is xp)

        It needs to be a static function, otherwise it seems dask cannot handle it properly with map_blocks.
        """
        # The original implementation:
        #   for index, t in enumerate(time_vector):
        #       volume[index, :, :, :] = volume[index, :, :, :] * da.exp((TE * t) / T2)
        # Also, valid, and it might be faster:
        volume *= xp.exp((TE * time_vector) / T2)
        return volume

    def assemble_graph(self) -> CustomArray:
        """
        Create a computational dask graph. It can be used to compute it or to add further operations.

        :return: CustomArray with numpy or cupy, based on the selected device when created the model.
        """

        Console.printf("info", f"Start to assemble whole graph on device {self.compute_on_device}:")

        # Defining based on the selected device if cupy or numpy should be used
        xp = None
        if self.compute_on_device == "cuda":
            xp = cp  # use cupy
        elif self.compute_on_device == "cpu":
            xp = np  # use numpy

        metabolites_volume_list = []
        for fid in tqdm(self.fid, total=len(self.fid.signal)):
            # (1) Prepare the data & reshape it
            #   1a) Get the name of the metabolite (unpack from the list)
            metabolite_name = fid.name[0]
            #   1b) Reshape FID for multiplication, put to a respective device and create dask array with chuck size defined by the user
            fid_signal = fid.signal.reshape(fid.signal.size, 1, 1, 1)
            fid_signal = xp.asarray(fid_signal)
            fid_signal = da.from_array(fid_signal, chunks=self.block_size[0])
            #   1c) Reshape time vector, create dask array with block size defined by the user
            time_vector = fid.time[:, xp.newaxis, xp.newaxis, xp.newaxis]
            time_vector = da.from_array(time_vector, chunks=(self.block_size[0], 1, 1, 1))
            #   1d) Same as for FID and time vector above
            mask = self.mask.reshape(1, self.mask.shape[0], self.mask.shape[1], self.mask.shape[2])
            mask = xp.asarray(mask)
            mask = da.from_array(mask, chunks=(1, self.block_size[1], self.block_size[2], self.block_size[3]))
            #   1f) Get T1 and T2 of respective metabolite with the metabolite name
            metabolic_map_t2 = self.metabolic_property_maps[metabolite_name].t2
            metabolic_map_t1 = self.metabolic_property_maps[metabolite_name].t1

            # (2) FID with 3D mask volume yields 4d volume
            volume_with_mask = fid_signal * mask

            # (3) Based on selected device prepare data
            dtype = eval("xp." + self.data_type) # yields in e.g., xp.complex64 which can be np.complex64 or cp.complex64
            if self.compute_on_device == "cpu":
                # Creates numpy arrays with xp
                TE = xp.asarray([self.TE])
                alpha = xp.asarray([self.alpha])
                TR = xp.asarray([self.TR])
                #dtype = xp.complex64  # TODO
            elif self.compute_on_device == "cuda":
                # Creates cupy arrays with xp
                volume_with_mask = volume_with_mask.map_blocks(xp.asarray, dtype=xp.complex64)
                time_vector = time_vector.map_blocks(xp.asarray, dtype=xp.complex64)
                TE = xp.asarray([self.TE])
                metabolic_map_t2 = metabolic_map_t2.map_blocks(xp.asarray, dtype=xp.complex64)
                metabolic_map_t1 = metabolic_map_t1.map_blocks(xp.asarray, dtype=xp.complex64)
                alpha = xp.asarray([self.alpha])
                TR = xp.asarray([self.TR])
                #dtype = xp.complex64

            # (4) Include T2 effects
            volume_metabolite = da.map_blocks(Model._transform_T2,
                                              xp,  # xp is either numpy (np) or cupy (cp) based on the selected device
                                              volume_with_mask,
                                              time_vector,
                                              TE,
                                              metabolic_map_t2,
                                              dtype=dtype)

            # (3) Include T1 effects
            volume_metabolite = da.map_blocks(Model._transform_T1,
                                              xp,  # xp is either numpy (np) or cupy (cp) based on the selected device
                                              volume_metabolite,
                                              alpha,
                                              TR,
                                              metabolic_map_t1,
                                              dtype=dtype)

            # (4) Include spatial concentration
            if self.compute_on_device == "cuda":
                volume_metabolite *= (self.metabolic_property_maps[metabolite_name].concentration).map_blocks(cp.asarray, dtype='c8')
            elif self.compute_on_device == "cpu":
                volume_metabolite *= self.metabolic_property_maps[metabolite_name].concentration

            # (5) Expand dims, since it is only for one metabolite and concatenate
            volume_metabolite = da.expand_dims(volume_metabolite, axis=0)
            metabolites_volume_list.append(volume_metabolite)

        # (6) Put all arrays together -> 1,100_000,150,150,150 + ... + 1,100_000,150,150,150 = 11,100_000,150,150,150
        volume_all_metabolites = da.concatenate(metabolites_volume_list, axis=0)

        # (7) Sum all metabolites & Create CustomArray from dask Array
        volume_sum_all_metabolites = da.sum(volume_all_metabolites, axis=0)
        volume_sum_all_metabolites = CustomArray(volume_sum_all_metabolites)

        # (8) If the computation device and target device does not match then adjust it
        if self.compute_on_device == "cuda" and self.return_on_device == "cpu":
            volume_sum_all_metabolites = volume_sum_all_metabolites.map_blocks(cp.asnumpy)
        elif self.compute_on_device == "cpu" and self.return_on_device == "cuda":
            volume_sum_all_metabolites = volume_sum_all_metabolites.map_blocks(cp.asarray)  # TODO: Check if it works

        # (9) Just clarify a computational graph is returned
        computational_graph: CustomArray = volume_sum_all_metabolites

        return computational_graph

    def build(self):
        """
        Do we need this? I guess not!

        :return:
        """
        raise NotImplementedError


class Simulator:
    # TODO
    # It should be able to simulate all spectral elements.

    def __init__(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def metabolites(self) -> FID:
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def water(self) -> FID:
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def lipids(self) -> FID:
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def macromolecules(self) -> FID:
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def water_suppression(self):
        # TODO: Perform on signal of FID? Spectrum required?

        # TODO: Currently working on this.
        # TODO: I need a dictionary and an input
        #   Does this method apply / use the water suppression or also simulates the Dictionary

        raise NotImplementedError("This method is not yet implemented")

    def lipid_suppression(self):
        # TODO: Perform on signal of FID? Spectrum required?
        raise NotImplementedError("This method is not yet implemented")



class LookupTableWET:
    """
    This class is for creating a lookup table for water suppression. The technique is WET.
    The suppression is given as a ratio of suppressed signal to non-suppressed signal.
    """
    def __init__(self,
                 T1_range: np.ndarray | list = np.ndarray([300, 5000]),                 # e.g., [300, 5000] in ms
                 T1_step_size: float = 50.0,                                            # e.g., 50 ms
                 T2: float = 250,                                                       # e.g., 250
                 B1_scales_inhomogeneity: np.ndarray | list = np.ndarray([0, 2]),        # e.g., [0, 2]
                 B1_scales_gauss: np.ndarray | list = np.array([0.01, 1]),              # e.g., [0.01, 1]
                 B1_scales_inhomogeneity_step_size: float = 0.05,                       # e.g., 0.05
                 B1_scales_gauss_step_size: float = 0.05,                               # e.g., 0.05
                 TR: float = 600.0,
                 TE: float = 0.0,
                 flip_angle_excitation_degree: float = 47.0,
                 flip_angles_WET_degree: np.ndarray | list = np.array([89.2, 83.4, 160.8]),
                 time_gaps_WET: np.ndarray | list = np.ndarray([30, 30, 30])
                 ):

        # A) Fixed input variables for Bloch Simulation
        self.TR = TR
        self.TE = TE
        self.T2 = T2
        self.flip_angle_excitation_rad = np.deg2rad(flip_angle_excitation_degree)
        self.flip_angles_WET_rad = np.deg2rad(flip_angles_WET_degree)
        self.time_gaps_WET = time_gaps_WET

        # B) Variables containing ranges for generating the dictionary
        #   Generate T1 values vector
        self._T1_step_size = T1_step_size
        self._T1_range = T1_range
        self.T1_values = np.arange(T1_range[0],                 # lower border
                                   T1_range[1]+T1_step_size,    # upper border, not inclusive (see math) and thus + step size
                                   T1_step_size)                # the step size

        #   Generate B1 values (scaling values) vector
        self._B1_scales_lower_border = np.min([B1_scales_gauss[0], B1_scales_inhomogeneity[0]])
        self._B1_scales_upper_border = np.max([B1_scales_gauss[1], B1_scales_inhomogeneity[1]])
        self._B1_scales_step_size = np.min([B1_scales_inhomogeneity_step_size, B1_scales_gauss_step_size])
        self.B1_scales_effective_values = np.arange(
            self._B1_scales_lower_border,                           # lower border
            self._B1_scales_upper_border+self._B1_scales_step_size, # upper border, not inclusive (see math) and thus + step size
            self._B1_scales_step_size                               # the step size
        )

        # C) Create Bloch Simulation object via fixed input variables
        self.bloch_simulation_WET = LookupTableWET._BlochSimulation(flip_angles=self.flip_angles_WET_rad,
                                                                    time_gap=self.time_gaps_WET,
                                                                    flip_final_excitation=self.flip_angle_excitation_rad,
                                                                    T2=self.T2,
                                                                    TE1=self.TE,
                                                                    TR=self.TR,
                                                                    off_resonance=0)

        #D) TODO: Test:
        self.simulated_data = tools.NamedAxesArray(input_array=np.full((len(self.B1_scales_effective_values), len(self.T1_values)), -111, dtype=float),
                                                   axis_values={
                                                       "B1_scale_effective": self.B1_scales_effective_values,
                                                       "T1_over_TR": self.T1_values / self.TR
                                                   },
                                                   device="cpu")
        ###self.simulated_data = xr.DataArray(
        ###    data=np.full((len(self.B1_scales_effective_values), len(self.T1_values)), -111, dtype=float),
        ###    coords={
        ###        "B1_scale_effective": self.B1_scales_effective_values,
        ###        "T1_over_TR": self.T1_values / self.TR
        ###    },
        ###    dims=["B1_scale_effective", "T1_over_TR"]
        ###)
        #self.simulated_data = np.full((len(self.B1_scales_effective_values), len(self.T1_values)), -111, dtype=float)
        #self.row_labels = np.asarray(self.B1_scales_effective_values)  # For B1 scales (rows)
        #self.col_labels = np.asarray(self.T1_values) / self.TR  # For T1/TR (columns)


        #### D) Storage of the lookup table data. Get lookup table of b1_scales and T1/TR
        ###self.simulated_data = pd.DataFrame(-111, # just initial value for each entry, very low and maybe helpfully to detect not filled values
        ###                                   index=self.B1_scales_effective_values,
        ###                                   columns=self.T1_values / self.TR,
        ###                                   dtype=float)




    #@delayed, possible to accelerate with dask!
    def _compute_one_attenuation_value(self, T1: float | np.float64, B1_scale: float | np.float64) -> np.float64:
        """
        Just for computing the attenuation for one T1 value and B1-scale value combination.

        :return: attenuation value for just one T1 and B1-scale value.
        """
        signal_without_WET, _ = self.bloch_simulation_WET.compute_signal_after_pulses(T1=T1, B1_scale=B1_scale, with_WET=False)
        signal_with_WET, _ = self.bloch_simulation_WET.compute_signal_after_pulses(T1=T1, B1_scale=B1_scale, with_WET=True)


        with np.errstate(divide='ignore', invalid='ignore'):
            attenuation = np.divide(signal_with_WET, signal_without_WET)

        return attenuation


    def create(self):
        """
        TODO

        :return:
        """
        Console.printf(
            "info",
            f"Start creating the Lookup Table for WET (water suppression enhanced through T1 effects)"
            f"\n => Axis 1: B1 scale | Resolution: {self._B1_scales_step_size:>6.3f} | Range: {self._B1_scales_lower_border:>6.3f}:{self._B1_scales_upper_border:>6.3f}"
            f"\n => Axis 2: T1/TR    | Resolution: {self._T1_step_size / self.TR:>6.3f} | Range: {self._T1_range[0] / self.TR:>6.3f}:{self._T1_range[1] / self.TR:>6.3f}"
        )

        for T1 in tqdm(self.T1_values):
            for B1_scale in self.B1_scales_effective_values:

                # Compute the attenuation value
                value = self._compute_one_attenuation_value(T1=T1, B1_scale=B1_scale)

                if np.isnan(value):
                    Console.printf("error",
                                   "NaN value occurred while creating dictionary. Check proposed ranges. Terminating program!")
                    sys.exit()
                else:
                    # Use .loc to assign the computed value based on coordinate labels.
                    self.simulated_data.set_value(B1_scale_effective=B1_scale, # first axis
                                                  T1_over_TR=T1/self.TR,       # second axis
                                                  value=value)                 # and to axis coordinated according values
                    #self.simulated_data.loc[{"B1_scale_effective": B1_scale, "T1_over_TR": T1 / self.TR}] = value

        total_entries = self.simulated_data.size
        Console.printf("success", f"Created WET lookup table with {total_entries} entries")


    def plot(self):
        """
        To plot the created lookup table as heatmap.
        :return: Nothing
        """
        T1_over_TR_formatted = [f"{val/self.TR:.2f}" for val in self.T1_values]
        B1_scale_formatted = [f"{val:.2f}" for val in self.B1_scales_effective_values]

        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(self.simulated_data,
                         annot=False,
                         cmap='viridis',
                         robust=True,
                         xticklabels=T1_over_TR_formatted,
                         yticklabels=B1_scale_formatted)
        plt.title('Heatmap of Lookup Table')
        plt.ylabel('B1 Scale Value')
        plt.xlabel('T1/TR Value')

        ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)

        plt.tight_layout()

        plt.show()


    def _find_nearest_available_keys(self, B1_scale, T1_over_TR, interpolation_type: str = "nearest", device="cpu"):
        """
        TODO
        Get only indices available in lookup table.

        """

        if interpolation_type != "nearest":
            raise ValueError("Only 'nearest' interpolation is supported at the moment.")

        if device.lower() not in ["cpu", "cuda", "gpu"]:
            raise ValueError(f"Device must be 'cpu' or 'gpu', but got '{device}'.")

        # Select appropriate NumPy or CuPy
        xp = np if device == "cpu" else cp
        interpn = interpn_cpu if device == "cpu" else interpn_gpu

        # Convert lookup tables to array format
        B1_scales_effective_values = xp.asarray(self.B1_scales_effective_values)
        T1_values = xp.asarray(self.T1_values)
        TR = xp.asarray(self.TR)

        # Create 1D grid
        x_grid_B1_scales = xp.arange(len(B1_scales_effective_values))
        x_grid_T1 = xp.arange(len(T1_values))

        # Compute T1/TR values
        T1_over_TR_values = T1_values / TR

        # Prepare interpolation points
        xi_B1 = B1_scale.reshape(-1)  # Convert from (1000, 1) to (1000,)
        xi_T1_over_TR = T1_over_TR.reshape(-1)  # Convert from (1000, 1) to (1000,)


        row_key = interpn(points=(B1_scales_effective_values,),
                          values=x_grid_B1_scales,
                          xi=xi_B1,
                          method="nearest",
                          bounds_error=False,
                          fill_value=None)


        col_key = interpn(points=(T1_over_TR_values,),
                          values=x_grid_T1,
                          xi=xi_T1_over_TR,
                          method="nearest",
                          bounds_error=False,
                          fill_value=None)

        # Reshape to original shape
        row_key = row_key.reshape(B1_scale.shape)
        col_key = col_key.reshape(T1_over_TR.shape)

        return xp.stack([row_key, col_key], axis=0)


    class _BlochSimulation:
        """
        Simulate Water Suppression (WET) pulses and compute the resulting magnetization
        using the Bloch equations.

        This class performs simulations using NumPy and runs only on the CPU.
        It is designed to store constant simulation parameters as instance attributes.
        """

        def __init__(self, flip_angles, time_gap, flip_final_excitation, T2, TE1, TR, off_resonance):
            """
            Initialize the simulation with constant parameters.

            :param flip_angles: Sequence of flip angles (in radians) for each WET pulse.
            :param time_gap: Sequence of durations between pulses (ms).
            :param flip_final_excitation: Flip angle (in radians) for the final excitation pulse.
            :param T2: Transverse relaxation time (ms).
            :param TE1: Echo time from the last pulse to acquisition (ms).
            :param TR: Repetition time (ms).
            :param off_resonance: Off-resonance frequency (Hz).
            """
            self.flip_angles = flip_angles
            self.time_gap = time_gap
            self.flip_final_excitation = flip_final_excitation
            self.T2 = T2
            self.TE1 = TE1
            self.TR = TR
            self.off_resonance = off_resonance

        @staticmethod
        def free_precess(time_interval, t1, t2, off_resonance):
            """
            Simulate free precession and decay over a given time interval.

            :param time_interval: Time interval in milliseconds (ms).
            :param t1: Longitudinal relaxation time in ms.
            :param t2: Transverse relaxation time in ms.
            :param off_resonance: Off-resonance frequency in Hz.
            :return: Tuple (a_fp, b_fp) where:
                     a_fp (np.ndarray, 3x3): Rotation and relaxation matrix.
                     b_fp (np.ndarray, 3,): Recovery term added to the magnetization.
            """
            angle = 2.0 * np.pi * off_resonance * time_interval / 1000.0  # radians
            e1 = np.exp(-time_interval / t1)
            e2 = np.exp(-time_interval / t2)

            decay_matrix = np.array([
                [e2, 0.0, 0.0],
                [0.0, e2, 0.0],
                [0.0, 0.0, e1]
            ], dtype=float)

            z_rot_matrix = LookupTableWET._BlochSimulation.z_rot(angle)
            a_fp = decay_matrix @ z_rot_matrix
            b_fp = np.array([0.0, 0.0, 1.0 - e1], dtype=float)

            return a_fp, b_fp

        @staticmethod
        def y_rot(angle):
            """
            Generate a rotation matrix for a rotation about the y-axis.

            :param angle: Rotation angle in radians.
            :return: np.ndarray (3x3) representing the rotation matrix about the y-axis.
            """
            cos_val = np.cos(angle)
            sin_val = np.sin(angle)
            return np.array([
                [cos_val, 0.0, sin_val],
                [0.0, 1.0, 0.0],
                [-sin_val, 0.0, cos_val]
            ], dtype=float)

        @staticmethod
        def z_rot(angle):
            """
            Generate a rotation matrix for a rotation about the z-axis.

            :param angle: Rotation angle in radians.
            :return: np.ndarray (3x3) representing the rotation matrix about the z-axis.
            """
            cos_val = np.cos(angle)
            sin_val = np.sin(angle)
            return np.array([
                [cos_val, -sin_val, 0.0],
                [sin_val, cos_val, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=float)

        def compute_signal_after_pulses(self, T1: float, B1_scale: float, with_WET: bool=True) -> tuple:
            """
            Compute the signal after multiple WET pulses followed by a final excitation pulse.
            Only T1 and B1_scale are provided since the other parameters are stored as instance attributes.

            :param T1: Longitudinal relaxation time (ms) for this simulation.
            :param B1_scale: Scaling factor for the B1 field.
            :return: Tuple (magnetization_fid_last, magnetization_fid_rest) where:
                     magnetization_fid_last (float): x-component of the magnetization at the final echo.
                     magnetization_fid_rest (np.ndarray): x-components after each WET pulse.
            """

            # Check if water suppression should be applied or not. If not, assign just empty list []
            flip_angles = self.flip_angles if with_WET else []
            time_gap = self.time_gap if with_WET else []

            n_wet_pulses = len(time_gap)
            total_delay = np.sum(time_gap)

            # Spoiler matrix: destroys transverse magnetization.
            spoiler_matrix = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=float)

            # Precompute rotations and free precession for each WET pulse.
            r_flip = []
            a_exc_to_next_exc = []
            b_exc_to_next_exc = []

            for ii in range(n_wet_pulses):
                r_flip.append(LookupTableWET._BlochSimulation.y_rot(B1_scale * flip_angles[ii]))
                a_fp, b_fp = LookupTableWET._BlochSimulation.free_precess(time_gap[ii], T1, self.T2, self.off_resonance)
                a_exc_to_next_exc.append(spoiler_matrix @ a_fp)
                b_exc_to_next_exc.append(b_fp)

            # Final excitation pulse.
            r_flip_last = LookupTableWET._BlochSimulation.y_rot(B1_scale * self.flip_final_excitation)

            # Free precession from final WET pulse to acquisition.
            a_exc_last_to_acq, b_exc_last_to_acq = LookupTableWET._BlochSimulation.free_precess(self.TE1, T1, self.T2,
                                                                                                self.off_resonance)

            # Free precession from acquisition to the end of TR.
            a_tr, b_tr = LookupTableWET._BlochSimulation.free_precess(self.TR - self.TE1 - total_delay, T1, self.T2, self.off_resonance)
            a_tr = spoiler_matrix @ a_tr  # Apply spoiler after acquisition.

            # Containers for magnetization states.
            magnetizations = [None] * ((n_wet_pulses + 1) * 2 + 2)
            magnetizations_fid = [None] * (n_wet_pulses + 1)

            # Start magnetization along +z.
            magnetizations[0] = np.array([0.0, 0.0, 1.0], dtype=float)

            # Iterate multiple times to approach steady state.
            for _ in range(30):
                idx = 0
                for ii in range(n_wet_pulses):
                    magnetizations[idx + 1] = r_flip[ii] @ magnetizations[idx]
                    idx += 1

                    magnetizations_fid[ii] = magnetizations[idx]

                    magnetizations[idx + 1] = (a_exc_to_next_exc[ii] @ magnetizations[idx] +
                                               b_exc_to_next_exc[ii])
                    idx += 1

                magnetizations[idx + 1] = r_flip_last @ magnetizations[idx]
                idx += 1

                magnetizations[idx + 1] = (a_exc_last_to_acq @ magnetizations[idx] +
                                           b_exc_last_to_acq)
                idx += 1

                magnetizations_fid[n_wet_pulses] = magnetizations[idx]

                magnetizations[idx + 1] = a_tr @ magnetizations[idx] + b_tr
                idx += 1

                magnetizations[0] = magnetizations[idx]

            magnetization_fid_rest = np.array(
                [magnetizations_fid[i][0] for i in range(n_wet_pulses)],
                dtype=float
            )
            magnetization_fid_last = magnetizations_fid[n_wet_pulses][0]

            return magnetization_fid_last, magnetization_fid_rest


if __name__ == '__main__':

    import file
    configurator = file.Configurator(path_folder="/home/mschuster/projects/Synthetic_MRSI/config/",
                                     file_name="paths_25092024.json")
    configurator.load()

    loaded_B1_map = file.Maps(configurator=configurator, map_type_name="B1").load_file()
    loaded_B1_map_shape = loaded_B1_map.loaded_maps.shape


    loaded_GM_map = file.Maps(configurator=configurator, map_type_name="GM_segmentation").load_file()
    loaded_WM_map = file.Maps(configurator=configurator, map_type_name="WM_segmentation").load_file()
    loaded_CSF_map = file.Maps(configurator=configurator, map_type_name="CSF_segmentation").load_file()

    #threshold_binary = 0.7
    show_axial_slice = 70
    #loaded_GM_map.loaded_maps = (loaded_GM_map.loaded_maps > threshold_binary).astype(int)
    #loaded_WM_map.loaded_maps = (loaded_WM_map.loaded_maps > threshold_binary).astype(int)
    #loaded_CSF_map.loaded_maps = (loaded_CSF_map.loaded_maps > threshold_binary).astype(int)

    loaded_GM_map = loaded_GM_map.interpolate_to_target_size(target_size=loaded_B1_map_shape, order=0)
    loaded_WM_map = loaded_WM_map.interpolate_to_target_size(target_size=loaded_B1_map_shape, order=0)
    loaded_CSF_map = loaded_CSF_map.interpolate_to_target_size(target_size=loaded_B1_map_shape, order=0)

    plt.figure()
    fig, axs = plt.subplots(nrows=1, ncols=4)
    axs[0].set_title(f"WM map (axial slice {show_axial_slice})")
    axs[1].set_title(f"GM map (axial slice {show_axial_slice})")
    axs[2].set_title(f"CSF map (axial slice {show_axial_slice})")
    axs[3].set_title(f"WM+GM+CSF map (axial slice {show_axial_slice})")

    img0 = axs[0].imshow(loaded_WM_map.loaded_maps[:, :, show_axial_slice])
    img1 = axs[1].imshow(loaded_GM_map.loaded_maps[:, :, show_axial_slice])
    img2 = axs[2].imshow(loaded_CSF_map.loaded_maps[:, :, show_axial_slice])
    img3 = axs[3].imshow(loaded_WM_map.loaded_maps[:, :, show_axial_slice] + loaded_GM_map.loaded_maps[:, :, show_axial_slice] + loaded_CSF_map.loaded_maps[:, :, show_axial_slice])

    fig.colorbar(img0, ax=axs[0])
    fig.colorbar(img1, ax=axs[1])
    fig.colorbar(img2, ax=axs[2])
    fig.colorbar(img3, ax=axs[3])
    #plt.show()

    TR=600
    lookup_table_WET_test = LookupTableWET(T1_range=[300, 5000],
                                           T1_step_size=50,
                                           T2=250,
                                           B1_scales_inhomogeneity=[1e-10,3], # TODO
                                           B1_scales_gauss=[0.01, 1],
                                           B1_scales_inhomogeneity_step_size=0.05,
                                           B1_scales_gauss_step_size=0.05,
                                           TR=TR,
                                           TE=0, # TODO -> why 0???
                                           flip_angle_excitation_degree=47.0,
                                           flip_angles_WET_degree=[89.2, 83.4, 160.8],
                                           time_gaps_WET=[30, 30, 30])

    lookup_table_WET_test.create()
    lookup_table_WET_test.plot()
    #print(lookup_table_WET_test.get_entry(B1_scale=1.0, T1_over_TR=2.1))

    loaded_B1_map = file.Maps(configurator=configurator, map_type_name="B1")
    loaded_B1_map.load_file()

    shape_B1_map = loaded_B1_map.loaded_maps.shape
    scaled_B1_map = loaded_B1_map.loaded_maps / 39.0
    scaled_B1_map[scaled_B1_map < 0] = 0.001

    plt.figure()
    plt.imshow(scaled_B1_map[:,:,50])
    plt.colorbar()
    #plt.show()
    #sys.exit()

    Console.printf("info", f"Scaled B1 Map: min={np.min(scaled_B1_map)}, max={np.max(scaled_B1_map)}")

    #print(lookup_table_WET_test.get_entry4(B1_scale=np.array([1.0, 2.2, 1.0]), T1_over_TR=np.array([0.8, 0.8, 0.8]), device="cpu"))

    # TODO: get entry is in only get index to get entry!!!!!
    # TODO: scaled B1_map -> consider - values or to set it to 0 ????
    # TODO: Then, also load gray and white matter!!!

    # (#1) Source T1 GM & WM for 7T: https://pmc.ncbi.nlm.nih.gov/articles/PMC3375320/
    T1_GM = (1550+1804+1940+1550+1950+2132+2007+2000) / 8 #ms      (#1)
    T1_WM = (890+1043+1130+950+1200+1220+1357+1500) / 8 #ms        (#1)
    T1_CSF = 4470 #ms (average T1 for CSF was given in the paper)  (#1)

    T1_over_TR_GM_map = loaded_GM_map.loaded_maps * (T1_GM/TR)
    T1_over_TR_WM_map = loaded_WM_map.loaded_maps * (T1_WM/TR)
    T1_over_TR_CSF_map = loaded_CSF_map.loaded_maps * (T1_CSF/TR)

    print(f"GM min: {np.min(T1_over_TR_GM_map)}; max: {np.max(T1_over_TR_GM_map)}")
    print(f"WM min: {np.min(T1_over_TR_WM_map)}; max: {np.max(T1_over_TR_WM_map)}")
    print(f"CSF min: {np.min(T1_over_TR_CSF_map)}; max: {np.max(T1_over_TR_CSF_map)}")
    #sys.exit()

    #print(np.min(scaled_B1_map.flatten()), np.max(scaled_B1_map.flatten()))
    #print(np.full_like(scaled_B1_map, 1).flatten()))
    lookup_table_WET_test.simulated_data.set_interpolation_method(method="linear")

    result_GM = lookup_table_WET_test.simulated_data.get_value(B1_scale_effective=scaled_B1_map, T1_over_TR=T1_over_TR_GM_map)
    result_WM = lookup_table_WET_test.simulated_data.get_value(B1_scale_effective=scaled_B1_map, T1_over_TR=T1_over_TR_WM_map)
    result_CSF = lookup_table_WET_test.simulated_data.get_value(B1_scale_effective=scaled_B1_map, T1_over_TR=T1_over_TR_CSF_map)


    plt.figure(1)
    plt.title("GM Attenuation Map")
    plt.subplot(2, 3, 1)
    plt.imshow(result_GM[:, :, 50])
    plt.subplot(2, 3, 2)
    plt.imshow(np.rot90(result_GM[:, 90, :]))
    plt.subplot(2, 3, 3)
    plt.imshow(np.rot90(result_GM[90, :, :]))
    plt.colorbar()
    plt.subplot(2, 3, 4)
    plt.imshow(np.log2(result_GM[:, :, 50]+0.1))
    plt.subplot(2, 3, 5)
    plt.imshow(np.log2(np.rot90(np.abs(result_GM[:, 90, :]+0.1))))
    plt.subplot(2, 3, 6)
    plt.imshow(np.log2(np.rot90(result_GM[90, :, :]+0.1)))
    plt.colorbar()


    plt.figure(2)
    plt.title("WM Attenuation Map")
    plt.subplot(2, 3, 1)
    plt.imshow(result_WM[:, :, 50])
    plt.subplot(2, 3, 2)
    plt.imshow(np.rot90(result_WM[:, 90, :]))
    plt.subplot(2, 3, 3)
    plt.imshow(np.rot90(result_WM[90, :, :]))
    plt.colorbar()
    plt.subplot(2, 3, 4)
    plt.imshow(np.log2(result_WM[:, :, 50]+0.1))
    plt.subplot(2, 3, 5)
    plt.imshow(np.log2(np.rot90(np.abs(result_WM[:, 90, :]+0.1))))
    plt.subplot(2, 3, 6)
    plt.imshow(np.log2(np.rot90(result_WM[90, :, :]+0.1)))
    plt.colorbar()


    plt.figure(3)
    plt.title("CSF Attenuation Map")
    plt.subplot(2, 3, 1)
    plt.imshow(result_CSF[:, :, 50])
    plt.subplot(2, 3, 2)
    plt.imshow(np.rot90(result_CSF[:, 90, :]))
    plt.subplot(2, 3, 3)
    plt.imshow(np.rot90(result_CSF[90, :, :]))
    plt.colorbar()
    plt.subplot(2, 3, 4)
    plt.imshow(np.log2(result_CSF[:, :, 50]+0.1))
    plt.subplot(2, 3, 5)
    plt.imshow(np.log2(np.rot90(np.abs(result_CSF[:, 90, :]+0.1))))
    plt.subplot(2, 3, 6)
    plt.imshow(np.log2(np.rot90(result_CSF[90, :, :]+0.1)))
    plt.colorbar()
    plt.show()
