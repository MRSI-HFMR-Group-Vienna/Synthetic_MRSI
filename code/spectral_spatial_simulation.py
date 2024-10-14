from __future__ import annotations      #TODO: due to circular import. Maybe solve different!
from typing import TYPE_CHECKING        #TODO: due to circular import. Maybe solve different!
if TYPE_CHECKING:                       #TODO: due to circular import. Maybe solve different!
    from spatial_metabolic_distribution import MetabolicPropertyMap  #TODO: due to circular import. Maybe solve different!

from dask.diagnostics import ProgressBar
from tools import CustomArray
from dask.array import Array
from printer import Console
import dask.array as da
from tqdm import tqdm
import numpy as np
# import numba
# from numba import cuda
import tools
import cupy
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
        Greate a computational dask graph. It can be used to compute it or to add further operations.

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
        Do we need this?

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
        raise NotImplementedError("This method is not yet implemented")

    def lipid_suppression(self):
        # TODO: Perform on signal of FID? Spectrum required?
        raise NotImplementedError("This method is not yet implemented")
