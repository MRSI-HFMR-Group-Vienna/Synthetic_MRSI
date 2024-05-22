from spatial_metabolic_distribution import MetabolicPropertyMap
from dask.diagnostics import ProgressBar
from tools import CustomArray
from dask.array import Array
from printer import Console
import dask.array as da
from tqdm import tqdm
import numpy as np
#import numba
#from numba import cuda
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

    def get_name_abbreviation(self) -> list[str]:
        """
        TODO

        :return:
        """
        name_abbreviation = []
        for name in self.name:
            start_index = name.find("(")
            end_index = name.find(")")
            abbreviation = name[start_index + 1:end_index]
            name_abbreviation.append(abbreviation)

        return name_abbreviation

    def get_signal_by_name(self, compound_name:str):
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

    def __init__(self, block_size: tuple, TE: float, TR: float, alpha: float, path_cache: str = None):
        if path_cache is not None:
            if os.path.exists(path_cache):
                self.path_cache = path_cache
            else:
                Console.printf("error", f"Terminating the program. Path does not exist: {path_cache}")
                sys.exit()
            dask.config.set(temporary_directory=path_cache)

        self.block_size = block_size

        self.TE = TE
        self.TR = TR
        self.alpha = alpha

        self.fid = FID()  # instantiate an empty FID to be able to sum it ;)
        self.metabolic_property_maps: dict[str, MetabolicPropertyMap] = {}
        self.mask = None

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

    ##@staticmethod
    ##def _transform_T1(volume, alpha, TR, T1):
    ##    return volume * np.sin(np.deg2rad(alpha)) * (1 - da.exp(-TR / T1)) / (1 - (da.cos(np.deg2rad(alpha)) * da.exp(-TR / T1)))

    ##@staticmethod
    ##def _transform_T2(volume, time_vector, TE, T2):
    ##    for index, t in enumerate(time_vector):
    ##        volume[index, :, :, :] = volume[index, :, :, :] * da.exp((TE * t) / T2)
    ##    return volume

    ###@staticmethod
    ###def _transform_T1(volume, alpha, TR, T1):
    ###    return volume * np.sin(np.deg2rad(alpha)) * (1 - np.exp(-TR / T1)) / (1 - (np.cos(np.deg2rad(alpha)) * np.exp(-TR / T1)))

    ###@staticmethod
    ###def _transform_T1(volume, alpha, TR, T1):
    ###    return volume * np.sin(np.deg2rad(alpha)) * (1 - np.exp(-TR / T1)) / (1 - (np.cos(np.deg2rad(alpha)) * np.exp(-TR / T1)))

    @staticmethod
    def _transform_T1(volume, alpha, TR, T1):
        return volume * cp.sin(cp.deg2rad(alpha)) * (1 - cp.exp(-TR / T1)) / (1 - (cp.cos(np.deg2rad(alpha)) * cp.exp(-TR / T1)))

    ###@staticmethod
    ###def _transform_T2(volume, time_vector, TE, T2):
    ###    #for index, t in enumerate(time_vector):
    ###    #    volume[index, :, :, :] = volume[index, :, :, :] * da.exp((TE * t) / T2)
    ###    volume *= np.exp((TE * time_vector[:, np.newaxis, np.newaxis, np.newaxis]) / T2)
    ###    return volume

    @staticmethod
    def _transform_T2(volume, time_vector, TE, T2):
        #for index, t in enumerate(time_vector):
        #    volume[index, :, :, :] = volume[index, :, :, :] * da.exp((TE * t) / T2)
        volume *= cp.exp((TE * time_vector) / T2)
        return volume


    ###@numba.stencil
    ###@staticmethod
    ###def _transform_T2_numba(volume, time_vector, TE, T2):
    ###    return (volume[0, 0, 0, 0] * da.exp((TE * time_vector[0]) / T2))
    ###
    ###@numba.njit
    ####@staticmethod
    ###def transform_T2_numba(volume, time_vector, TE, T2):
    ###    return Model._transform_T2_numba(volume, time_vector, TE, T2)

    def assemble_graph(self) -> CustomArray:

        Console.printf("info", "Start to assemble whole graph:")
        metabolites_volume_list = []
        for fid in tqdm(self.fid, total=len(self.fid.signal)):
            # Prepare the data & reshape it
            metabolite_name = fid.name[0]
            # block_size = self.metabolic_property_maps[metabolite_name].block_size
            fid_signal = fid.signal.reshape(fid.signal.size, 1, 1, 1)
            fid_signal = cp.asarray(fid_signal) # TODO
            fid_signal = da.from_array(fid_signal, chunks=self.block_size[0])

            time_vector = fid.time[:, cp.newaxis, cp.newaxis, cp.newaxis]
            time_vector = da.from_array(time_vector, chunks=(self.block_size[0], 1, 1, 1))
            #print(time_vector)
            #input("-------------------------------------------------------")

            mask = self.mask.reshape(1, self.mask.shape[0], self.mask.shape[1], self.mask.shape[2])
            mask = cp.asarray(mask) # TODO
            mask = da.from_array(mask, chunks=(1, self.block_size[1], self.block_size[2], self.block_size[3]))
            metabolic_map_t2 = self.metabolic_property_maps[metabolite_name].t2
            metabolic_map_t1 = self.metabolic_property_maps[metabolite_name].t1

            # 87.712 sec
            # reshape fid signal from (1536,) --> (1536,1,1,1); time vector (1536,)
            # reshape mask from (112,128,80) --> (1,112,128,80)
            #
            #

            # With GPU:
            # 57.65 sec
            # --> T1 & T2 transform on GPU
            # --> concatenate not on GPU

            # With GPU:
            # 50.809 sec
            # --> not multiply mask and FID
            # --> T1 & T2 transform on GPU
            # --> concatenate ALSO on GPU

            # With GPU:
            # 42.99 sec
            # --> multiply mask and FID
            # --> T1 & T2 transform on GPU
            # --> concatenate ALSO on GPU

            # (1) FID with 3D mask volume yields 4d volume
            # TODO: ISSUE HERE!
            #fid_signal = fid_signal.map_blocks(cp.asarray, dtype='c8')
            #mask = mask.map_blocks(cp.asarray, dtype='bool')
            #volume_with_mask = fid_signal * mask
            volume_with_mask = fid_signal * mask

            #print(f"volume_with_mask shape: {volume_with_mask}")

            #print(mask.shape)
            #print(fid_signal.shape)
            #input("-ß-ß-ß-ß-ß-ß-ß-")

            # (2) Include T2 effects
            volume_metabolite = da.map_blocks(Model._transform_T2,
                                              volume_with_mask.map_blocks(cp.asarray, dtype='c8'),  # x = x.map_blocks(cupy.asarray)
                                              time_vector.map_blocks(cp.asarray, dtype='c8'),
                                              cp.asarray([self.TE]),
                                              metabolic_map_t2.map_blocks(cp.asarray, dtype='c8'),
                                              dtype='c8')

            #print(f"volume_metabolite shape (after T2): {volume_metabolite}")

            #volume_metabolite = volume_metabolite.map_blocks(np.asarray, dtype="complex64")
            ##print(type(volume_with_mask))
            ##print(type(volume_with_mask))
            ##print(type(time_vector))
            ##print(type(self.TE))
            ##input("???????????")

            ###volume_metabolite = da.map_blocks(Model._transform_T2,
            ###                                  volume_with_mask,  # x = x.map_blocks(cupy.asarray)
            ###                                  time_vector,
            ###                                  self.TE,
            ###                                  metabolic_map_t2)

            # (3) Include T1 effects
            volume_metabolite = da.map_blocks(Model._transform_T1,
                                              volume_metabolite.map_blocks(cp.asarray, dtype='c8'),
                                              cp.asarray([self.alpha]),
                                              cp.asarray([self.TR]),
                                              metabolic_map_t1.map_blocks(cp.asarray, dtype='c8'))

            #print(f"volume_metabolite shape (after T1): {volume_metabolite}")

            ##volume_metabolite = Model._transform_T2(volume_with_mask, time_vector, self.TE, metabolic_map_t2)
            ###volume_metabolite = Model._transform_T1(volume_metabolite, self.alpha, self.TR, metabolic_map_t1)


            # TODO --> cupy to numpy
            #volume_metabolite = volume_metabolite.map_blocks(cp.asnumpy)

            # (4) Include spatial concentration
            volume_metabolite *= (self.metabolic_property_maps[metabolite_name].concentration).map_blocks(cp.asarray, dtype='c8')

            #print(f"volume_metabolite (Include spatial concentration): {volume_metabolite}")

            # (5) Expand dims, since it is only for one metabolite
            volume_metabolite = da.expand_dims(volume_metabolite, axis=0)

            #print(f"volume_metabolite (expand_dims): {volume_metabolite}")

            # TODO
            volume_metabolite = volume_metabolite.map_blocks(cp.asnumpy)
            metabolites_volume_list.append(volume_metabolite)

        # (6) Put all arrays together -> 1,100_000,150,150,150 + ... + 1,100_000,150,150,150 = 11,100_000,150,150,150
        volume_all_metabolites = da.concatenate(metabolites_volume_list, axis=0)

        #print(f"volume_all_metabolites shape (concatenate): {volume_all_metabolites}")

        # (7) Sum all metabolites
        volume_sum_all_metabolites = da.sum(volume_all_metabolites, axis=0)
        volume_sum_all_metabolites = CustomArray(volume_sum_all_metabolites)

        #print(f"volume_all_metabolites shape (sum): {volume_sum_all_metabolites}")

        computational_graph: CustomArray = volume_sum_all_metabolites

        # TODO: Print summary!

        return computational_graph

        ######print(da.compute(volume_sum_all_metabolites.blocks.ravel())) # 200
        ######input("======= STOP HERE ========= in progress")
        #####import matplotlib.pyplot as plt
        #####print(volume_sum_all_metabolites)

    #####
    #####fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns
    #####
    ###### Plotting data on each subplot using imshow
    #####transversal = np.abs(volume_sum_all_metabolites[100, :, :, 40].compute())
    #####im1 = axes[0].imshow(transversal)
    #####axes[0].set_title('transversal')
    #####cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    #####
    #####coronal = np.abs(volume_sum_all_metabolites[100, :, 50, :].compute())
    #####im2 = axes[1].imshow(coronal)
    #####axes[1].set_title('coronal')
    #####cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    #####
    #####sagittal = np.abs(volume_sum_all_metabolites[100, 50, :, :].compute())
    #####im3 = axes[2].imshow(sagittal)
    #####axes[2].set_title('sagittal')
    #####cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    #####
    ###### Adding some space between subplots
    #####plt.tight_layout()
    #####
    ###### Display the plot
    #####plt.show()
    #####
    #####input("======= STOP HERE =========")
    #####
    #####
    ######volume_sum_all_metabolites.compute()

    def build(self):
        pass


# def transform_T1(volume, alpha, TR, T1):
#    return volume * np.sin(np.deg2rad(alpha)) * (1 - da.exp(-TR / T1)) / (1 - (da.cos(np.deg2rad(alpha)) * da.exp(-TR / T1)))
#
# def transform_T2(volume, time_vector, TE, T2):
#    for index, t in enumerate(time_vector):
#        volume[index, :, :, :] = volume[index, :, :, :] * da.exp((TE * t) / T2)
#    return volume


####class Model:
####    # TODO: DEPRECATED!!!!!!!!!!!!!! Remove it!!!!!!!!!!!!!
####
####    def __init__(self, path_cache: str, file_name_cache: str, partial_model: bool, partial_model_index: int = None):
####        if os.path.exists(path_cache):
####            self.path_cache = path_cache
####        else:
####            Console.printf("error", f"Terminating the program. Path does not exist: {path_cache}")
####            sys.exit()
####
####        self.file_name_cache = file_name_cache
####        self.fid = FID()  # instantiate an empty FID to be able to sum it ;)
####        self.metabolic_property_maps: dict[str, MetabolicPropertyMap] = {}
####        self.mask = None
####        self.volume: dask.array.core.Array = None  # TODO: Rename -> basically the output of the model -> is it a metabolic map?
####        self.model_shape = None  # shape of volume
####
####        self.partial_model = partial_model  # Defines if the model oly contains a sub-volume of the whole volume
####        self.partial_model_index = partial_model_index
####
####        # Necessary for a partial model
####        if self.partial_model is True:
####            if partial_model_index is None:
####                Console.printf("error",
####                               f"Since the Spectral-Spatial Model has only a partial volume and thus is a 'partial Model'"
####                               f"the index of the sub-volume in the main volume need to be set! Aborting the program!")
####                sys.exit()
####
####                # The coordinate system of the partial model. The index increases first in z, then x, then y.
####                #        origin
####                #          +----------> (+z)
####                #         /|
####                #       /  |
####                #     /    |
####                # (+x)     |
####                #          v (+y)
####
####    def add_fid(self, fid: FID) -> None:
####        """
####        Add a FID from the class `~FID`, which can contain multiple signals, to the Model. All further added FID will perform the
####        implemented __add__ in the `~FID` class. Thus, the loaded_fid will be merged. Resulting in just one fid object containing all
####        signals.
####
####        Example usage 1:
####         => Add FID of metabolites
####         => Add FID of lipids
####         => Add FID of water simulation
####         => Add FID of macromolecules simulation
####        Example usage 2:
####         => Add FID metabolite 1
####         => Add FID metabolite 2
####
####        :param fid: fid from the class `~FID`
####        :return: Nothing
####        """
####        try:
####            self.fid + fid  # sum fid according to the __add__ implementation in the FID class
####            Console.printf("success", f"Added compound '{fid.name} to the spectral spatial model.")
####        except Exception as e:
####            Console.printf("error", f"Error in adding compound '{fid.name} to the spectral spatial model. Exception: {e}")
####
####    def add_mask(self, mask: np.ndarray) -> None:
####        # TODO: Should it be an object containing further information of the Mask, transformations, etc.. -> then make class "Mask"!
####        # TODO: Just one mask per model?
####        """
####        For adding one mask to the model. It is just an numpy array with no further information.
####
####        :param mask: Numerical values of the mask as numpy array
####        :return: Nothing
####        """
####        self.mask = mask
####
####    def add_metabolic_property_map(self, metabolic_property_map: MetabolicPropertyMap):
####        """
####        Map for scaling the FID at the respective position in the volume. One map is per metabolite.
####
####        :param metabolic_property_map: Values to scale the FID at the respective position in the volume
####        :return: Nothing
####        """
####
####        Console.printf("info", f"Adding metabolic scaling map of compound {metabolic_property_map.chemical_compound_name}")
####        self.metabolic_property_maps[metabolic_property_map.chemical_compound_name] = metabolic_property_map
####
####        # self.fid_scaling_map = fid_scaling_map
####
####    def add_metabolic_property_maps(self, fid_scaling_maps: list[MetabolicPropertyMap]):
####        """
####        Multiple Maps for scaling the FID at the respective position in the volume. Each map is for one metabolite.
####
####        :param metabolic_property_map:
####        :return: Nothing
####        """
####        Console.add_lines("Adding the following metabolic scaling maps:")
####        for i, metabolic_property_map in enumerate(fid_scaling_maps):
####            Console.add_lines(f"{i}: {metabolic_property_map.chemical_compound_name}")
####            self.metabolic_property_maps[metabolic_property_map.chemical_compound_name] = metabolic_property_map
####
####        Console.printf_collected_lines("success")
####
####    def line_broader(self):
####        # TODO: Here or in Simulator?
####        raise NotImplementedError("This method is not yet implemented")
####
####    def build(self, data_type: np.dtype = np.complex64):
####        Console.printf_section("Spectral-spatial-simulation")
####        # For building the model: This includes at the current moment mask * map * FID -> (x,y,z, metabolite, FID(s))
####        # Thus, required is:
####        #   self.mask
####        #   self.FID
####        #   self.fid_scaling_map -> TODO: contains at the moment values in the range [0,1].
####
####        # (a) Set path where memmap file should be created and create memmap
####        #     Or load memmap if file already exits.
####        path_cache_file = os.path.join(self.path_cache, f"{self.file_name_cache}_spectral_spatial_volume.npy")
####        Console.printf("info", f"Using cache path: {path_cache_file}")
####
####        if (os.path.exists(path_cache_file)
####                and (Console.ask_user(f"(y) Overwrite existing file '{self.file_name_cache}'? Or (n) load old one instead", exit_if_false=False) is not True)):
####            Console.printf("info", f"loading the file '{self.file_name_cache}' instead because it already exits in the selected path!")
####            self.model_shape = np.load(f"{self.file_name_cache}_shape.npy")
####            self.volume = da.from_array(np.memmap(path_cache_file, dtype=data_type, mode='r+', shape=tuple(self.model_shape)))  # TODO TODO TODO save somewhere the shape!
####            Console.printf("success", f"Loaded volume of shape: {self.volume.shape}")
####            return
####
####        # (b) Get the required shape of the model & create cache file
####        # -> if only one FID signal is added, then add one dimension
####        # -> if multiple FID signals are added, then don't add dimensions
####        # => final dimension need to be (x,y,z, Metabolite, FID)
####        fid_shape = (1,) + self.fid.signal.shape if self.fid.signal.ndim == 1 else self.fid.signal.shape
####        self.model_shape = (self.mask.shape + fid_shape)
####        np.save(os.path.join(f"{self.file_name_cache}_shape.npy"), np.array(self.model_shape))
####
####        # (c) Create memmap on disk with zeros. Then, convert it to as dask array.
####        self.volume = da.from_array(np.memmap(path_cache_file, dtype=data_type, mode='w+', shape=self.model_shape))
####        # volume[:] = 0
####
####        # (c) Estimate space required on hard drive
####        #  -> self.model_shape.size returns the shape and np.prod the number of elements in the planned memmap array.
####        #  -> the np.dtype(data_type) creates and object and .itemsize returns the size (in bytes) of each element
####        space_required_mb = tools.SpaceEstimator.for_numpy(data_shape=self.model_shape, data_type=data_type, unit="MB")
####
####        # (d) Ask the user for agreement of the required space
####        # Check if the simulation does not exceed a certain size
####        # Console.check_condition(space_required_mb < 30000, ask_continue=True)
####        Console.ask_user(f"Estimated required space [MB]: {space_required_mb}")
####
####        # (e) Find non zero elements in the mask
####        mask_nonzero_indices = np.nonzero(self.mask)
####
####        # (d) Check if each FID has a respective scaling map!
####        compounds_FIDs = self.fid.name
####        compounds_maps = list(self.metabolic_property_maps.keys())
####        if not len(compounds_FIDs) == len(compounds_maps):
####            Console.add_lines("Different amount of FIDs and scaling maps.")
####            Console.add_lines(f"Compounds FIDs given: {compounds_FIDs}")
####            Console.add_lines(f"Compounds Scaling Maps given: {compounds_maps}")
####            Console.add_lines(f"Terminate program!")
####            Console.printf_collected_lines("error")
####            sys.exit()
####        if not compounds_FIDs <= compounds_maps:
####            Console.add_lines("For not each compound a FID and a respective scaling map is given.")
####            Console.add_lines(f"Compounds FIDs given: {compounds_FIDs}")
####            Console.add_lines(f"Compounds Scaling Maps given: {compounds_maps}")
####            Console.add_lines(f"Terminate program!")
####            Console.printf_collected_lines("error")
####            sys.exit()
####
####        Console.printf("info", "Begin assigning FID to volume:")
####        indent = " " * 22
####        for x, y, z in tqdm(zip(*mask_nonzero_indices), total=len(mask_nonzero_indices[0]), desc=indent + "Assigning FID to volume"):
####            for fid_signal_index in range(self.fid.signal.ndim):
####                # Scale the FID of each chemical compound based on the given scaling map for this compound
####                # self.volume[x, y, z, fid_signal_index, :] = self.fid.signal[fid_signal_index] * self.metabolic_property_maps[self.fid.name[fid_signal_index]][x, y, z]
####
####                # Multiply the FID at the respective position with the maps of the metabolic property map
####                chemical_compound_fid_name = self.fid.name[fid_signal_index]
####                self.volume[x, y, z, fid_signal_index, :] = \
####                    (
####                            self.fid.signal[fid_signal_index] *
####                            self.metabolic_property_maps[chemical_compound_fid_name].concentration[x, y, z] *
####                            self.metabolic_property_maps[chemical_compound_fid_name].t2 *
####                            self.metabolic_property_maps[chemical_compound_fid_name].t1
####                    )
####
####        Console.printf("success", f"Created volume of shape: {self.volume.shape}")
####        self.volume.flush()  # Write any changes in the array to the file on disk.
####        Console.printf("success", f"Flushed updated volume to path: {self.path_cache}. Actual used space [MB]: {os.path.getsize(path_cache_file) / (1024 * 1024)}")
####
####    def _sum_all_fids_in_volume(self):
####        # TODO: maybe implement in build or call extra?! Do we need volume where FIDs not summed up (Guess not?)? And flush before or after to disk?
####        raise NotImplementedError("This method is not yet implemented")


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
