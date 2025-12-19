from __future__ import annotations      #TODO: due to circular import. Maybe solve different!
from typing import TYPE_CHECKING        #TODO: due to circular import. Maybe solve different!

if TYPE_CHECKING:                       #TODO: due to circular import. Maybe solve different!
    from spatial_metabolic_distribution import MetabolicPropertyMap  #TODO: due to circular import. Maybe solve different!

from cupyx.scipy.interpolate import interpn as interpn_gpu
from scipy.interpolate import interpn as interpn_cpu
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
from tools import CustomArray
from printer import Console
from pathlib import Path
import dask.array as da
import seaborn as sns
from tqdm import tqdm
import numpy as np
# import numba
# from numba import cuda
import tools
import cupy as cp
import dask
import sys
import os


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
                 sampling_period: float = None,
                 sampling_period_unit = None):
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
        TODO Describe!

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
        To get one signal from the whole FID by name.

        :param compound_name: Depending on the available names. An example coule be "NAcetylAspartate (NAA)".
        :return: A FID object containing only the desired signal.
        """
        try:
            index = self.name.index(compound_name)
            signal = self.signal[index, :]
            return FID(signal=signal, name=[compound_name], time=self.time)
        except ValueError:
            Console.printf("error", f"Chemical compound not available: {compound_name}. Please check your spelling and possible whitespaces!")
            return

    def get_signal(self, signal_data_type: np.dtype=np.complex64, mute=True):
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

    def get_spectrum(self, x_type="ppm", *, reference_frequency=297_223_042, ppm_center=4.65) -> dict[str, np.ndarray]: # 2.65
        r"""
        To get the spectrum of all signals contained in the FID (e.g. of NAA, Glutamate, and so on). Also, a single signal
        is possible, but then return is of shape (1, signal)!

        When ppm is computed:

        .. math::
            \text{ppm} = \text{ppm}_{\text{center}} - \frac{f}{f_{ref}}\times 10^{6} ~~~~
        .. math:: ~
        .. math::
            \text{with:}~f=\text{frequency [Hz]}, f_{ref}=\text{reference frequency [Hz]}, \text{and}~10^{6}~\text{to get [MHz]}
        .. math::
            \text{and:}~\text{ppm}_{\text{center}}~\text{specifies which chemical-shift value (in ppm) corresponds to a frequency offset of 0 Hz}

        It returns a dictionary with dict['x'] the type in 'ppm', 'frequency' (Hz), 'time' (sec) and dict ['y'] of the magnitude.
        """
        # Possible scales on x-axis
        return_types_possible = ["ppm", "frequency"]
        if x_type not in return_types_possible:
            Console.printf("error", f"Return types of FID only possible: {return_types_possible}. But you gave: {x_type}. Terminate program!")
            sys.exit()

        # A) Compute the frequency vector
        N = self.time.size
        dwell_time = self.time[1] - self.time[0]
        frequency_hz = np.fft.fftfreq(N, d=dwell_time)
        frequency_hz = np.fft.fftshift(frequency_hz)
        x = frequency_hz

        # Ensure that shape is (1, y) to enable handling multidimensional arrays
        if self.signal.ndim == 1:
            signal = self.signal[np.newaxis, :]
        else:
            signal = self.signal
        spectrum = np.fft.fft(signal, axis=1)
        spectrum = np.fft.fftshift(spectrum, axes=1)

        # B) Convert the frequency vector a ppm scale (spectroscopic):
        if x_type == "ppm":
            #ppm = ppm_center - (frequency / reference_frequency) * 1e6
            ppm = (frequency_hz / reference_frequency) * 1e6 + ppm_center
            x = ppm

        return {'x': x, 'y': spectrum}


    def plot(self, x_type="ppm", y_type="magnitude", plot_offset=1_500, show=True, save_path: str=None, *, reference_frequency=297_223_042, ppm_center=4.65, additional_description:str="", legend_position='upper left', figsize=(15,8)) -> None:
        """
        To plot all signals contained in the current FID object. It also supports of the FID object only hold currently one signal.

        The plot supports to plot:
            - The signals in the time domain (x_type='time')
            - The signals in the frequency domain in Hz (x_type='frequency')
            - The signals in the frequency domain in ppm (x_type='ppm')

        Further, it is possible to save this plot and/or show it.

        If ppm is chosen for x_type then spectroscopic axis form [X to 0], thus values are descending.

        **(!) Please note:** If the magnitude is of the individual signals is low and the **plot_offset** relatively high, then the individual signals appear flat in the plot.
        **(!) The default plot_offset is 2000**!

        :param x_type: Defines the domain. Can be "time", "frequency", "ppm".
        :param plot_offset: Offset in vertical axis (y-axis) for preventing signal overlapping.
        :param show: If plot is shown directly after creating it.
        :param save_path: Path including filename to save plot.
        :param reference_frequency: Important when using ppm, can be e.g., those of tetramethylsilane (TMS), or water, etc...
        :param ppm_center: Defines which chemical shift value corresponds to a frequency offset of 0 Hz
        :param additional_description: Additional string. Please make "\\n" for new lines!
        :param legend_position: See matplotlib.
        :return: Nothing
        """

        # 0) Define possible quantities for x and y axis
        x_type_possible = ["time", "frequency", "ppm"]
        y_type_possible = {
            "magnitude": np.abs,
            "real": np.real,
            "imag": np.imag,
            "phase_rad": partial(np.angle, deg=False),
            "phase_deg": partial(np.angle, deg=True),
        }

        # 1) Check if chosen type for x-axis is valid
        if not x_type in x_type_possible:
            Console.printf("error", f"Only possible to choose '{x_type_possible}' for x-axis, but you have chosen '{x_type}'. Terminate program!")
            sys.exit()

        # 2) Then, cases possible: plotting either in time domain or the frequency domain (frequency=>Hz or ppm)
        if x_type == "time":
            scales = self.time
            signal = self.signal[np.newaxis, :] if self.signal.ndim == 1 else self.signal # to ensure signal has dim (1, X)
        else:
            spectrum_dict = self.get_spectrum(x_type=x_type, reference_frequency=reference_frequency, ppm_center=ppm_center)
            scales = spectrum_dict["x"]
            signal = spectrum_dict["y"]

        # 3a) Left subplot: To create the figure and plot all signals
        fig, (ax_main, ax_sidebar) = plt.subplots(figsize=figsize,
                                                  nrows=1,
                                                  ncols=2,
                                                  gridspec_kw={'width_ratios': [5, 1]})

        #  -> Convert complex signal to possible quantity and print error if quantity not supported
        if not y_type in y_type_possible:
            Console.printf("error",f"Only possible to choose '{y_type_possible}' for y-axis, but you have chosen '{y_type}'. Terminate program!")
            sys.exit()
        else:
            transform = y_type_possible[y_type]

        #  -> Plot all signals
        for i, (name, scale, signal)  in enumerate(zip(self.name, scales, signal)):
            ax_main.plot(scales, transform(signal)+(i*plot_offset), label=name, linewidth=0.8)
        ax_main.set_title('Absolute Values of the FID')
        ax_main.set_xlabel(f"{x_type}")

        # 3b) Right subplot: To create the description
        ax_sidebar.axis('off')
        ax_sidebar.set_title('Description')  # keep the box/frame
        text_description = (f"{'Plot offset':.<17}: {plot_offset}\n"
                            f"{'Datetime':.<17}: {datetime.now().replace(microsecond=0)}\n\n"
                            f"{'Additional info':.<17}: {'Nothing' if additional_description == '' else additional_description}")

        # 3c) If ppm then add additional information
        if x_type == "ppm":
            ppm_string = (f"{'Ref. frequency':.<17}: {reference_frequency}\n"
                          f"{'ppm center':.<17}: {ppm_center if x_type == 'ppm' else None}\n")
            text_description = ppm_string + text_description

        # 3d) Just to ensure this is first in text string
        text_description = f"{'Show y-values':.<17}: {y_type}\n" + text_description

        # 4) Add text to the right (description) text subplot
        ax_sidebar.text(
            0.01, 0.99, text_description,
            transform=ax_sidebar.transAxes,
            va='top',
            ha='left',
            family='monospace'
        )

        # 5a) Just required in case ppm occurs, thus x-axis is mirrored (spectroscopic view)
        handles, labels = ax_main.get_legend_handles_labels()
        handles = handles[::-1]
        labels = labels[::-1]

        # 5b) Apply mirroring of axis in case ppm is chosen
        if x_type == "ppm":
            ax_main.invert_xaxis()

        # 5c) Create the whole figure
        ax_main.legend(handles, labels, loc=legend_position, fontsize=9, frameon=True, markerfirst=False, ncol=1)
        plt.tight_layout()


        # 6) Save to desired path and/or show
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()


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
        """
        Simpy summary of the whole model.

        :return: Nothing
        """
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
        Console.add_lines(f" Cache path  ... {Path(*Path(self.path_cache).parts[-2:])} (shortened)")
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
            self.fid = self.fid + fid  # sum fid according to the __add__ implementation in the FID class
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


###    @staticmethod
###    def _transform_T1_T2(xp: np|cp, volume: da.Array, time_vector, TE, T2, alpha, TR, T1):
###        # TODO: DOCSTRING!
###
###        volume = Model._transform_T1(xp, volume, alpha, TR, T1)
###        volume = Model._transform_T2(xp, volume, time_vector, TE, T2)
###
###
###        return volume

    @staticmethod
    def _transform_T1(xp: np|cp, volume: da.Array, alpha, TR, T1):
        r"""

        Applying a T1 map (ideally masked) to a volume of fid signals. Thus, transforming via the formular below the
        input volume V_{in} --> V_{out}:
        .. math::

            V_{out} = V_{in} \, \sin(\alpha) \, \frac{1 - e^{-TR/T_1}}{1 - \cos(\alpha) \, e^{-TR/T_1}}


        For the transformation a flip angle, TR and T1 is used.
           * alpha ... scalar value (either numpy or cupy)
           * TR    ... scalar value (either numpy or cupy)
           * T1    ... matrix       (either numpy or cupy)
           * xp    ... either numpy or cupy -> using the whole imported library (np or cp is xp)

        (!) It needs to be a static function, otherwise it seems dask cannot handle it properly with map_blocks.
        """
        decay = xp.sin(xp.radians(alpha)) * (1 - xp.exp(-TR / T1)) / (1 - (xp.cos(xp.radians(alpha)) * xp.exp(-TR / T1)))
        volume = xp.where(volume != 0, volume * decay, volume) # only compute where volume is not zero!
        return volume

    @staticmethod
    def _transform_T2(xp: np|cp, volume, time_vector, TE, T2):
        r"""
        Applying a T2 to the volume of fid signals. Thus, transforming the input volume V_{in} --> V_{out}:
        .. math::

            V_{out} = V_{in} \, e^{-\frac{TE + t}{T_2}}


        For the transformation a time vector, TE and T2 is used.
           * time vector ... vector (either numpy or cupy)
           * TE          ... scalar value (either numpy or cupy)
           * T2          ... matrix       (either numpy or cupy)
           * xp          ... either numpy or cupy -> using the whole imported library (np or cp is xp)

        It needs to be a static function, otherwise it seems dask cannot handle it properly with map_blocks.
        """
        decay = xp.where(T2 != 0, xp.exp(-(TE + time_vector) / T2), 1)
        volume = xp.where(volume != 0, volume * decay, volume)
        return volume


    def assemble_graph(self, all_steps_output=False) -> CustomArray | tuple[CustomArray,CustomArray,CustomArray,CustomArray]:
        """
        Create a computational dask graph. It can be used to compute it or to add further operations.
        Note: all_steps_output=True, it outputs graphs that includes only certain steps:
                    -> volume after applying the mask
                    -> volume after applying the T1
                    -> volume after applying the T2
                    -> volume after applying the concentration scaling

                    (!) Each of these steps builds on the previous one and is therefore not independent
                        of the preceding ones!

        :return: CustomArray with numpy or cupy, based on the selected device when created the model. Note: in case of
                 all_steps_output=True, a tuple of computational graphs of the respective applied steps is returned.
        """

        Console.printf("info", f"Start to assemble whole graph on device {self.compute_on_device}:")

        # Defining based on the selected device if cupy or numpy should be used
        xp = None
        if self.compute_on_device == "cuda":
            xp = cp  # use cupy
        elif self.compute_on_device == "cpu":
            xp = np  # use numpy
        else:
            Console.printf("error", f"Compute on device can be either 'cpu' or 'cuda' but you set {self.compute_on_device}. Terminate program!")
            sys.exit()

        metabolites_volume_list = []

        if all_steps_output:
            metabolites_volume_list__after_mask = []
            metabolites_volume_list__after_T1 = []
            metabolites_volume_list__after_T2 = []
            #metabolites_volume_list__after_concentration_scaling = [] not necessary since the final output!

        for fid in tqdm(self.fid, total=len(self.fid.signal)):
            # (1) Prepare the data & reshape it ########################################################################
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


            #################### FID + MASK = Volume ###################################################################
            # (2) FID with 3D mask volume yields 4d volume
            volume_with_mask = fid_signal * mask

            if all_steps_output:
                metabolites_volume_list__after_mask.append(volume_with_mask[None, ...])

            #################### APPLY T1 and T2 #######################################################################
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

            # (4) Include T1 and T2 effects
            #   4a) Include T1 effects
            volume_metabolite = da.map_blocks(Model._transform_T1,
                                              xp,  # xp is either numpy (np) or cupy (cp) based on the selected device
                                              volume_with_mask,
                                              alpha,
                                              TR,
                                              metabolic_map_t1,
                                              dtype=dtype)

            if all_steps_output:
                metabolites_volume_list__after_T1.append(volume_metabolite[None, ...])

            #   4b) Include T2 effects
            volume_metabolite = da.map_blocks(Model._transform_T2,
                                              xp,  # xp is either numpy (np) or cupy (cp) based on the selected device
                                              volume_metabolite,
                                              time_vector,
                                              TE,
                                              metabolic_map_t2,
                                              dtype=dtype)

            if all_steps_output:
                metabolites_volume_list__after_T2.append(volume_metabolite[None, ...])

            #################### SCALE WITH SPATIAL CONCENTRATION ######################################################
            # (5) Include spatial concentration
            if self.compute_on_device == "cuda":
                volume_metabolite *= (self.metabolic_property_maps[metabolite_name].concentration).map_blocks(cp.asarray, dtype='c8')
            elif self.compute_on_device == "cpu":
                volume_metabolite *= self.metabolic_property_maps[metabolite_name].concentration

            # (6) Expand dims, since it is only for one metabolite and concatenate
            #volume_metabolite = da.expand_dims(volume_metabolite, axis=0)
            volume_metabolite = volume_metabolite[None, ...] # using this version since volume_metabolite = da.expand_dims(volume_metabolite, axis=0) throws large chunks warning

            metabolites_volume_list.append(volume_metabolite)

        # (7) Put all arrays together -> 1,100_000,150,150,150 + ... + 1,100_000,150,150,150 = 11,100_000,150,150,150
        volume_all_metabolites = da.concatenate(metabolites_volume_list, axis=0)

        if all_steps_output:
            volume_all_metabolites__after_mask = da.concatenate(metabolites_volume_list__after_mask, axis=0)
            volume_all_metabolites__after_T1 = da.concatenate(metabolites_volume_list__after_T1, axis=0)
            volume_all_metabolites__after_T2 = da.concatenate(metabolites_volume_list__after_T2, axis=0)

        # (8) Sum all metabolites & Create CustomArray from dask Array
        volume_sum_all_metabolites = da.sum(volume_all_metabolites, axis=0)
        volume_sum_all_metabolites = CustomArray(volume_sum_all_metabolites)

        if all_steps_output:
            volume_sum_all_metabolites__after_mask = da.sum(volume_all_metabolites__after_mask, axis=0)
            volume_sum_all_metabolites__after_T1   = da.sum(volume_all_metabolites__after_T1, axis=0)
            volume_sum_all_metabolites__after_T2   = da.sum(volume_all_metabolites__after_T2, axis=0)

        # (9) If the computation device and target device does not match then adjust it
        if self.compute_on_device == "cuda" and self.return_on_device == "cpu":
            volume_sum_all_metabolites = volume_sum_all_metabolites.map_blocks(cp.asnumpy)
            if all_steps_output:
                volume_sum_all_metabolites__after_mask = volume_sum_all_metabolites__after_mask.map_blocks(cp.asnumpy)
                volume_sum_all_metabolites__after_T1   = volume_sum_all_metabolites__after_T1.map_blocks(cp.asnumpy)
                volume_sum_all_metabolites__after_T2   = volume_sum_all_metabolites__after_T2.map_blocks(cp.asnumpy)

        elif self.compute_on_device == "cpu" and self.return_on_device == "cuda":
            volume_sum_all_metabolites = volume_sum_all_metabolites.map_blocks(cp.asarray)
            if all_steps_output:
                volume_sum_all_metabolites__after_mask = volume_sum_all_metabolites__after_mask.map_blocks(cp.asarray)
                volume_sum_all_metabolites__after_T1   = volume_sum_all_metabolites__after_T1.map_blocks(cp.asarray)
                volume_sum_all_metabolites__after_T2   = volume_sum_all_metabolites__after_T2.map_blocks(cp.asarray)

        if not all_steps_output:
            # (10) Return the computional graph
            computational_graph: CustomArray = volume_sum_all_metabolites
            return computational_graph
        else:
            computational_graph__after_mask: CustomArray = volume_sum_all_metabolites__after_mask
            computational_graph__after_T1: CustomArray = volume_sum_all_metabolites__after_T1
            computational_graph__after_T2: CustomArray = volume_sum_all_metabolites__after_T2
            computational_graph__after_concentration_scaling: CustomArray = volume_sum_all_metabolites
            return (computational_graph__after_mask, computational_graph__after_T1, computational_graph__after_T2, computational_graph__after_concentration_scaling)


###    def assemble_graph(self) -> CustomArray:
###        """
###        Create a computational dask graph. It can be used to compute it or to add further operations.
###
###        :return: CustomArray with numpy or cupy, based on the selected device when created the model.
###        """
###
###        Console.printf("info", f"Start to assemble whole graph on device {self.compute_on_device}:")
###
###        # Defining based on the selected device if cupy or numpy should be used
###        xp = None
###        if self.compute_on_device == "cuda":
###            xp = cp  # use cupy
###        elif self.compute_on_device == "cpu":
###            xp = np  # use numpy
###        else:
###            Console.printf("error", f"Compute on device can be either 'cpu' or 'cuda' but you set {self.compute_on_device}. Terminate program!")
###            sys.exit()
###
###        metabolites_volume_list = []
###        for fid in tqdm(self.fid, total=len(self.fid.signal)):
###            # (1) Prepare the data & reshape it ########################################################################
###            #   1a) Get the name of the metabolite (unpack from the list)
###            metabolite_name = fid.name[0]
###            #   1b) Reshape FID for multiplication, put to a respective device and create dask array with chuck size defined by the user
###            fid_signal = fid.signal.reshape(fid.signal.size, 1, 1, 1)
###            fid_signal = xp.asarray(fid_signal)
###            fid_signal = da.from_array(fid_signal, chunks=self.block_size[0])
###            #   1c) Reshape time vector, create dask array with block size defined by the user
###            time_vector = fid.time[:, xp.newaxis, xp.newaxis, xp.newaxis]
###            time_vector = da.from_array(time_vector, chunks=(self.block_size[0], 1, 1, 1))
###            #   1d) Same as for FID and time vector above
###            mask = self.mask.reshape(1, self.mask.shape[0], self.mask.shape[1], self.mask.shape[2])
###            mask = xp.asarray(mask)
###            mask = da.from_array(mask, chunks=(1, self.block_size[1], self.block_size[2], self.block_size[3]))
###            #   1f) Get T1 and T2 of respective metabolite with the metabolite name
###            metabolic_map_t2 = self.metabolic_property_maps[metabolite_name].t2
###            metabolic_map_t1 = self.metabolic_property_maps[metabolite_name].t1
###
###            # (2) FID with 3D mask volume yields 4d volume #############################################################
###            volume_with_mask = fid_signal * mask
###
###            # (3) Based on selected device prepare data ################################################################
###            dtype = eval("xp." + self.data_type) # yields in e.g., xp.complex64 which can be np.complex64 or cp.complex64
###            if self.compute_on_device == "cpu":
###                # Creates numpy arrays with xp
###                TE = xp.asarray([self.TE])
###                alpha = xp.asarray([self.alpha])
###                TR = xp.asarray([self.TR])
###                #dtype = xp.complex64  # TODO
###            elif self.compute_on_device == "cuda":
###                # Creates cupy arrays with xp
###                volume_with_mask = volume_with_mask.map_blocks(xp.asarray, dtype=xp.complex64)
###                time_vector = time_vector.map_blocks(xp.asarray, dtype=xp.complex64)
###                TE = xp.asarray([self.TE])
###                metabolic_map_t2 = metabolic_map_t2.map_blocks(xp.asarray, dtype=xp.complex64)
###                metabolic_map_t1 = metabolic_map_t1.map_blocks(xp.asarray, dtype=xp.complex64)
###                alpha = xp.asarray([self.alpha])
###                TR = xp.asarray([self.TR])
###                #dtype = xp.complex64
###
###            # (4) Include T1 and T2 effects
###            #   4a) Include T1 effects
###            volume_metabolite = da.map_blocks(Model._transform_T1,
###                                              xp,  # xp is either numpy (np) or cupy (cp) based on the selected device
###                                              volume_with_mask,
###                                              alpha,
###                                              TR,
###                                              metabolic_map_t1,
###                                             dtype=dtype)
###
###            #   4b) Include T2 effects
###            volume_metabolite = da.map_blocks(Model._transform_T2,
###                                              xp,  # xp is either numpy (np) or cupy (cp) based on the selected device
###                                              volume_metabolite,
###                                              time_vector,
###                                              TE,
###                                              metabolic_map_t2,
###                                              dtype=dtype)
###
###            # (5) Include spatial concentration
###            if self.compute_on_device == "cuda":
###                volume_metabolite *= (self.metabolic_property_maps[metabolite_name].concentration).map_blocks(cp.asarray, dtype='c8')
###            elif self.compute_on_device == "cpu":
###                volume_metabolite *= self.metabolic_property_maps[metabolite_name].concentration
###
###            # (6) Expand dims, since it is only for one metabolite and concatenate
###            #volume_metabolite = da.expand_dims(volume_metabolite, axis=0)
###            volume_metabolite = volume_metabolite[None, ...] # using this version since volume_metabolite = da.expand_dims(volume_metabolite, axis=0) throws large chunks warning
###
###            metabolites_volume_list.append(volume_metabolite)
###
###        # (7) Put all arrays together -> 1,100_000,150,150,150 + ... + 1,100_000,150,150,150 = 11,100_000,150,150,150
###        volume_all_metabolites = da.concatenate(metabolites_volume_list, axis=0)
###
###        # (8) Sum all metabolites & Create CustomArray from dask Array
###        volume_sum_all_metabolites = da.sum(volume_all_metabolites, axis=0)
###        volume_sum_all_metabolites = CustomArray(volume_sum_all_metabolites)
###
###        # (9) If the computation device and target device does not match then adjust it
###        if self.compute_on_device == "cuda" and self.return_on_device == "cpu":
###            volume_sum_all_metabolites = volume_sum_all_metabolites.map_blocks(cp.asnumpy)
###        elif self.compute_on_device == "cpu" and self.return_on_device == "cuda":
###            volume_sum_all_metabolites = volume_sum_all_metabolites.map_blocks(cp.asarray)
###
###        # (10) Just clarify a computational graph is returned
###        computational_graph: CustomArray = volume_sum_all_metabolites
###
###        return computational_graph
###
####    def build(self):
####        """
####        Do we need this? I guess not! Would give whole computed model, which would be very big!
####
####
####        :return:
####        """
####        raise NotImplementedError


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
                 B1_scales_inhomogeneity: np.ndarray | list = np.ndarray([0, 2]),       # e.g., [0, 2]
                 B1_scales_gauss: np.ndarray | list = np.array([0.01, 1]),              # e.g., [0.01, 1]
                 B1_scales_inhomogeneity_step_size: float = 0.05,                       # e.g., 0.05
                 B1_scales_gauss_step_size: float = 0.05,                               # e.g., 0.05
                 TR: float = 600.0,
                 TE: float = 0.0,
                 flip_angle_excitation_degree: float = 47.0,
                 flip_angles_WET_degree: np.ndarray | list = np.array([89.2, 83.4, 160.8]),
                 time_gaps_WET: np.ndarray | list = np.ndarray([30, 30, 30]),
                 off_resonance: float = 0):

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
                                                                    time_gaps=self.time_gaps_WET,
                                                                    flip_final_excitation=self.flip_angle_excitation_rad,
                                                                    T2=self.T2,
                                                                    TE1=self.TE,
                                                                    TR=self.TR,
                                                                    off_resonance=off_resonance)

        #D) TODO: Test:
        self.simulated_data = tools.NamedAxesArray(input_array=np.full((len(self.B1_scales_effective_values), len(self.T1_values)), -111, dtype=float),
                                                   axis_values={
                                                       "B1_scale_effective": self.B1_scales_effective_values,
                                                       "T1_over_TR": self.T1_values / self.TR
                                                   },
                                                   device="cpu")


        # TODO: Delete after test usage!
        #self.negative_with_WET = 0
        #self.negative_without_WET = 0
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
            attenuation = np.divide(np.abs(signal_with_WET), np.abs(signal_without_WET))

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
                    self.simulated_data.set_value(B1_scale_effective=B1_scale, # first axis
                                                  T1_over_TR=T1/self.TR,       # second axis
                                                  value=value)                 # and to axis coordinated according values
                    #self.simulated_data.loc[{"B1_scale_effective": B1_scale, "T1_over_TR": T1 / self.TR}] = value

        total_entries = self.simulated_data.size
        Console.printf("success", f"Created WET lookup table with {total_entries} entries. Values Range: [{np.min(self.simulated_data)}, {np.max(self.simulated_data)}]")


    class _BlochSimulation:
        """
        Simulate Water Suppression (WET) pulses and compute the resulting magnetization
        using the Bloch equations.

        This class performs simulations using NumPy and runs only on the CPU.
        It is designed to store constant simulation parameters as instance attributes.
        """

        def __init__(self, flip_angles, time_gaps, flip_final_excitation, T2, TE1, TR, off_resonance):
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
            self.time_gaps = time_gaps
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

            # angle = Δϕ=ωΔt with ω=2πf (and also off resonance); divided by 1000 to get from [ms] ->[s]
            angle = 2.0 * np.pi * off_resonance * time_interval / 1000.0  # radians
            e1 = np.exp(-time_interval / t1) # Mz(t) = M0 * (1-e^(-t/T1)    # TODO
            e2 = np.exp(-time_interval / t2) # Mxy(t) = Mxy(0) * e^(-t/T2)  # TODO

            decay_matrix = np.array([
                [e2, 0.0, 0.0],                   # | Mx |          | e2   0    0 |   | Mx |
                [0.0, e2, 0.0],                   # | My |      =   | 0    e2   0 | * | My |
                [0.0, 0.0, e1]                    # | Mz |end       | 0    0   e1 |   | Mz |start
            ], dtype=float)


            z_rot_matrix = LookupTableWET._BlochSimulation.z_rot(angle)
            a_fp = decay_matrix @ z_rot_matrix
            # (!) TODO: Because on resonance, nothing happens
            # if system is "on-resonance" (thus if off-resonance == 0), and since the z_rot (rotation around z-axis):
            #
            #                         |cos(Δϕ), -sin(Δϕ),  0|,     when on resonance yields    | 1  0  0 |
            # z_rot_matrix = Rz(Δϕ) = |sin(Δϕ),  cos(Δϕ),  0|,     ----------------------->    | 0  1  0 |
            #                         |0,        0,        1|                                  | 0  0  1 |
            #
            # ... it yields identity matrix (since cos(0) == 1 and sin(0) == 0). Thus, since decay matrix @ z rot matrix => yields only decay matrix for a_fp

            b_fp = np.array([0.0, 0.0, 1.0 - e1], dtype=float)
            # Mz(Δt) = Mz(0) * e^(−Δt/T1) + M0 * (1−e−Δt/T1) ... with M0 = 1 # TODO

            # a_fp => exponential decay + rotation of the magnetisation vector
            #          => transverse part   (xy via e2, effect of T2 decay)
            #          => longitudinal part (z via e1, effect of T1 decay)
            #
            # b_fp => for the T1 recovery offset, meaning for pulling the Mz back to its equilibrium. Since main magnetic B0,
            #         tge equilibrium is in z-plane (M0 = 1).
            #
            #                | e2 * cos(Δϕ)  -e2 * sin(Δϕ)    0 |             | 0    |
            #         a_fp = | e2 * sin(Δϕ)   e2 * cos(Δϕ)    0 |   , b_fp =  | 0    |
            #                | 0              0              e1 |             | 1-e1 |
            #
            # Finally, it can yield: M_new = a_fp * M_old + b_fp or same: M(t+Δt) = a_fp * M(t) + b_fp.

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
            time_gaps = self.time_gaps if with_WET else []

            #print(f"time_gaps: {time_gaps}")

            n_wet_pulses = len(time_gaps)
            total_delay = np.sum(time_gaps)

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

            # (!) Only enter this part if WET pulses should be applied, thus if with_WET == True. Else n_wet_pulses == [], thus range(len([])) == 0
            for ii in range(n_wet_pulses):

                # *) Calculate the flip angle (rotation matrix around y-axis) for each B1 scale * flip angle
                #    Here the flip angle gets scaled based on the actual RF-Field conditions
                r_flip.append(LookupTableWET._BlochSimulation.y_rot(B1_scale * flip_angles[ii])) # TODO: Remove abs

                #if np.any(LookupTableWET._BlochSimulation.y_rot(B1_scale * flip_angles[ii]) < 0):
                #    print("LookupTableWET._BlochSimulation.y_rot(B1_scale * flip_angles[ii]) < 0")
                #    print(LookupTableWET._BlochSimulation.y_rot(B1_scale * flip_angles[ii]))
                #    print(np.abs(LookupTableWET._BlochSimulation.y_rot(B1_scale * flip_angles[ii])))
                #    sys.exit()

                a_fp, b_fp = LookupTableWET._BlochSimulation.free_precess(time_gaps[ii], T1, self.T2, self.off_resonance)
                a_exc_to_next_exc.append(spoiler_matrix @ a_fp)
                b_exc_to_next_exc.append(b_fp)

            # Final excitation pulse.
            r_flip_last = LookupTableWET._BlochSimulation.y_rot(B1_scale * self.flip_final_excitation)

            # Free precession from final WET pulse to acquisition.
            a_exc_last_to_acq, b_exc_last_to_acq = LookupTableWET._BlochSimulation.free_precess(self.TE1, T1, self.T2, self.off_resonance)

            # Free precession from acquisition to the end of TR.
            a_tr, b_tr = LookupTableWET._BlochSimulation.free_precess(self.TR - self.TE1 - total_delay, T1, self.T2, self.off_resonance)
            a_tr = spoiler_matrix @ a_tr  # Apply spoiler after acquisition.

            # Containers for magnetization states.
            ### print(f"n_wet_pulses: {n_wet_pulses}")
            magnetizations = [None] * ((n_wet_pulses + 1) * 2 + 2)
            magnetizations_fid = [None] * (n_wet_pulses + 1)

            ### print(magnetizations)
            ### print("~~~~~~~~~~~~")
            ### print(magnetizations_fid)
            ### sys.exit()

            # Start magnetization along +z.
            magnetizations[0] = np.array([0.0, 0.0, 1.0], dtype=float) # TODO: x,y,z -> where y stays 0


            # Iterate multiple times to approach steady state.
            for _ in range(30):
                idx = 0

                after_wet = False

                #-----------------------------------------------------------------------------
                for ii in range(n_wet_pulses):
                    # Take previous magnetisation
                    magnetizations[idx + 1] = r_flip[ii] @ magnetizations[idx]
                    idx += 1

                    after_wet = True

                    # TODO: magnetisation FID?
                    magnetizations_fid[ii] = magnetizations[idx]

                    ## To calculate magnetisation afterwards (TODO: after pulse?)
                    # M_after = a_fp @ M_before + b_fp
                    magnetizations[idx + 1] = (a_exc_to_next_exc[ii] @ magnetizations[idx] + b_exc_to_next_exc[ii])
                    idx += 1
                #print(f"magnetizations: {magnetizations}")
                #sys.exit()
                #------------------------------------------------------------------------------

                # TODO: Last excitations after WET pulses
                magnetizations[idx + 1] = r_flip_last @ magnetizations[idx]
                idx += 1

                magnetizations[idx + 1] = (a_exc_last_to_acq @ magnetizations[idx] + b_exc_last_to_acq)
                idx += 1

                magnetizations_fid[n_wet_pulses] = magnetizations[idx]

                # TODO: At last free precession. Why?
                magnetizations[idx + 1] = a_tr @ magnetizations[idx] + b_tr
                idx += 1

                magnetizations[0] = magnetizations[idx]

            magnetization_fid_rest = np.array(
                [magnetizations_fid[i][0] for i in range(n_wet_pulses)],
                dtype=float
            )


            ### if after_wet is False:
            ###     print("with_WET = False:")
            ###     print(f"magnetizations: {magnetizations}")
            ###     print(f"magnetizations_fid : {magnetizations_fid}\n")
            ###     #print(magnetizations_fid[n_wet_pulses])
            ###     #print(magnetizations_fid[n_wet_pulses][0])
            ###     #print("lalalalalala")
            ### elif after_wet:
            ###     print("with_WET = True:")
            ###     print(f"magnetizations: {magnetizations}")
            ###     print(f"magnetizations_fid : {magnetizations_fid}\n")
            ###     #print(magnetizations_fid[n_wet_pulses])
            ###     #print(magnetizations_fid[n_wet_pulses][0])
            ###     sys.exit()

            magnetization_fid_last = magnetizations_fid[n_wet_pulses][0]

            return magnetization_fid_last, magnetization_fid_rest

###        def compute_signal_after_pulses(self, T1: float, B1_scale: float, with_WET: bool = True) -> tuple:
###            """
###            Compute the signal after multiple WET pulses followed by a final excitation pulse.
###            Only T1 and B1_scale are provided since the other parameters are stored as instance attributes.
###            Returns a tuple (magnetization_fid_last, magnetization_fid_rest) where:
###              - magnetization_fid_last (float): x-component of the magnetization at the final echo.
###              - magnetization_fid_rest (np.ndarray): x-components after each WET pulse.
###            """
###
###            # Use the WET settings if active, otherwise use empty lists.
###            flip_angles = self.flip_angles if with_WET else []
###            time_gap = self.time_gap if with_WET else []
###            n_wet_pulses = len(time_gap)
###            total_delay = np.sum(time_gap)
###
###            # Spoiler matrix: destroys transverse magnetization.
###            spoiler_matrix = np.array([
###                [0.0, 0.0, 0.0],
###                [0.0, 0.0, 0.0],
###                [0.0, 0.0, 1.0]
###            ], dtype=float)
###
###            # Precompute rotations and free precession for each WET pulse.
###            r_flip = [LookupTableWET._BlochSimulation.y_rot(B1_scale * flip_angles[ii])
###                      for ii in range(n_wet_pulses)]
###            a_exc_to_next_exc = []
###            b_exc_to_next_exc = []
###            for ii in range(n_wet_pulses):
###                a_fp, b_fp = LookupTableWET._BlochSimulation.free_precess(time_gap[ii], T1, self.T2, self.off_resonance)
###                a_exc_to_next_exc.append(spoiler_matrix @ a_fp)
###                b_exc_to_next_exc.append(b_fp)
###
###            # Final excitation pulse.
###            r_flip_last = LookupTableWET._BlochSimulation.y_rot(B1_scale * self.flip_final_excitation)
###            # Free precession from final WET pulse to acquisition.
###            a_exc_last_to_acq, b_exc_last_to_acq = LookupTableWET._BlochSimulation.free_precess(self.TE1, T1, self.T2,
###                                                                                                self.off_resonance)
###            # Free precession from acquisition to the end of TR.
###            a_tr, b_tr = LookupTableWET._BlochSimulation.free_precess(self.TR - self.TE1 - total_delay, T1, self.T2,
###                                                                      self.off_resonance)
###            a_tr = spoiler_matrix @ a_tr
###
###            # Initialize the magnetization state along +z.
###            current = np.array([0.0, 0.0, 1.0], dtype=float)
###            # We want to store the fiducial (FID) signals.
###            # Pre-initialize a list of fixed length with -1000 as a sentinel value.
###            # Each fid signal is a 3-element vector.
###            fid_signals = [np.full(3, -1000, dtype=float) for _ in range(n_wet_pulses + 1)]
###
###            # Iterate a fixed number of times to reach steady state.
###            # Instead of manual index management, we build the cycle step by step.
###            for _ in range(30):
###                cycle_fid = []  # temporary list for the current cycle's fid signals
###                M = current.copy()  # start this cycle with the current magnetization
###
###                # Apply each WET pulse:
###                for ii in range(n_wet_pulses):
###                    # 1. RF pulse rotation.
###                    M = r_flip[ii] @ M
###                    # Save the signal after the pulse.
###                    cycle_fid.append(M.copy())
###                    # 2. Free precession (with spoiling) to the next excitation.
###                    M = a_exc_to_next_exc[ii] @ M + b_exc_to_next_exc[ii]
###
###                # Final excitation pulse.
###                M = r_flip_last @ M
###                # Free precession from the final WET pulse to the acquisition.
###                M = a_exc_last_to_acq @ M + b_exc_last_to_acq
###                cycle_fid.append(M.copy())
###                # Free precession from acquisition to the end of TR.
###                M = a_tr @ M + b_tr
###                # Update the starting state for the next cycle.
###                current = M.copy()
###                # Keep the fid signals from the last cycle.
###                fid_signals = cycle_fid
###
###            # Extract the x-component from each fid signal.
###            # The last signal corresponds to the final echo.
###            magnetization_fid_last = fid_signals[-1][0]
###            # The rest are the signals after each WET pulse.
###            magnetization_fid_rest = np.array([fid[0] for fid in fid_signals[:-1]], dtype=float)
###
###            return magnetization_fid_last, magnetization_fid_rest

    def plot(self):
        """
        Plot the lookup table as a heatmap using matplotlib. Negative values
        are overlaid in red.
        :return: Nothing
        """
        # Format tick labels.
        T1_over_TR_formatted = [f"{val / self.TR:.2f}" for val in self.T1_values]
        B1_scale_formatted = [f"{val:.2f}" for val in self.B1_scales_effective_values]

        # Ensure the simulated data is a numpy array.
        data = np.array(self.simulated_data)

        # Create a figure and axis.
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create two masked arrays:
        pos_data = np.ma.masked_less(data, 0)
        neg_data = np.ma.masked_greater_equal(data, 0)

        # Plot non-negative values using the viridis colormap.
        im1 = ax.imshow(pos_data, cmap='viridis', aspect='auto')

        # Overlay negative values in red.
        im2 = ax.imshow(neg_data, cmap=ListedColormap(['red']), aspect='auto')

        # Set x and y ticks with custom labels.
        ax.set_xticks(np.arange(len(T1_over_TR_formatted)))
        ax.set_xticklabels(T1_over_TR_formatted, fontsize=7, rotation=90, ha='right')
        ax.set_yticks(np.arange(len(B1_scale_formatted)))
        ax.set_yticklabels(B1_scale_formatted, fontsize=7)

        # Add a colorbar for the viridis part.
        cbar = fig.colorbar(im1, ax=ax)
        cbar.ax.tick_params(labelsize=7)

        # Set axis titles.
        ax.set_title('Heatmap of Lookup Table')
        ax.set_xlabel('T1/TR Value')
        ax.set_ylabel('B1 Scale Value')

        plt.tight_layout()
        plt.show()

    import numpy as np
    import matplotlib.pyplot as plt


    def plot_waterfall_interactive(self):
        import plotly.graph_objects as go
        """
        Creates an interactive waterfall plot using Plotly.
        Each column (signal) of the lookup table is offset to prevent overlap,
        and segments with negative values are overlaid with a transparent red fill.
        """
        # Prepare formatted labels.
        T1_over_TR_formatted = [f"{val / self.TR:.2f}" for val in self.T1_values]
        B1_scale_values = np.array(self.B1_scales_effective_values, dtype=float)

        # Convert simulated data to a NumPy array.
        data = np.array(self.simulated_data)  # shape: (num_y, num_x)
        num_y, num_x = data.shape

        # Define offsets for waterfall effect.
        horizontal_offset = 0.1  # shift right per trace
        vertical_offset = 0.1  # shift upward per trace

        # Create a Plotly figure.
        fig = go.Figure()

        # Loop over each column (each signal).
        for i in range(num_x):
            signal = data[:, i]
            # Apply offsets.
            x_vals = B1_scale_values + i * horizontal_offset
            y_vals = signal + i * vertical_offset

            # Add the main signal trace.
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                name=f"T1/TR: {T1_over_TR_formatted[i]}",
                line=dict(width=1)
            ))

            # Identify indices where the original signal is below zero.
            neg_mask = signal < 0
            if np.any(neg_mask):
                # For the negative regions, we create a filled area between the curve and the baseline.
                x_neg = x_vals[neg_mask]
                y_neg = y_vals[neg_mask]
                # The baseline for this trace is the offset level.
                baseline = np.full_like(x_neg, i * vertical_offset)

                # Create a closed polygon for the fill area.
                fill_x = np.concatenate([x_neg, x_neg[::-1]])
                fill_y = np.concatenate([y_neg, baseline[::-1]])

                fig.add_trace(go.Scatter(
                    x=fill_x,
                    y=fill_y,
                    mode='none',
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.3)',
                    showlegend=False,
                    hoverinfo='skip'
                ))

        # Update the layout for improved aesthetics.
        fig.update_layout(
            title='Interactive Waterfall Plot with Negative Value Overlay',
            xaxis_title='B1 Scale Value (offset applied)',
            yaxis_title='Signal Amplitude (offset applied)',
            template='plotly_white'
        )

        fig.show()

    def plot_waterfall(self):
        """
        Plot each column of the lookup table as an offset signal (waterfall plot)
        and overlay a transparent red fill where the original signal is below 0.
        Additionally, annotate each curve with its T1/TR value next to the line.
        """
        # Format labels.
        T1_over_TR_formatted = [f"{val / self.TR:.2f}" for val in self.T1_values]
        B1_scale_values = np.array(self.B1_scales_effective_values, dtype=float)

        # Convert simulated data to a NumPy array (shape: [num_y, num_x]).
        data = np.array(self.simulated_data)
        num_y, num_x = data.shape

        # Offsets for the waterfall effect.
        horizontal_offset = 0.03  # shift in x per signal
        vertical_offset = 0.1  # shift in y per signal

        plt.figure(figsize=(10, 8))

        for i in range(num_x):
            # Extract the signal for the i-th column.
            signal = data[:, i]
            # Compute the x and y values with offsets.
            x_vals = B1_scale_values + i * horizontal_offset
            y_vals = signal + i * vertical_offset

            # Plot the offset signal.
            plt.plot(x_vals, y_vals, lw=1)

            # Determine the baseline (offset level) for this curve.
            base_line = i * vertical_offset
            # Calculate the portion of the original signal (before offset) that is below 0.
            negative_part = np.where(signal < 0, signal + i * vertical_offset, base_line)

            # Use fill_between to overlay the negative parts.
            plt.fill_between(x_vals, negative_part, base_line,
                             where=(signal < 0), color='red', alpha=0.3)

            # Annotate the curve: place the T1/TR label at the end of the curve.
            # Here, we use the last x and y values, with a slight offset for clarity.
            plt.text(x_vals[-1] + horizontal_offset,
                     y_vals[-1],
                     f"{T1_over_TR_formatted[i]}",
                     fontsize=7,
                     verticalalignment='center',
                     color='black')

        plt.xlabel('B1 Scale Value (with horizontal offset...)')
        plt.ylabel('Signal Amplitude (with vertical offset...)')
        plt.title('Waterfall Plot of Lookup Table')
        plt.tight_layout()
        plt.show()

    ###   def plot(self):
###       """
###       To plot the created lookup table as heatmap.
###       :return: Nothing
###       """
###       T1_over_TR_formatted = [f"{val/self.TR:.2f}" for val in self.T1_values]
###       B1_scale_formatted = [f"{val:.2f}" for val in self.B1_scales_effective_values]

###       plt.figure(figsize=(8, 6))
###       ax = sns.heatmap(self.simulated_data,
###                        annot=False,
###                        cmap='viridis',
###                        robust=True,
###                        xticklabels=T1_over_TR_formatted,
###                        yticklabels=B1_scale_formatted)
###       plt.title('Heatmap of Lookup Table')
###       plt.ylabel('B1 Scale Value')
###       plt.xlabel('T1/TR Value')

###       ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
###       ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)

###       plt.tight_layout()

###       plt.show()


###    def _find_nearest_available_keys(self, B1_scale, T1_over_TR, interpolation_type: str = "nearest", device="cpu"):
###        """
###        TODO
###        Get only indices available in lookup table.
###
###        """
###
###        if interpolation_type != "nearest":
###            raise ValueError("Only 'nearest' interpolation is supported at the moment.")
###
###        if device.lower() not in ["cpu", "cuda", "gpu"]:
###            raise ValueError(f"Device must be 'cpu' or 'gpu', but got '{device}'.")
###
###        # Select appropriate NumPy or CuPy
###        xp = np if device == "cpu" else cp
###        interpn = interpn_cpu if device == "cpu" else interpn_gpu
###
###        # Convert lookup tables to array format
###        B1_scales_effective_values = xp.asarray(self.B1_scales_effective_values)
###        T1_values = xp.asarray(self.T1_values)
###        TR = xp.asarray(self.TR)
###
###        # Create 1D grid
###        x_grid_B1_scales = xp.arange(len(B1_scales_effective_values))
###        x_grid_T1 = xp.arange(len(T1_values))
###
###        # Compute T1/TR values
###        T1_over_TR_values = T1_values / TR
###
###        # Prepare interpolation points
###        xi_B1 = B1_scale.reshape(-1)  # Convert from (1000, 1) to (1000,)
###        xi_T1_over_TR = T1_over_TR.reshape(-1)  # Convert from (1000, 1) to (1000,)
###
###
###        row_key = interpn(points=(B1_scales_effective_values,),
###                          values=x_grid_B1_scales,
###                          xi=xi_B1,
###                          method="nearest",
###                          bounds_error=False,
###                          fill_value=None)
###
###
###        col_key = interpn(points=(T1_over_TR_values,),
###                          values=x_grid_T1,
###                          xi=xi_T1_over_TR,
###                          method="nearest",
###                          bounds_error=False,
###                          fill_value=None)
###
###        # Reshape to original shape
###        row_key = row_key.reshape(B1_scale.shape)
###        col_key = col_key.reshape(T1_over_TR.shape)
###
###        return xp.stack([row_key, col_key], axis=0)


if __name__ == '__main__':

    # ========================= TO LOAD THE MAPS ANS INTERPOLATE IT ====================================================
    import file
    configurator = file.Configurator(path_folder="/home/mschuster/projects/Synthetic_MRSI/config/",
                                     file_name="paths_14032025.json")
    configurator.load()

    loaded_B1_map = file.Maps(configurator=configurator, map_type_name="B1").load_file()
    loaded_B1_map_shape = loaded_B1_map.loaded_maps.shape

    loaded_B0_map = file.Maps(configurator=configurator, map_type_name="B0").load_file()
    loaded_B0_map = loaded_B0_map.interpolate_to_target_size(target_size=loaded_B1_map_shape, order=1)


    loaded_GM_map = file.Maps(configurator=configurator, map_type_name="GM_segmentation").load_file()
    loaded_WM_map = file.Maps(configurator=configurator, map_type_name="WM_segmentation").load_file()
    loaded_CSF_map = file.Maps(configurator=configurator, map_type_name="CSF_segmentation").load_file()

    loaded_GM_map = loaded_GM_map.interpolate_to_target_size(target_size=loaded_B1_map_shape, order=1)
    loaded_WM_map = loaded_WM_map.interpolate_to_target_size(target_size=loaded_B1_map_shape, order=1)
    loaded_CSF_map = loaded_CSF_map.interpolate_to_target_size(target_size=loaded_B1_map_shape, order=1)
    Console.printf("info", "Interpolated all maps via 'order 1' to B1 map shape")

    # ==================================================================================================================

    #### show_axial_slice = 70
    #### plt.figure()
    #### fig, axs = plt.subplots(nrows=1, ncols=4)
    #### axs[0].set_title(f"WM map (axial slice {show_axial_slice})")
    #### axs[1].set_title(f"GM map (axial slice {show_axial_slice})")
    #### axs[2].set_title(f"CSF map (axial slice {show_axial_slice})")
    #### axs[3].set_title(f"WM+GM+CSF map (axial slice {show_axial_slice})")
    ####
    #### img0 = axs[0].imshow(loaded_WM_map.loaded_maps[:, :, show_axial_slice])
    #### img1 = axs[1].imshow(loaded_GM_map.loaded_maps[:, :, show_axial_slice])
    #### img2 = axs[2].imshow(loaded_CSF_map.loaded_maps[:, :, show_axial_slice])
    #### img3 = axs[3].imshow(loaded_WM_map.loaded_maps[:, :, show_axial_slice] + loaded_GM_map.loaded_maps[:, :, show_axial_slice] + loaded_CSF_map.loaded_maps[:, :, show_axial_slice])
    ####
    #### fig.colorbar(img0, ax=axs[0])
    #### fig.colorbar(img1, ax=axs[1])
    #### fig.colorbar(img2, ax=axs[2])
    #### fig.colorbar(img3, ax=axs[3])
    #### #plt.show()

    # =============================== TO CREATE THE LOOKUP TABLE =======================================================
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
                                           time_gaps_WET=[30, 30, 30],
                                           off_resonance=0)

    lookup_table_WET_test.create()
    lookup_table_WET_test.plot()
    #lookup_table_WET_test.plot_waterfall_interactive()
    lookup_table_WET_test.plot_waterfall()
    # ==================================================================================================================

    # =========================== To check different off-resonance frequency ===========================================
###    lookup_table_WET_test_2 = LookupTableWET(T1_range=[300, 5000],
###                                             T1_step_size=50,
###                                             T2=250,
###                                             B1_scales_inhomogeneity=[1e-10,3],
###                                             B1_scales_gauss=[0.01, 1],
###                                             B1_scales_inhomogeneity_step_size=0.05,
###                                             B1_scales_gauss_step_size=0.05,
###                                             TR=TR,
###                                             TE=0,
###                                             flip_angle_excitation_degree=47.0,
###                                             flip_angles_WET_degree=[89.2, 83.4, 160.8],
###                                             time_gaps_WET=[30, 30, 30],
###                                             off_resonance=100)
###
###    lookup_table_WET_test.create()
###    lookup_table_WET_test_2.create()
###
###
###    #####################################
###    #####################################
###    # Format labels for the axes.
###    T1_over_TR_formatted = [f"{val / lookup_table_WET_test.TR:.2f}" for val in lookup_table_WET_test.T1_values]
###    B1_scale_formatted = [f"{val:.2f}" for val in lookup_table_WET_test.B1_scales_effective_values]
###
###    # Compute the difference between the simulated datasets.
###    data_diff = np.array(lookup_table_WET_test.simulated_data) - np.array(lookup_table_WET_test_2.simulated_data)
###
###    print("------------------------------------------------------------------------")
###    print(f"data_diff min: {np.min(data_diff)} | data_diff max: {np.max(data_diff)}")
###    print("------------------------------------------------------------------------")
###
###    # Mask negative values so only non-negative differences are plotted.
###    pos_data = np.ma.masked_less(data_diff, 0)
###
###    # Create a figure and axis.
###    fig, ax = plt.subplots(figsize=(8, 6))
###
###    # Plot the positive differences using the viridis colormap.
###    im = ax.imshow(pos_data, cmap='viridis', aspect='auto')
###
###    # Set x and y ticks with custom labels.
###    ax.set_xticks(np.arange(len(T1_over_TR_formatted)))
###    ax.set_xticklabels(T1_over_TR_formatted, fontsize=7, rotation=90, ha='right')
###    ax.set_yticks(np.arange(len(B1_scale_formatted)))
###    ax.set_yticklabels(B1_scale_formatted, fontsize=7)
###
###    # Add a colorbar for the positive differences.
###    cbar = fig.colorbar(im, ax=ax)
###    cbar.ax.tick_params(labelsize=7)
###
###    # Set axis titles.
###    ax.set_title('Heatmap of Difference Lookup Table')
###    ax.set_xlabel('T1/TR Value')
###    ax.set_ylabel('B1 Scale Value')
###
###    plt.tight_layout()
###    plt.show()
###
###    print("lalalalalalalalalalala lllll")
###    sys.exit()
###
###    #####################################
###    #####################################
    # ==================================================================================================================




    # ===================== Display B1 map =============================================================================
    ###shape_B1_map = loaded_B1_map.loaded_maps.shape
    ###scaled_B1_map[scaled_B1_map < 0] = 0.001
    ###
    ###plt.figure()
    ###plt.imshow(scaled_B1_map[:,:,50])
    ###plt.colorbar()
    ###plt.show()
    ###Console.printf("info", f"Scaled B1 Map: min={np.min(scaled_B1_map)}, max={np.max(scaled_B1_map)}")
    # ==================================================================================================================


    # =============================== Create map from lookup table =====================================================
    scaled_B1_map = loaded_B1_map.loaded_maps / 39.0
    scaled_B1_map[scaled_B1_map < 0] = 0.001
    Console.printf("info", "set values < 0 to 0 in scaled B1 map (map/39°)")

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

    # T1_over_TR_GM_map + T1_over_TR_WM_map + T1_over_TR_CSF_map --> lookup_table_WET_test.simulated_data.get_value

    Console.add_lines("Scaled maps:")
    Console.add_lines(f" => GM min: {np.min(T1_over_TR_GM_map)}; max: {np.max(T1_over_TR_GM_map)}")
    Console.add_lines(f" => WM min: {np.min(T1_over_TR_WM_map)}; max: {np.max(T1_over_TR_WM_map)}")
    Console.add_lines(f" => CSF min: {np.min(T1_over_TR_CSF_map)}; max: {np.max(T1_over_TR_CSF_map)}")
    Console.printf_collected_lines("info")
    #sys.exit()

    #print(np.min(scaled_B1_map.flatten()), np.max(scaled_B1_map.flatten()))
    #print(np.full_like(scaled_B1_map, 1).flatten()))
    #lookup_table_WET_test.simulated_data.set_interpolation_method(method="linear")
    lookup_table_WET_test.simulated_data.set_interpolation_method(method="linear")

    #result_GM = lookup_table_WET_test.simulated_data.get_value(B1_scale_effective=scaled_B1_map, T1_over_TR=T1_over_TR_GM_map)
    #result_WM = lookup_table_WET_test.simulated_data.get_value(B1_scale_effective=scaled_B1_map, T1_over_TR=T1_over_TR_WM_map)
    #result_CSF = lookup_table_WET_test.simulated_data.get_value(B1_scale_effective=scaled_B1_map, T1_over_TR=T1_over_TR_CSF_map)

    result_all = lookup_table_WET_test.simulated_data.get_value(B1_scale_effective=scaled_B1_map,
                                                                T1_over_TR=T1_over_TR_GM_map + T1_over_TR_WM_map + T1_over_TR_CSF_map)
    Console.printf("success", "Get map from Lookup Table via T1/R = (T1/TR)_GM + (T1/TR)_WM + (T1/TR)_CSF)")

    # ==================================================================================================================

    # ====================== OLD: separate plot LookupTable results of WM, GM, CSF =====================================
    ###plt.figure(1)
    ###plt.title("GM Remaining Signal Map")
    ###plt.subplot(2, 3, 1)
    ###plt.imshow(result_GM[:, :, 50])
    ###plt.subplot(2, 3, 2)
    ###plt.imshow(np.rot90(result_GM[:, 90, :]))
    ###plt.subplot(2, 3, 3)
    ###plt.imshow(np.rot90(result_GM[90, :, :]))
    ###plt.colorbar()
    ###plt.subplot(2, 3, 4)
    ###plt.imshow(np.log2(result_GM[:, :, 50]+0.1))
    ###plt.subplot(2, 3, 5)
    ###plt.imshow(np.log2(np.rot90(np.abs(result_GM[:, 90, :]+0.1))))
    ###plt.subplot(2, 3, 6)
    ###plt.imshow(np.log2(np.rot90(result_GM[90, :, :]+0.1)))
    ###plt.colorbar()


    ###plt.figure(2)
    ###plt.title("WM Remaining Signal Map")
    ###plt.subplot(2, 3, 1)
    ###plt.imshow(result_WM[:, :, 50])
    ###plt.subplot(2, 3, 2)
    ###plt.imshow(np.rot90(result_WM[:, 90, :]))
    ###plt.subplot(2, 3, 3)
    ###plt.imshow(np.rot90(result_WM[90, :, :]))
    ###plt.colorbar()
    ###plt.subplot(2, 3, 4)
    ###plt.imshow(np.log2(result_WM[:, :, 50]+0.1))
    ###plt.subplot(2, 3, 5)
    ###plt.imshow(np.log2(np.rot90(np.abs(result_WM[:, 90, :]+0.1))))
    ###plt.subplot(2, 3, 6)
    ###plt.imshow(np.log2(np.rot90(result_WM[90, :, :]+0.1)))
    ###plt.colorbar()


    ####plt.figure(3)
    ####plt.title("CSF Remaining Signal Map")
    ####plt.subplot(2, 3, 1)
    ####plt.imshow(result_CSF[:, :, 50])
    ####plt.subplot(2, 3, 2)
    ####plt.imshow(np.rot90(result_CSF[:, 90, :]))
    ####plt.subplot(2, 3, 3)
    ####plt.imshow(np.rot90(result_CSF[90, :, :]))
    ####plt.colorbar()
    ####plt.subplot(2, 3, 4)
    ####plt.imshow(np.log2(result_CSF[:, :, 50]+0.1))
    ####plt.subplot(2, 3, 5)
    ####plt.imshow(np.log2(np.rot90(np.abs(result_CSF[:, 90, :]+0.1))))
    ####plt.subplot(2, 3, 6)
    ####plt.imshow(np.log2(np.rot90(result_CSF[90, :, :]+0.1)))
    ####plt.colorbar()
    ####plt.show()
    # ==================================================================================================================

    # ==================== Plot combination GM+WM+CSF ==================================================================


    plt.figure(4)
    plt.title("Remaining Signal Map (used (T1/TR)_GM map + (T1/TR_WM)_map + (T1/TR_CSF)_map as LookupTable input) (all)")
    plt.subplot(1, 3, 1)
    plt.imshow(result_all[:, :, 50])
    plt.subplot(1, 3, 2)
    plt.imshow(np.rot90(result_all[:, 90, :]))
    plt.subplot(1, 3, 3)
    plt.imshow(np.rot90(result_all[90, :, :]))
    plt.colorbar()
    plt.show()
    ###plt.subplot(2, 3, 4)
    ###plt.imshow(np.log2(result_all[:, :, 50]+0.1))
    ###plt.imshow(np.log2(result_all[:, :, 50]+0.1))
    ###plt.subplot(2, 3, 5)
    ###plt.imshow(np.log2(np.rot90(np.abs(result_all[:, 90, :]+0.1))))
    ###plt.subplot(2, 3, 6)
    ###plt.imshow(np.log2(np.rot90(result_all[90, :, :]+0.1)))
    ###plt.colorbar()
    #plt.show()

    # T1/TR und B1+ -> für Kombinationen wird negativ im Lookup Table
    # If B0 map in ppm ---> 297.223 Hz/ppm
    # B0 * pulse * B1+ * T1/TR

    # ==================================================================================================================


    # ================= To incorporate the B0 map and Gauss Pulse ======================================================
    import scipy.io as sio

    path_gauss_wet = configurator.data["path"]["pulse"]["gauss_wet"]
    gauss_wet_pulse = sio.loadmat(path_gauss_wet)
    amplitude_pulse = gauss_wet_pulse["Pulse1_Freq"].squeeze()
    frequency_pulse = gauss_wet_pulse["Freq_Vec"].squeeze()

    import matplotlib.pyplot as plt
    plt.plot(frequency_pulse, amplitude_pulse)
    plt.show()

    plt.figure(4)
    plt.title("Loaded B0 map")
    plt.subplot(1, 3, 1)
    plt.imshow(loaded_B0_map.loaded_maps[:, :, 50])
    plt.subplot(1, 3, 2)
    plt.imshow(np.rot90(loaded_B0_map.loaded_maps[:, 90, :]))
    plt.subplot(1, 3, 3)
    plt.imshow(np.rot90(loaded_B0_map.loaded_maps[90, :, :]))
    plt.colorbar()
    plt.show()


    from scipy.interpolate import interp1d
    # to create interpolation function:
    interp_func = interp1d(frequency_pulse, amplitude_pulse)

    loaded_B0_map_flat = loaded_B0_map.loaded_maps.ravel()
    scaled_B0_map_wet_pulse = np.empty_like(loaded_B0_map_flat)


    # =============================================================================================================================

    # 1 Understand the problem!
    #   --> have shape 180 x 180 x 109 MB --> yields ~27 MB --> lets calculate with 30 MB --> thus using 1000 at once yields 30000 MB (30 GB)
    #
    #
    #

    dummy = np.empty((180, 109), dtype=np.float64)

    # 2 Develop solutions

    Console.start_timer()
    result_all_gauss = lookup_table_WET_test.simulated_data.get_value(
        B1_scale_effective=dummy,
        T1_over_TR=T1_over_TR_GM_map + T1_over_TR_WM_map + T1_over_TR_CSF_map)
    Console.stop_timer()

    sys.exit()

    Console.printf("info", "Scaling B0 values based on pulse")

    # for just one Δf_B0 --> need A(B1_scale * B1_Gauss(Δf + Δf_B0))
    delta_f = 0
    for i, item in tqdm(enumerate(loaded_B0_map_flat), total=len(loaded_B0_map_flat)):
        scaled_B0_map_wet_pulse[i] = interp_func(delta_f + -1*loaded_B0_map_flat[i]) # TODO: Added minus as Bernhard told!

    scaled_B0_map_wet_pulse = scaled_B0_map_wet_pulse.reshape(loaded_B0_map.loaded_maps.shape)

    #amplitude_at_freq = interp_func(freq_to_find)


    # =============================================================================================================================

    for i in tqdm(range(500000), total=500000):
        result_all_gauss = lookup_table_WET_test.simulated_data.get_value(B1_scale_effective=scaled_B1_map * scaled_B0_map_wet_pulse,
                                                                          T1_over_TR=T1_over_TR_GM_map + T1_over_TR_WM_map + T1_over_TR_CSF_map)

    Console.printf("success", "Get map from Lookup Table via T1/R = (T1/TR)_GM + (T1/TR)_WM + (T1/TR)_CSF) AND Gauss(B0)")
    # TODO -B0?


    plt.figure(5)
    plt.title("Remaining Signal Map (used (T1/TR)_GM map + (T1/TR_WM)_map + (T1/TR_CSF)_map as LookupTable input) + Gauss Pulse")
    plt.subplot(2, 3, 1)
    plt.imshow(result_all_gauss[:, :, 50])
    plt.subplot(2, 3, 2)
    plt.imshow(np.rot90(result_all_gauss[:, 90, :]))
    plt.subplot(2, 3, 3)
    plt.imshow(np.rot90(result_all_gauss[90, :, :]))
    plt.colorbar()
    plt.subplot(2, 3, 4)
    plt.imshow(np.log2(result_all_gauss[:, :, 50]+0.1))
    plt.imshow(np.log2(result_all_gauss[:, :, 50]+0.1))
    plt.subplot(2, 3, 5)
    plt.imshow(np.log2(np.rot90(np.abs(result_all_gauss[:, 90, :]+0.1))))
    plt.subplot(2, 3, 6)
    plt.imshow(np.log2(np.rot90(result_all_gauss[90, :, :]+0.1)))
    plt.colorbar()
    plt.show()
