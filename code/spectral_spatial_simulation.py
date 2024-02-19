from printer import Console
from tqdm import tqdm
import numpy as np
import os, sys


class FID:
    """
    The FID includes the basic attributes, including the signal and time vector, as
    well as the name of the chemical compound refereed to it. Further, the T2 value and
    the concentration.
    Also, it is possible to get the signal in various data types and thus, if
    necessary, decrease the memory load.
    """

    def __init__(self, signal: np.ndarray = None, time: np.ndarray = None, name: list[str] = None, signal_data_type: np.dtype = None, sampling_period: float = None):
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
        if signal_data_type is not None:
            self.signal: np.ndarray = signal.astype(signal_data_type)
        else:
            self.signal: np.ndarray = signal  # signal vector

        self.time: np.ndarray = time      # time vector
        self.name: list = name            # name, e.g., of the respective metabolite
        self.concentration: float = None
        self.t2_value: np.ndarray = None # TODO
        self.sampling_period = sampling_period # it is just 1/(sampling frequency)


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
        self.signal = np.sum(self.signal, axis=0) # To sum up all signals
        self.name = [" + ".join(self.name)]       # To put all compound names in one string (in a list)
        return self


    def get_spectrum(self) -> np.ndarray:
        """
        To get the spectrum of each
        """
        # TODO: just check if right axis is used. Should be fft to each row here?
        # TODO: Also think about the return type! Maybe just return magnitude?
        frequency = np.fft.fftfreq(self.time.size, self.sampling_period)
        magnitude = np.fft.fft(self.signal, axis=1)
        return {'frequency' : frequency,
                'magnitude' : magnitude}

    def change_signal_data_type(self, signal_data_type: np.dtype) -> None:
        """
        For changing the data type of the FID. Possible usecase: convert FID signals to lower bit signal, thus reduce required space.

        :param signal_data_type: Numpy data type
        :return: Nothing
        """
        self.signal = self.signal.astype(signal_data_type)


    def get_partially_FID(self, fid_indices: list[int]): # TODO define FID as return type -> FID doe snot work
        """
        To get only a fraction of the original FID.

        :param fid_indices: List of indices of the respective FIDs in the self object.
        :return: FID object
        """

        signal_partially = self.signal[fid_indices]
        name_partially = [self.name[i] for i in fid_indices]
        time = self.time
        #t2_values TODO
        #sampling_period TODO

        return FID(signal=signal_partially, name=name_partially, time=time)

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
            fid = FID(signal=self.signal, time=self.time, name=self.name)     # new fid object containing the information of this object
            fid.signal = np.vstack((self.signal, other.signal))               # vertical stack signals, thus add row vectors
            fid.name = self.name.copy() + other.name.copy()                   # since lists not immutable, copy() is required!

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

class B0Inhomogeneities:
    # TODO: It should be one map per volunteer
    def __init__(self):
        # TODO
        # data -> a map?
        # unit -> can be Hz, ppm, arbitrary
        raise NotImplementedError("This method is not yet implemented")

class Model:
    # TODO What should it contain?

    def __init__(self, path_cache: str, file_name_cache: str):
        if os.path.exists(path_cache):
            self.path_cache = path_cache
        else:
            Console.printf("error", f"Terminating the program. Path does not exist: {path_cache}")
            sys.exit()

        self.file_name_cache = file_name_cache
        self.fid = FID() # instantiate an empty FID to be able to sum it ;)
        self.fid_scaling_map = None
        self.mask = None
        self.volume = None # TODO: Rename -> basically the output of the model -> is it a metabolic map?
        self.model_shape = None # shape of volume

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
            self.fid + fid # sum fid according to the __add__ implementation in the FID class
            Console.printf("success", f"Added compound '{fid.name} to the spectral spatial model.")
        except Exception as e:
            Console.printf("error", f"Error in adding compound '{fid.name} to the spectral spatial model. Exception: {e}")

    def add_mask(self, mask: np.ndarray) -> None:
        # TODO: Should it be an object containing further information of the Mask, transformations, etc.. -> then make class "Mask"!
        # TODO: Just one mask per model?
        """
        For adding one mask to the model. It is just an numpy array with no further information.

        :param mask: Numerical values of the mask as numpy array
        :return: Nothing
        """
        self.mask = mask

    def add_fid_scaling_map(self, fid_scaling_map: np.ndarray):
        """
        For scaling the FID at the respective position in the volume. TODO: Right way? Maybe rename?

        :param fid_scaling_map: Values to scale the FID at the respective position in the volume
        :return: Nothing
        """

        self.fid_scaling_map = fid_scaling_map


    def line_broader(self):
        # TODO: Here or in Simulator?
        raise NotImplementedError("This method is not yet implemented")

    def build(self, data_type: np.dtype = np.complex64):
        Console.printf_section("Spectral-spatial-simulation")
        # For building the model: This includes at the current moment mask * map * FID -> (x,y,z, metabolite, FID(s))
        # Thus, required is:
        #   self.mask
        #   self.FID
        #   self.fid_scaling_map -> TODO: contains at the moment values in the range [0,1].


        # (a) Set path where memmap file should be created and create memmap
        #     Or load memmap if file already exits.
        path_cache_file = os.path.join(self.path_cache, f"{self.file_name_cache}_array.npy")
        Console.printf("info", f"Using cache path: {path_cache_file}")

        if (os.path.exists(path_cache_file)
                and (Console.ask_user(f"(y) Overwrite existing file '{self.file_name_cache}'? Or (n) load old one instead", exit_if_false=False) is not True)):
            Console.printf("info", f"loading the file '{self.file_name_cache}' instead because it already exits in the selected path!")
            self.model_shape = np.load(f"{self.file_name_cache}_shape.npy")
            self.volume = np.memmap(path_cache_file, dtype=data_type, mode='r+', shape=tuple(self.model_shape)) # TODO TODO TODO save somewhere the shape!
            Console.printf("success", f"Loaded volume of shape: {self.volume.shape}")
            return


        # (b) Get the required shape of the model & create cache file
        # -> if only one FID signal is added, then add one dimension
        # -> if multiple FID signals are added, then don't add dimensions
        # => final dimension need to be (x,y,z, Metabolite, FID)
        fid_shape = (1,) + self.fid.signal.shape if self.fid.signal.ndim == 1 else self.fid.signal.shape
        self.model_shape = (self.mask.shape + fid_shape)
        np.save(os.path.join(f"{self.file_name_cache}_shape.npy"), np.array(self.model_shape))

        # (c) Create memmap on disk with 0s
        self.volume = np.memmap(path_cache_file, dtype=data_type, mode='w+', shape=self.model_shape)
        #volume[:] = 0

        # (c) Estimate space required on hard drive
        #  -> self.model_shape.size returns the shape and np.prod the number of elements in the planned memmap array.
        #  -> the np.dtype(data_type) creates and object and .itemsize returns the size (in bytes) of each element
        space_required_mb = np.prod(self.model_shape) * np.dtype(data_type).itemsize * 1 / (1024 * 1024)

        # (d) Ask the user for agreement of the required space
        # Check if the simulation does not exceed a certain size
        # Console.check_condition(space_required_mb < 30000, ask_continue=True)
        Console.ask_user(f"Estimated required space [MB]: {space_required_mb}")


        # (e) Find non zero elements in the mask
        mask_nonzero_indices = np.nonzero(self.mask)

        Console.printf("info", "Begin assigning FID to volume:")
        indent = " " * 22
        for x,y,z in tqdm(zip(*mask_nonzero_indices), total=len(mask_nonzero_indices[0]), desc=indent+"Assigning FID to volume"):
            for fid_signal_index in range(self.fid.signal.ndim):
                self.volume[x,y,z,fid_signal_index,:] = self.fid.signal[fid_signal_index] * self.fid_scaling_map[x,y,z]

        Console.printf("success", f"Created volume of shape: {self.volume.shape}")
        self.volume.flush() # Write any changes in the array to the file on disk.
        Console.printf("success", f"Flushed updated volume to path: {self.path_cache}. Actual used space [MB]: {os.path.getsize(path_cache_file) / (1024*1024)}")

        # Option 2: However: volume at end is not a memmap any more!
        #volume = np.memmap(path_cache_file, dtype=data_type, mode='w+', shape=(112,128,80,1,1))
        #mask = self.mask[:,:,:,np.newaxis, np.newaxis]
        #fid = self.fid.signal[np.newaxis,np.newaxis,np.newaxis, np.newaxis,:]
        #scaling = self.fid_scaling_map[:,:,:,np.newaxis, np.newaxis]
        #volume = mask * fid * scaling
        # Just for checking the shape
        #print(f"self.fid.signal.shape: {self.fid.signal.shape}")
        #print(f"self.fid_scaling_map.shape: {self.fid_scaling_map.shape}")
        #print(f"volume.shape: {volume.shape}")


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
