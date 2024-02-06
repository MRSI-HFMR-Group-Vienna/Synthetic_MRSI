from printer import Console
import numpy as np
import sys


class FID:
    """
    The FID includes the basic attributes, including the signal and time vector, as
    well as the name of the chemical compound refereed to it. Further, the T2 value and
    the concentration.
    Also, it is possible to get the signal in various data types and thus, if
    necessary, decrease the memory load.
    """

    def __init__(self, signal: np.ndarray, time: np.ndarray, name: list[str], signal_data_type: np.dtype = np.complex64):
        if not signal.shape[-1] == time.shape[-1]:
            Console.printf("error",
                           f"Shape of signal and time vector does not match! Signal length is {signal.shape[-1]} while time length is {time.shape[-1]}. Terminating the program!")
            sys.exit()

        self.signal: np.ndarray = signal  # signal vector
        self.time: np.ndarray = time      # time vector
        self.name: list = name            # name, e.g., of the respective metabolite
        self.concentration: float = None
        self.t2_value: float = None


    def get_signal(self, signal_data_type: np.dtype, mute=True):
        """
        To get the signal with a certain precision. Useful to reduce the required space.

        :param signal_data_type:
        :param mute: By default, True. If False, then the precision, according to the data type, is printed to the console.
        :return: Amplitude of chosen data type and thus precision.
        """
        signal = self.signal.astype(dtype=signal_data_type)
        if not mute:
            Console.printf("info", f"Get Amplitude of {self.name} with precision of {np.finfo(signal.dtype).precision} decimal places")

        return signal

    def show_signal_shape(self) -> None:
        """
        Print the shape of the FID signal to the console.

        :return: None
        """
        Console.printf("info", f"FID Signal shape: {self.signal.shape}")

    def __add__(self, other):
        """
        For merging different FID signals. Add two together with just "+"
        :param other:
        :return:
        """
        if not np.array_equal(self.time, other.time):
            Console.printf("error", f"Not possible to sum the two FIDs since the time vectors are different! Vector 1: {self.time.shape}, Vector 2; {other.times.shape}")
            return
        if not self.signal.shape[-1] == other.signal.shape[-1]:
            Console.printf("error", "Not possible to sum the two FIDs since the length does not match!")
            return
        else:
            fid = FID(signal=self.signal, time=self.time, name=self.name) # new empty fid object
            fid.signal = np.vstack((self.signal, other.signal))           # vertical stack signals, thus add row vectors
            #fid.name.extend(["other.name"])
            return fid

    def __str__(self):
        return f"FID of chemical compound '{self.name}' with signal shape: {self.signal.shape}"


class Spectrum:
    """
    Transform the :class: `~FID` from the time domain to the frequency domain.
    TODO: What else?
    """

    def __init__(self, fid: FID):
        self.fid = fid
        self.signal: np.ndarray = np.array([])  # TODO: in frequency domain, from FT from FID
        self.frequency: np.ndarray = np.array([])  # TODO: in frequency domain, from FT from FID

    def __add__(self, other):
        # TODO: For summing up the individual spectra. Also, store information about what spectra are summed.
        pass

    def fourier_transform_FID(self):
        # TODO: Chose from the FID object the relevant parts and the achieve a fourier transformation!
        raise NotImplementedError("This method is not yet implemented")


class Model:
    # TODO What should it contain?

    def __int__(self):
        self.metabolites: list[Spectrum] = None
        self.lipids: list[Spectrum] = None

    def __init__(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def add_metabolite(self, metabolite: Spectrum) -> None:
        """
        Add a metabolite to the list of metabolites. From file, simulation, etc. possible. However, to create or use
        an object of class :class: `~Spectrum` is necessary.
        TODO: Could be also a dict due to easily access by name?

        :param metabolite: Spectrum (containing also the FID) of a metabolite
        :return: Nothing
        """
        self.metabolites.append(metabolite)
        Console.printf("success", f"Added metabolite '{metabolite.fid.name} to spectral model.")

    def add_lipid(self, lipid: Spectrum) -> None:
        """
        Add a lipid to the list of lipids. From file, simulation, etc. possible. However, to create or use
        an object of class :class: `~Spectrum` is necessary.
        TODO: Could be also a dict due to easily access by name?

        :param lipid: Spectrum (containing also the FID) of a lipid
        :return: Nothing
        """
        self.lipids.append(lipid)
        Console.printf("success", f"Added lipid '{lipid.fid.name} to spectral model.")

    def sum_spectra(self):
        # TODO: Should be a function of Spectrum (but execute here)?
        # TODO: Use the __add__ method in the class Spectrum
        raise NotImplementedError("This method is not yet implemented")

    def line_broader(self):
        # TODO: Here or in Simulator?
        raise NotImplementedError("This method is not yet implemented")


class Simulator:
    # TODO
    # It should be able to simulate all spectral elements.

    def __init__(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def metabolites(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def water(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def lipids(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def macromolecules(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def water_suppression(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def lipid_suppression(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

