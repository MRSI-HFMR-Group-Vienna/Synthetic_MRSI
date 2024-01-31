import default

# for JMRUI
import json
import os
import sys
import numpy as np
from oct2py import Oct2Py

import spatial
import spectral
from printer import Console
from more_itertools import collapse

# for NeuroImage
import nibabel as nib
from pathlib import Path

# for Configurator
import datetime


class JMRUI2:
    """
    For reading the data from an m.-File generated from an jMRUI. TODO: Implement for the .mat file.
    """

    def __init__(self, path: str, signal_data_type: np.dtype = np.float64, mute: bool = False):
        self.path = path
        self.signal_data_type = signal_data_type
        self.mute = mute

    def load_m_file(self) -> dict:
        parameters: dict = {}

        # Read the content of the .m file
        with open(self.path, 'r') as file:
            file_content = file.read()
            file_content = file_content.replace('{', '[').replace('}', ']')

        # Create a dictionary to store variable names and their values
        data_found = False  # for entering the mode to append the FID data
        amplitude = "["
        for line in file_content.splitlines():
            parts = line.split('=')
            if len(parts) == 2:
                var_name = parts[0].strip()
                var_value = parts[1].strip().rstrip(';')  # Remove trailing ';' if present
                parameters[var_name] = var_value

            if data_found:
                amplitude += line + ", "
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
        time_vector = parameters["DIM_VALUES"][0].split(":")
        time_vector_start, time_vector_stepsize, time_vector_end = float(time_vector[0]), float(time_vector[1]), float(time_vector[2])
        time = np.arange(time_vector_start, time_vector_end, time_vector_stepsize)
        time = time[0:len(time) - 1]

        # For transforming the signal values given in the file to a numpy array
        parameters['SIZE'] = eval(parameters['SIZE'])  # str to list
        amplitude = amplitude.replace('\t', ' ').replace('[', '').replace(']', '').replace(';', '')  # .split(",")
        amplitude_list_strings = amplitude.split(",")
        amplitude_list_strings = [[float(num) for num in string_element.split()] for string_element in amplitude_list_strings]
        amplitude = np.asarray(amplitude_list_strings[0:len(amplitude_list_strings) - 2])

        # Short overview of a chosen data type for the FID signal, used space and precision (in digits)
        Console.printf("info", f"Loaded FID signal as {self.signal_data_type} \n" +
                       f" -> thus using space: {amplitude.nbytes / 1024} KB \n" +
                       f" -> thus using digits: {np.finfo(amplitude.dtype).precision}",
                       mute=self.mute)

        return {"parameters": parameters,
                "signal": amplitude,
                "time": time}

    def show_parameters(self) -> None:
        """
        Printing the successfully read parameters formatted to the console.

        :return: None
        """
        # np.set_printoptions(20)  # for printing numpy numbers with 20 digits to the console
        for key, value in self.parameters.items():
            Console.add_lines(f"Key: {key}, Value: {value}")

        Console.printf_collected_lines("success")
        # np.set_printoptions() resetting numpy printing options


class JMRUI:
    """
    For reading the data from an m.-File generated from an jMRUI. TODO: Implement for the .mat file.
    """

    def __init__(self, path: str, signal_data_type: np.dtype = np.float64, mute: bool = False):
        self.path = path
        self.signal_data_type = signal_data_type
        self.mute = mute

    def load_m_file(self) -> dict:
        """
        Read content from an MATLAB (.m) file, interpret it via Octave and assign it to the following object variables:
         -> self.signal
         -> self.parameters
        :return: None
        """

        parameters: dict = {}

        # Check if file and path exists
        if not os.path.exists(self.path):
            Console.printf("error", "Path and/or file does not exist! Termination of the program.")
            sys.exit()  # Termination of the program

        file = open(self.path, 'r')  # Open the .m file
        oc = Oct2Py(convert_to_float=False)  # Create an Oct2Py object, thus running an Octave session

        data_string = "["  # Creating a string that holds the FID data which later is interpreted with Octave.
        parameter_name = ""  # Initial value needed for the if statement below.

        # Iterate through all lines in the MATLAB (.m) file
        for line_number, line_content in enumerate(file):
            # (1) Split the data, thus get the parameter names

            if parameter_name != "DATA":  # (!) If DATA occurs, the line will be skipped because of
                parameter_name, _ = line_content.split("=")  # continue and it will never enter again in this if
                if parameter_name == "DATA": continue  # condition

                # (2) For each line that can be interpreted by octave (thus no error), unwrap and transform to python
                # and insert into dictionary
                try:
                    oc.eval(line_content)  # evaluate line by octave
                    content_raw = oc.pull(parameter_name)  # get values by name from workspace (octave session)

                    # unwrapping to level 3 and assembly names again
                    if parameter_name == "DIM_VALUES":
                        content_transformed = list(collapse(content_raw, levels=3))
                        content = content_transformed[:2] + [content_transformed[2:]]

                    # unwrapping to level 1
                    else:
                        content_transformed = list(collapse(content_raw, levels=1))
                        content = content_transformed

                    # (3) Assignment of the content. If a list contains just one item then unwrap it again.
                    parameters[parameter_name] = content[0] if (len(content) == 1 and type(content) == list) else content

                except:
                    Console.add_lines(f"Error in line {line_number + 1} with content: {line_content} -> It will be excluded!")

            # (4) Treat data differently: create big string, interpreted it by octave and insert to dictionary.
            else:
                data_string += line_content + "\n"

        # Print out the excluded lines, i.e. those that could not be interpreted with octave.
        Console.printf_collected_lines("warning", mute=self.mute)

        # Convert signal to desired data type and time (a list) to a numpy array
        amplitude: np.ndarray = oc.eval(data_string).astype(self.signal_data_type)  # Transform signal values as string to numpy.ndarray()
        time: np.ndarray = np.asarray(parameters["DIM_VALUES"][0])  # Add time vector to time variable
        time = time[0:len(time) - 1]  # Otherwise time vector is +1 longer than signal vector

        # End octave session and also close the file
        oc.exit()
        file.close()

        # Short overview of chosen data type for the FID signal, used space and precision (in digits)
        Console.printf("info", f"Loaded FID signal as {self.signal_data_type} \n" +
                       f" -> thus using space: {amplitude.nbytes / 1024} KB \n" +
                       f" -> thus using digits: {np.finfo(amplitude.dtype).precision}",
                       mute=self.mute)

        return {"parameters": parameters,
                "signal": amplitude,
                "time": time}


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

    def load_nii(self) -> object:
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
        self.data = self.nifti_object.get_fdata()
        self.shape = self.nifti_object.shape

        Console.printf("success", f"Loaded file '{self.name}':"
                                  f"\n    Shape -> {self.shape}"
                                  f"\n    Pixel dimensions: -> {self.header.get_zooms()}"
                                  f"\n    In memory cache? -> {self.nifti_object.in_memory}",  # TODO what was exactly the purpose?
                       mute=False)

        return self


class Configurator:
    """
    For loading paths and configurations from a json file. If necessary, different
    instances for different config files can be created.
    """

    def __init__(self, path_folder: str, file_name: str, info: str = None) -> None:
        if os.path.exists(path_folder):  # check if the path to the folder exists
            self.path_folder: str = path_folder  # path to the config file
        else:
            Console.printf("error", f"Path does not exists: {path_folder}. Terminate program!")
            sys.exit()

        self.file_name: str = file_name  # file name or desired file name
        self.file_path: str = os.path.join(self.path_folder, self.file_name)  # file name and path
        self.info: str = info  # additional information of the configurator instance
        self.data: dict = None

    def load(self) -> object:
        """
        For loading a json file and storing it as a dictionary.

        :return: Nothing
        """
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                self.data = json.load(file)
        else:
            Console.printf("error", f"Could not load '{self.file_name}'. Wrong path, name or may not exist!")

        return self

    def save(self, new_data: dict):
        """
        For creating or overwriting a formatted json config file from a dictionary.

        :param new_data:
        :return:
        """

        if os.path.exists(self.file_path):
            Console.ask_user(f"Overwrite file '{self.file_path}' ?")
        else:
            Console.printf("info", f"New file '{self.file_name}' will be created")

        try:
            with open(self.file_path, "w") as file:
                json.dump(new_data, file, indent=4, default=str)  # default str converts everything not known to a string
        except Exception as e:
            Console.printf("error", f"Could not create/overwrite the file: {type(e).__name__}: {e}")

    def print_formatted(self) -> None:
        """
        For printing the content of the JSON file to the console.

        :return: None
        """

        Console.printf("info", f"Content of the config file: {self.file_name} \n"
                               f"{json.dumps(self.data, indent=4)}")


class Mask:
    """
    For loading a mask from a file (e.g., metabolic mask, lipid mask, B0 inhomogeneities, ...). It requires a
    available JSON configuration file. See :class: `Configurator`.
    """

    @staticmethod
    def load(configurator: Configurator, mask_name: str) -> NeuroImage:
        """
        For loading a mask from a path. The necessary path is available in a JSON config file (path_file),
        the :class: `Configurator does handle it.

        :param configurator: Handles the paths, thus also for the masks.
        :param mask_name: The name of the mask. Needed to be available in the JSON file.
        :return: An object of the class :class: `Neuroimage`. The variable 'data' of the object returns the mask data itself (numpy memmap).
        """

        # Load the paths from JSON file in the configurator object
        configurator.load()

        # Then, check if Mask exists and load the path. Otherwise, warn the user and exit program.
        available_masks = list(configurator.data["path"]["mask"].keys())
        if mask_name not in available_masks:
            Console.printf("error", f"Mask '{mask_name}' not listed in {configurator.file_name}. Only listed: {available_masks}. Terminating the program!")
            sys.exit()

        # Finally, load the mask according to the path
        mask = (NeuroImage(path=configurator.data["path"]["mask"][mask_name]).
                load_nii())

        Console.printf("success", f"Thus, loaded the '{mask_name}' mask")

        return mask


class MetabolicAtlas:
    # TODO

    def __init__(self):
        # TODO
        pass

    def load(self):
        # TODO
        pass


class T1Image:
    # TODO
    def __init__(self):
        # TODO
        pass

    def load(self):
        # TODO
        pass


class FIDs:
    """
    This class is for creating an FID containing several attributes. The FID signal and parameters can be
    either added from e.g., a simulation or loaded from a MATLAB file (.m). Moreover, it is able to convert
    the Real and Imaginary part in the file given as (Re, Im) -> Re + j*Im.
    """

    def __init__(self, configurator: Configurator, concentrations: np.ndarray = None, t2_values: np.ndarray = None):
        self.configurator = configurator
        self.parameters: dict = {}
        self.fids: list = []

    def load(self, fid_name: str, signal_data_type: np.dtype = np.float64):
        """
        For loading and splitting the FIDs according to the respective chemical compound (metabolites, lipids).
        Then, create on :class: `spectral.FID` for each chemical compound and store it into a list.
        Additional: since the complex signal is represented in two columns (one for real and one for imaginary),
        it has to be transformed to a complex signal.

        :param fid_name: Name of the FID (e.g., 'metabolites', 'lipids')
        :param signal_data_type: Desired data type of the signal signal (numpy data types).
        :return: Nothing. Access the list fids of the object for the signals!
        """
        self.configurator.load()
        path = self.configurator.data["path"]["fid"][fid_name]
        jmrui = JMRUI2(path=path, signal_data_type=signal_data_type)
        data = jmrui.load_m_file()

        parameters, amplitude, time = data["parameters"], data["signal"], data["time"]
        self.parameters = parameters

        amplitude_complex = self.__transform_signal_complex(amplitude, time)
        self.__assign_signal_to_compound(amplitude_complex, time)

    def __transform_signal_complex(self, amplitude: np.ndarray, conjugate: bool = True) -> np.ndarray:
        """
        Transform the values given for each row as [Real, Imaginary] to [Real + j*Imaginary]
        :return: None

        :param conjugate: complex conjugation if True. Default True.
        """
        signal_shape_previous = amplitude.shape
        re = amplitude[:, 0]
        im = amplitude[:, 1]
        amplitude = np.vectorize(complex)(re, im)
        amplitude = np.conjugate(amplitude) if conjugate is True else amplitude

        Console.printf("success",
                       f"Transformed FID signal to complex values: {signal_shape_previous} -> {amplitude.shape}")

        return amplitude

    def __assign_signal_to_compound(self, amplitude: np.ndarray, time: np.ndarray) -> None:
        """
        For splitting the whole signal into the parts corresponding to the respective compound (metabolite or lipid).
        This is given in the .m file which this class ins handling.

        :param amplitude: Overall signal (whole signal given in .m-File)
        :param time: Corresponding time vector
        :return: Nothing
        """

        # Resize and thus split the whole signal into parts corresponding to the respective compound
        signal_reshaped = amplitude.reshape(int(self.parameters['SIZE'][2]), -1)

        # Create for each part of the signal (signal) corresponding to one compound an FID object,
        # the put into a list containing all FID objects
        Console.add_lines("Assigned FID parts:")
        for column_number, name in enumerate(self.parameters['DIM_VALUES'][2]):
            fid = spectral.FID(amplitude=signal_reshaped[column_number],
                               time=time,
                               name=name)
            self.fids.append(fid)

            Console.add_lines(f"{column_number}. {name + ' ':-<30}-> shape: {signal_reshaped[column_number].shape}")

        Console.printf_collected_lines("success")

    def show_parameters(self) -> None:
        """
        Printing the successfully read parameters formatted to the console.

        :return: None
        """
        # np.set_printoptions(20)  # for printing numpy numbers with 20 digits to the console
        for key, value in self.parameters.items():
            Console.add_lines(f"Key: {key}, Value: {value}")

        Console.printf_collected_lines("success")
        # np.set_printoptions() resetting numpy printing options


if __name__ == "__main__":
    configurator = Configurator(path_folder="/home/mschuster/projects/SimulationMRSI/config",
                                file_name="config_04012024.json")

    metabolic_mask = Mask.load(configurator=configurator,
                               mask_name="metabolites")
