import default
from cupyx.scipy.ndimage import zoom as zoom_gpu
import spectral_spatial_simulation
from typing_extensions import Self
from tools import JsonConverter
from scipy.ndimage import zoom
from tools import deprecated
from scipy.io import loadmat
from printer import Console
from pathlib import Path
import nibabel as nib
import numpy as np
import cupy as cp
import pint
import h5py
import json
import sys
import os


class JMRUI:
    """
    For reading the data from an m.-File generated from an jMRUI. TODO: Implement for the .mat file. Further, also rename JMRUI instead of JMRUI!
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

    def load_nii(self, mute: bool = False):
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
                                  f"\n    Shape             -> {self.shape}"
                                  f"\n    Pixel dimensions: -> {self.header.get_zooms()}"
                                  f"\n    Values range:     -> [{round(np.min(self.data), 3)}, {round(np.max(self.data), 3)}]"
                                  f"\n    In memory cache?  -> {self.nifti_object.in_memory}",  # TODO what was exactly the purpose?
                       mute=mute)

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


class Maps:
    """
    This class can be used at the moment for carious purposes:

    1) To load data:
        -> from one nii file, yields dictionary of just values
        -> from one h5 file, yields dictionary (e.g., with keys 'imag', 'real')
        -> from multiple nii files in a respective dictionary (for metabolites)

    2) To interpolate data
        -> one or dictionary of nii
        -> one h5 with multiple items (e.g., 'imag', 'real)

    The class automatically figures out the filetype, at the moment based on '.nii', '.nii.gz', '.h5', '.hdf5'.
    """
    # TODO: Maybe program more flexible

    def __init__(self, configurator: Configurator, map_type_name: str):
        self.configurator = configurator
        self.map_type_name = map_type_name  # e.g. B0, B1, metabolites
        self.loaded_maps: dict[str, np.memmap | h5py._hl.dataset.Dataset] | np.memmap | np.ndarray = None
        self.file_type_allowed = ['.nii', '.nii.gz', '.h5', '.hdf5']



        self.main_path = Path(self.configurator.data["path"]["maps"][self.map_type_name])
        if self.main_path.is_file():
            # Case 1: the main_path points to just one file
            Console.printf("info", "Maps object: The provided path points to a file")
            self.file_type = self.main_path.suffix

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
            Console.printf("info", "Maps object: The provided does not exist or is neither a file nor a folder. Exiting system.")
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
        self.loaded_maps = {}  # key is abbreviation like "Glu" for better matching

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
            Case 2: TO handle nii files: Results NOT in dictionary ->  self.loaded_maps = values
            """
            Console.printf("info", f"Loading nii file for map type {self.map_type_name}")
            # Using NeuroImage to load nii file
            loaded_map = NeuroImage(path=self.main_path).load_nii(mute=True).data
            # Use map_type_name as the key, or choose an appropriate one
            self.loaded_maps = loaded_map
            Console.add_lines(
                f"Loaded nii map: {os.path.basename(self.main_path)} | Shape: {loaded_map.shape} | "
                f"Values range: [{round(np.min(loaded_map), 3)}, {round(np.max(loaded_map), 3)}] | "
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
        self.configurator.load()
        main_path = self.configurator.data["path"]["maps"][self.map_type_name]

        self.loaded_maps: dict = {} # dict required for the operations afterwards

        Console.add_lines("Loaded maps: ")
        for i, (working_name, file_name) in enumerate(working_name_and_file_name.items()):
            path_to_map = os.path.join(main_path, file_name)
            loaded_map = NeuroImage(path=path_to_map).load_nii(mute=True).data
            self.loaded_maps[working_name] = loaded_map

            Console.add_lines(
                f"  {i}: working name: {working_name:.>10} | {'Shape:':<8} {loaded_map.shape} | "
                f"Values range: [{round(np.min(loaded_map), 3)}, {round(np.max(loaded_map), 3)}] "
            )
        Console.printf_collected_lines("success")

        return self

    def interpolate_to_target_size(self, target_size: tuple, order: int = 3) -> Self:
        """
        To interpolate one or multiple maps to the desired target size. Check for more information direct this
        method and the comments.

        :param target_size: desired size as tuple
        :param order: interpolation order
        :return: the object of the whole class
        """

        if isinstance(self.loaded_maps, dict):
            """
            Case 1: Multiple maps are handled in the Maps class. Thus a dictionary with keys is used.
                    This can be for example either due to multiple nii maps (Glucose , Choline, ...),
                    but also if o h5 file has "real" and "imag" for example.
            """
            Console.add_lines("Interpolate loaded maps: ")
            for i, (working_name, loaded_map) in enumerate(self.loaded_maps.items()):
                zoom_factor = np.divide(target_size, loaded_map.shape)
                self.loaded_maps[working_name] = zoom(input=loaded_map, zoom=zoom_factor, order=order)
                Console.add_lines(f"  {i}: {working_name:.<10}: {loaded_map.shape} --> {self.loaded_maps[working_name].shape}")
            Console.printf_collected_lines("success")
        else:
            """
            Case 2: Only one map is handled in the Maps class. This can be for example one nii file. 
                    In this case, no dictionary is used.
            """
            initial_shape = self.loaded_maps.shape
            zoom_factor = np.divide(target_size, self.loaded_maps.shape)
            self.loaded_maps = zoom(input=self.loaded_maps, zoom=zoom_factor, order=order)
            Console.printf("success", f"Only one map exits: Interpolated map: {initial_shape} --> {self.loaded_maps.shape}")
        return self

###class Maps:
###    # TODO: Maybe program more flexible
###
###    def __init__(self, configurator: Configurator, map_type_name: str, file_type: str = 'nii'):
###        self.configurator = configurator
###        self.map_type_name = map_type_name  # e.g. B0, B1, metabolites
###        self.loaded_maps: dict[
###            str, np.memmap | h5py._hl.dataset.Dataset] = {}  # key is abbreviation like "Glu" for better matching, h5py._hl.dataset.Dataset behaves like a memmap? and returns a numpy array if specific value accessed
###        self.file_type_allowed = ['nii', 'h5']
###
###        if file_type not in self.file_type_allowed:
###            Console.printf("error", f"Only possible to load formats: {self.file_type_allowed}. But it was given: {file_type}")
###            sys.exit()
###
###    def load_file(self):
###        """
###        To load a single map from file. At the moment only h5 is supported. TODO: Implement also nii!
###
###        :return:
###        """
###
###        Console.printf("warning", "Maps.load_file ==> by standard h5 is loaded. No nii support yet!")
###
###        self.configurator.load()
###        main_path = self.configurator.data["path"]["maps"][self.map_type_name]
###
###        Console.add_lines(f"Loaded h5py {self.map_type_name} map named: {os.path.basename(main_path)}")
###        with h5py.File(main_path, "r") as file:
###            for key in file.keys():
###                item = file[key]
###
###                if isinstance(item, h5py.Dataset):
###                    Console.add_lines(f" => Key: {key:<5} | Type: {'Dataset':<10} | Shape: {item.shape} | Data Type: {item.dtype}")
###                elif isinstance(item, h5py.Group):
###                    Console.add_lines(f" => Key: {key:<5} | Type: {'Group':<10}")
###
###                Console.add_lines("     Attributes: ")
###                if len(item.attrs.items()) == 0:
###                    Console.add_lines(f"      Not available")
###                else:
###                    for attr_key, attr_val in item.attrs.items():
###                        Console.add_lines(f"        Attribute key: {attr_key:<10} | value: {attr_val:<20}")
###
###                self.loaded_maps[key] = item
###
###            Console.printf_collected_lines("success")
###
###    def load_files_from_folder(self, working_name_and_file_name: dict[str, str]):
###        # TODO
###
###        self.loaded_maps = {}
###
###        Console.printf("warning", "Maps.load_files_from_folder ==> by standard nii is loaded. No h5 support yet!")
###        self.configurator.load()
###        main_path = self.configurator.data["path"]["maps"][self.map_type_name]
###
###        Console.add_lines("Loaded maps: ")
###        for i, (working_name, file_name) in enumerate(working_name_and_file_name.items()):
###            path_to_map = os.path.join(main_path, file_name)
###            loaded_map = NeuroImage(path=path_to_map).load_nii(mute=True).data
###            self.loaded_maps[working_name] = loaded_map
###
###            Console.add_lines(
###                f"  {(i)}: working name: {working_name:.>10} | {'Shape:':<8} {loaded_map.shape} | Values range: [{round(np.min(loaded_map), 3)}, {round(np.max(loaded_map), 3)}] ")
###        Console.printf_collected_lines("success")
###
###        return self
###
###    def interpolate_to_target_size(self, target_size: tuple, order: int = 3):
###        # TODO Docstring
###        # TODO tqdm/progressbar for interpolation!
###
###        Console.add_lines("Interpolate loaded maps: ")
###        for i, (working_name, loaded_map) in enumerate(self.loaded_maps.items()):
###            zoom_factor = np.divide(target_size, loaded_map.shape)
###            self.loaded_maps[working_name] = zoom(input=loaded_map, zoom=zoom_factor, order=order)
###            Console.add_lines(f"  {(i)}: {working_name:.<10}: {loaded_map.shape} --> {self.loaded_maps[working_name].shape}")
###        Console.printf_collected_lines("success")
###
###        return self


class FID:
    """
    This class is for creating an FID containing several attributes. The FID signal and parameters can be
    either added from e.g., a simulation or loaded from a MATLAB file (.m). Moreover, it is able to convert
    the Real and Imaginary part in the file given as (Re, Im) -> Re + j*Im.
    """

    def __init__(self, configurator: Configurator):
        self.configurator = configurator
        self.parameters: dict = {}
        self.loaded_fid: spectral_spatial_simulation.FID = spectral_spatial_simulation.FID()

    def load(self, fid_name: str, signal_data_type: np.dtype = np.float64):  # TODO: replace fid_name to fid_type_name??? --> see Maps class above
        """
        For loading and splitting the FID according to the respective chemical compound (metabolites, lipids).
        Then, create on :class: `spectral_spatial_simulation.FID` for each chemical compound and store it into a list.
        Additional: since the complex signal is represented in two columns (one for real and one for imaginary),
        it has to be transformed to a complex signal.

        :param fid_name: Name of the FID (e.g., 'metabolites', 'lipids')
        :param signal_data_type: Desired data type of the signal signal (numpy data types).
        :return: Nothing. Access the list loaded_fid of the object for the signals!
        """
        self.configurator.load()
        path = self.configurator.data["path"]["fid"][fid_name]
        jmrui = JMRUI(path=path, signal_data_type=signal_data_type)
        data = jmrui.load_m_file()

        parameters, signal, time = data["parameters"], data["signal"], data["time"]
        self.parameters = parameters

        signal_complex = self.__transform_signal_complex(signal, time)
        self.__assign_signal_to_compound(signal_complex, time)

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

        # Merge the signals into one FID, thus also the names
        for column_number, name in enumerate(self.parameters['DIM_VALUES'][2]):
            self.loaded_fid += spectral_spatial_simulation.FID(signal=signal_reshaped[column_number], time=time, name=[name])
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


class CoilSensitivityMaps:
    """
    For loading and interpolating the coil sensitivity maps. At the moment, only HDF5 files are supported.
    The maps can be interpolated on 'cpu' and desired 'gpu'. It is strongly recommended to use the 'gpu'
    regarding computational performance.
    An object of the configurator is necessary to get the path to the coil sensitivity maps.
    """

    def __init__(self, configurator: Configurator):
        self.configurator = configurator
        self.maps = None

    def load_h5py(self, keys: list = ["imag", "real"], dtype=np.complex64) -> None:
        """
        For loading coil sensitivity maps of the type HDF5. The default keys are 'real' and 'imag' for loading the complex values.
        If different keys are defined in the HDF5 it is necessary to explicitely define them with the argument 'keys'.

        :param dtype: desired data type. Since the data is complex ude numpy complex data types.
        :return: None.
        """

        # Load the HDF5 file via h5py
        path_maps = self.configurator.data["path"]["maps"]["coil_sensitivity"]
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
        maps_complex = maps_complex.astype(np.complex64)

        Console.add_lines("Coil Sensitivity Maps:")
        Console.add_lines(f" => Could find keys {h5py_file_keys_required} in the HDF5 file")
        Console.add_lines(f" => Loaded and converted maps to {maps_complex.dtype}")
        Console.printf_collected_lines("success")

        self.maps = maps_complex

    def interpolate(self, target_size, order=3, compute_on_device="cpu", gpu_index=0, return_on_device="cpu") -> np.ndarray | cp.ndarray:
        """
        To interpolate the loaded coil sensitivity maps with the selected order to a target size on the desired device. Possible is the 'cpu' and 'cuda'.
        If 'cuda' is selected, with the index the desired device can be chosen. Further, the result can be returned on 'cpu' or 'cuda'.

        :param target_size: target size to interpolate the coil sensitivity maps
        :param order: desired order (0: Nearest-neighbor, 1: Bilinear interpolation, 2: Quadratic interpolation, 3: Cubic interpolation (default),...). See scipy.ndimage or cupyx.scipy.ndimage for more information.
        :param compute_on_device: device where the interpolation takes place ('cuda' is recommended)
        :param gpu_index: if 'cuda' is selected in the param compute_on_device, then possible to choose a gpu if multiple gpus are available. The default is 0.
        :param return_on_device: After the interpolation, the device where the array should be returned.
        :return: Interpolated array
        """

        # Check if maps are already loaded
        if self.maps is None:
            Console.printf("error", "No maps data available to interpolate. You might call 'load_h5py' first! Aborting program.")
            sys.exit()

        # Get the zoom factor, based on the current maps size and desired size
        zoom_factor = np.divide(target_size, self.maps.shape)
        Console.printf("info", "Start interpolating coil sensitivity maps")
        Console.add_lines("Interpolated the coil sensitivity maps:")
        Console.add_lines(f" => From size: {self.maps.shape} --> {target_size}")
        Console.add_lines(f" => with interpolation order: {order}")
        Console.add_lines(f" => on device {compute_on_device}{':' + str(gpu_index) if compute_on_device == 'cuda' else ''} --> and returned on {return_on_device}")

        # Interpolate on CUDA or CPU based on the user decision
        maps_complex_interpolated = None
        if compute_on_device == "cpu":
            maps_complex_interpolated = zoom(input=self.maps, zoom=zoom_factor, order=order)
        elif compute_on_device == "cuda":
            with cp.cuda.Device(gpu_index):
                maps_complex_interpolated = zoom_gpu(input=cp.asarray(self.maps), zoom=zoom_factor, order=order)
        else:
            Console.printf("error", f"Device {compute_on_device} not supported. Use either 'cpu' or 'cuda'.")

        Console.printf_collected_lines("success")

        # Return on the desired device after interpolation
        if return_on_device == "cpu" and isinstance(maps_complex_interpolated, cp.ndarray):
            maps_complex_interpolated = cp.asnumpy(maps_complex_interpolated)
        elif return_on_device == "cuda" and isinstance(maps_complex_interpolated, np.ndarray):
            maps_complex_interpolated = cp.asarray(maps_complex_interpolated)
        elif return_on_device not in ["cpu", "cuda"]:
            Console.printf("error", f"Return on device {return_on_device} not supported. Choose either 'cpu' or 'cuda'. Abort program.")
            sys.exit()

        # Save to object variable
        self.maps = maps_complex_interpolated

        # Just return the interpolated result..
        return maps_complex_interpolated


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
