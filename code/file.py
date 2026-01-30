## just for type checking, to solve circular imports ##
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sampling import CoilSensitivityVolume
#######################################################


from cupyx.scipy.ndimage import zoom as zoom_gpu
import spectral_spatial_simulation
#from spatial_metabolic_distribution import MetabolicPropertyMap, MetabolicPropertyMaps
from spatial_metabolic_distribution import ParameterMap, ParameterVolume
from typing_extensions import Self
from tools import JsonConverter, UnitTools, JupyterPlotManager, SpaceEstimator, ArrayTools
import matplotlib.pyplot as plt
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

from interfaces import WorkingVolume, Plot

# For enabling to use units
u = pint.UnitRegistry()


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

### DELETE -> Moved to interfaces module
###class WorkingVolume(ABC):
###    """
###    For other classes to be inherited. The need of implementing the working equivalent to the loader class.
###    For example: file.ParameterMaps is only for loading the volume, then another class in another module
###    implements the other functionalities to 'work' with this volume.
###    """
###
###    @abstractmethod
###    def to_working_volume(self):
###        """Defines an abstract (interface) method. Subclasses must implement it to transform their data into
###        a target class in another module."""
###        ...





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

#    def change_data_type(self, data_type, verbose: bool=True):
#        data_type_before = self.values.dtype
#        data_type_after = data_type
#        self.values = self.values.astype(data_type_after)
#        Console.printf("success", f"Changed data type from '{data_type_before}' => '{data_type_after}'", mute=not verbose)


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


class ParameterMaps(WorkingVolume, Plot):
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
            Case 2: To handle nii files: Results NOT in dictionary ->  self.loaded_maps = values
            """
            Console.printf("info", f"Loading nii file for map type {self.map_type_name}")
            # Using NeuroImage to load nii file
            loaded_map = NeuroImage(path=self.main_path).load_nii(mute=True, report_nan=True).data
            # Use map_type_name as the key, or choose an appropriate one
            self.loaded_maps = loaded_map
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


    def to_working_volume(self, data_type: str = None, verbose: bool = True) -> ParameterVolume:
        """
        This transforms the loaded maps directly to a 4D array of (metabolites, X, Y, Z)

        :return: a ParameterVolume (see module spatial_metabolic_distribution)
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
        path = self.configurator.data["fid"][fid_name]["path"]
        jmrui = JMRUI(path=path, signal_data_type=signal_data_type)
        data = jmrui.load_m_file()

        parameters, signal, time = data["parameters"], data["signal"], data["time"]
        self.parameters = parameters

        signal_complex = self.__transform_signal_complex(signal, time)
        self.__assign_signal_to_compound(signal_complex, time)

        # At this point the FID is already created. Just adding the unit.
        unit = self.configurator.data["fid"][fid_name]["unit_time"]
        try:
            self.loaded_fid.unit_time: pint.Unit = u.Unit(unit)
            Console.printf("success", f"Assigned unit to the loaded time vector: {self.loaded_fid.unit_time}")
        except:
            Console.printf("error", f"Could not assign unit to loaded FID: '{unit}'. No valid unit.")

        if ArrayTools.check_nan(self.loaded_fid.signal, verbose=False):
            Console.printf("warning", "Loaded signal of FID contains NaNs.")
        if ArrayTools.check_nan(self.loaded_fid.time, verbose=False):
            Console.printf("warning", "loaded time vector of FID contains NaNs.")

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
        #
        #
        #

### DELETE BELOW:
###class CoilSensitivityMaps:
###    """
###    For loading and interpolating the coil sensitivity maps. At the moment, only HDF5 files are supported.
###    The maps can be interpolated on 'cpu' and desired 'gpu'. It is strongly recommended to use the 'gpu'
###    regarding computational performance.
###    An object of the configurator is necessary to get the path to the coil sensitivity maps.
###    """
###
###    def __init__(self, configurator: Configurator):
###        self.configurator = configurator
###        self.maps = None
###        self.shape = None
###
###    def load_h5py(self, keys: list = ["imag", "real"], data_type=np.complex64) -> None:
###        """
###        For loading coil sensitivity maps of the type HDF5. The default keys are 'real' and 'imag' for loading the complex values.
###        If different keys are defined in the HDF5 it is necessary to explicitely define them with the argument 'keys'.
###
###        :param dtype: desired data type. Since the data is complex ude numpy complex data types.
###        :return: None.
###        """
###
###        # Load the HDF5 file via h5py
###        path_maps = self.configurator.data["maps"]["coil_sensitivity"]["path"]
###        h5py_file = h5py.File(path_maps, "r")
###        h5py_file_keys_required = keys
###
###        # If the default keys or by the user defined keys are not available in the HDF5 file:
###        if not all(key in h5py_file.keys() for key in h5py_file_keys_required):
###            Console.add_lines(f"Could not find keys {h5py_file_keys_required} in the HDF5 file")
###            Console.add_lines(f" => Instead the following keys are available: {h5py_file.keys()}.")
###            Console.add_lines(f" => Aborting the program.")
###            Console.printf_collected_lines("error")
###            sys.exit()
###
###        # If the keys are existing in the HDF5 file, load them and convert to complex values:
###        maps_real = h5py_file["real"][:]
###        maps_imag = h5py_file["imag"][:]
###        maps_complex = maps_real + 1j * maps_imag
###        maps_complex = maps_complex.astype(data_type, copy=False)
###
###        Console.add_lines("Coil Sensitivity Maps:")
###        Console.add_lines(f" => Could find keys {h5py_file_keys_required} in the HDF5 file")
###        Console.add_lines(f" => Loaded and converted maps to {maps_complex.dtype}")
###        Console.add_lines(f" => Data shape: {maps_complex.shape}")
###        Console.add_lines(f" => Space required: {}")
###        Console.printf_collected_lines("success")
###
###        self.maps = maps_complex
###
###
###    def interpolate(self, target_size, order=3, compute_on_device="cpu", gpu_index=0, return_on_device="cpu") -> np.ndarray | cp.ndarray:
###        """
###        To interpolate the loaded coil sensitivity maps with the selected order to a target size on the desired device. Possible is the 'cpu' and 'cuda'.
###        If 'cuda' is selected, with the index the desired device can be chosen. Further, the result can be returned on 'cpu' or 'cuda'.
###
###        :param target_size: target size to interpolate the coil sensitivity maps
###        :param order: desired order (0: Nearest-neighbor, 1: Bilinear interpolation, 2: Quadratic interpolation, 3: Cubic interpolation (default),...). See scipy.ndimage or cupyx.scipy.ndimage for more information.
###        :param compute_on_device: device where the interpolation takes place ('cuda' is recommended)
###        :param gpu_index: if 'cuda' is selected in the param compute_on_device, then possible to choose a gpu if multiple gpus are available. The default is 0.
###        :param return_on_device: After the interpolation, the device where the array should be returned.
###        :return: Interpolated array
###        """
###
###        # Check if maps are already loaded
###        if self.maps is None:
###            Console.printf("error", "No maps data available to interpolate. You might call 'load_h5py' first! Aborting program.")
###            sys.exit()
###
###        self.shape = self.maps.shape
###
###        # Get the zoom factor, based on the current maps size and desired size
###        zoom_factor = np.divide(target_size, self.maps.shape)
###        Console.printf("info", "Start interpolating coil sensitivity maps")
###        Console.add_lines("Interpolated the coil sensitivity maps:")
###        Console.add_lines(f" => From size: {self.maps.shape} --> {target_size}")
###        Console.add_lines(f" => with interpolation order: {order}")
###        Console.add_lines(f" => on device {compute_on_device}{':' + str(gpu_index) if compute_on_device == 'cuda' else ''} --> and returned on {return_on_device}")
###
###        # Interpolate on CUDA or CPU based on the user decision
###        maps_complex_interpolated = None
###        if compute_on_device == "cpu":
###            maps_complex_interpolated = zoom(input=self.maps, zoom=zoom_factor, order=order)
###        elif compute_on_device == "cuda":
###            with cp.cuda.Device(gpu_index):
###                maps_complex_interpolated = zoom_gpu(input=cp.asarray(self.maps), zoom=zoom_factor, order=order)
###        else:
###            Console.printf("error", f"Device {compute_on_device} not supported. Use either 'cpu' or 'cuda'.")
###
###        Console.printf_collected_lines("success")
###
###        # Return on the desired device after interpolation
###        if return_on_device == "cpu" and isinstance(maps_complex_interpolated, cp.ndarray):
###            maps_complex_interpolated = cp.asnumpy(maps_complex_interpolated)
###        elif return_on_device == "cuda" and isinstance(maps_complex_interpolated, np.ndarray):
###            maps_complex_interpolated = cp.asarray(maps_complex_interpolated)
###        elif return_on_device not in ["cpu", "cuda"]:
###            Console.printf("error", f"Return on device {return_on_device} not supported. Choose either 'cpu' or 'cuda'. Abort program.")
###            sys.exit()
###
###        # Save to object variable
###        self.maps = maps_complex_interpolated
###
###        # Just return the interpolated result..
###        return maps_complex_interpolated

class CoilSensitivityMaps(WorkingVolume, Plot):
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

    def to_working_volume(self) -> CoilSensitivityVolume | None:
        """
        Converts the from file.CoilSensitivityMaps to sampling.CoilSensitivityVolume.

        :return: CoilSensitivityVolume in sampling
        """
        from sampling import CoilSensitivityVolume

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


###    def interpolate(self, target_size, order=3, compute_on_device="cpu", gpu_index=0, return_on_device="cpu") -> np.ndarray | cp.ndarray:
###        """
###        To interpolate the loaded coil sensitivity maps with the selected order to a target size on the desired device. Possible is the 'cpu' and 'cuda'.
###        If 'cuda' is selected, with the index the desired device can be chosen. Further, the result can be returned on 'cpu' or 'cuda'.
###
###        :param target_size: target size to interpolate the coil sensitivity maps
###        :param order: desired order (0: Nearest-neighbor, 1: Bilinear interpolation, 2: Quadratic interpolation, 3: Cubic interpolation (default),...). See scipy.ndimage or cupyx.scipy.ndimage for more information.
###        :param compute_on_device: device where the interpolation takes place ('cuda' is recommended)
###        :param gpu_index: if 'cuda' is selected in the param compute_on_device, then possible to choose a gpu if multiple gpus are available. The default is 0.
###        :param return_on_device: After the interpolation, the device where the array should be returned.
###        :return: Interpolated array
###        """
###
###        # Check if maps are already loaded
###        if self.maps is None:
###            Console.printf("error", "No maps data available to interpolate. You might call 'load_h5py' first! Aborting program.")
###            sys.exit()
###
###        self.shape = self.maps.shape
###
###        # Get the zoom factor, based on the current maps size and desired size
###        zoom_factor = np.divide(target_size, self.maps.shape)
###        Console.printf("info", "Start interpolating coil sensitivity maps")
###        Console.add_lines("Interpolated the coil sensitivity maps:")
###        Console.add_lines(f" => From size: {self.maps.shape} --> {target_size}")
###        Console.add_lines(f" => with interpolation order: {order}")
###        Console.add_lines(f" => on device {compute_on_device}{':' + str(gpu_index) if compute_on_device == 'cuda' else ''} --> and returned on {return_on_device}")
###
###        # Interpolate on CUDA or CPU based on the user decision
###        maps_complex_interpolated = None
###        if compute_on_device == "cpu":
###            maps_complex_interpolated = zoom(input=self.maps, zoom=zoom_factor, order=order)
###        elif compute_on_device == "cuda":
###            with cp.cuda.Device(gpu_index):
###                maps_complex_interpolated = zoom_gpu(input=cp.asarray(self.maps), zoom=zoom_factor, order=order)
###        else:
###            Console.printf("error", f"Device {compute_on_device} not supported. Use either 'cpu' or 'cuda'.")
###
###        Console.printf_collected_lines("success")
###
###        # Return on the desired device after interpolation
###        if return_on_device == "cpu" and isinstance(maps_complex_interpolated, cp.ndarray):
###            maps_complex_interpolated = cp.asnumpy(maps_complex_interpolated)
###        elif return_on_device == "cuda" and isinstance(maps_complex_interpolated, np.ndarray):
###            maps_complex_interpolated = cp.asarray(maps_complex_interpolated)
###        elif return_on_device not in ["cpu", "cuda"]:
###            Console.printf("error", f"Return on device {return_on_device} not supported. Choose either 'cpu' or 'cuda'. Abort program.")
###            sys.exit()
###
###        # Save to object variable
###        self.maps = maps_complex_interpolated
###
###        # Just return the interpolated result..
###        return maps_complex_interpolated


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
