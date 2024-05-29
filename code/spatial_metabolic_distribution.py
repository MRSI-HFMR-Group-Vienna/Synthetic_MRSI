from cupyx.scipy.ndimage import zoom as zoom_gpu
from scipy.ndimage import zoom as zoom_cpu
from dataclasses import dataclass, asdict
from dataclasses import dataclass
from tools import CustomArray
from printer import Console
import dask.array as da
from tqdm import tqdm
import numpy as np
import cupy as cp
import pint
import sys

import dask
from dask.delayed import Delayed

import default


class Model:
    # TODO
    def __init__(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def add_mask(self):
        # TODO. Create masks is in the simulator. Right place there?
        raise NotImplementedError("This method is not yet implemented")

    def add_t1_image(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def add_subject_variability(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def add_pathological_alterations(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")


class MetabolicAtlas:
    # TODO
    def __init__(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def transform_to_t1(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def load(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")


class Maps:
    """
    For managing a bunch of metabolic maps. The intention is to manage a bundle based on the category,
    for example, all concentration maps, or all T1 maps, or T2 maps, and so on.
    """

    def __init__(self, maps: dict[str, np.ndarray | np.memmap] = None):
        """
        Either instantiate a Maps object empty or already with maps.

        :param maps: dictionary with maps and name.
        """
        if maps is None:
            self.maps: dict[str, np.ndarray | np.memmap] = {}
        else:
            self.maps: dict[str, np.ndarray | np.memmap] = maps

    def interpolate_to_target_size(self, target_size: tuple, order: int = 3, target_device: str = 'cpu'):
        """
        To interpolate all maps that the Maps object contains to a desired target size. The order of interpolation
        can also be set. For more details see zoom of scipy.ndimage (CPU) or cupyx.scipy.ndimage (CUDA).

        It is further possible to perform the interpolation of CPU or CUDA.

        :param target_size: Interpolation to desired size. Insert dimensions as tuple.
        :param order: Desired interpolation (e.g., bilinear). Thus set according number.
        :param target_device: CPU (cpu) or CUDA (cuda)
        :return: the Maps object
        """

        Console.printf("info", f"Start interpolating metabolic maps on {target_device}")
        Console.add_lines("Interpolate maps: ")

        # Select method based on the desired device
        if target_device == 'cpu':
            zoom = zoom_cpu
        elif target_device == 'cuda':
            zoom = zoom_gpu
        else:
            Console.printf("error", f"Invalid target device: {target_device}. it must be either 'cpu' or 'cuda'.")
            sys.exit()

        # Interpolate each map in tghe dictionary
        for i, (working_name, loaded_map) in tqdm(enumerate(self.maps.items()), total=len(self.maps)):
            # Calculate the zoom factor required by the interpolation method
            zoom_factor = np.divide(target_size, loaded_map.shape)
            # When cuda is selected, convert numpy array to a cupy array
            loaded_map = cp.asarray(loaded_map) if target_device == 'cuda' else loaded_map
            # Interpolate with selected method
            interpolated = zoom(input=loaded_map, zoom=zoom_factor, order=order)
            # When cuda is selected, convert cupy array back to a numpy array and thus transfer to cpu
            self.maps[working_name] = interpolated if target_device == 'cpu' else cp.asnumpy(interpolated)

            Console.add_lines(f"  {(i)}: {working_name:.<10}: {loaded_map.shape} --> {self.maps[working_name].shape}")
        Console.printf_collected_lines("success")

        return self


class MetabolicPropertyMapsAssembler:
    # TODO Docstring

    def __init__(self,
                 block_size: tuple,
                 concentration_maps: Maps,  # TODO change to Maps that is in spectral_spatial_simulation!
                 t1_maps: Maps,
                 t2_maps: Maps,
                 concentration_unit,  # TODO define data type
                 t1_unit,
                 t2_unit):

        self.block_size = block_size

        self.concentration_maps = concentration_maps
        self.t1_maps = t1_maps
        self.t2_maps = t2_maps
        self.concentration_unit = concentration_unit
        self.t1_unit = t1_unit
        self.t2_unit = t2_unit

    def assemble(self):
        # TODO: Create Docstring
        # TODO Create metabolic property map --> i guess i need dictionary --> see main file
        metabolic_property_maps_dict: dict[str, MetabolicPropertyMap] = {}
        for name, _ in self.concentration_maps.maps.items():

            metabolic_property_map = MetabolicPropertyMap(chemical_compound_name=name,
                                                          block_size=self.block_size,
                                                          t1=self.t1_maps.maps[name],
                                                          t1_unit=self.t1_unit,
                                                          t2=self.t2_maps.maps[name],
                                                          t2_unit=self.t1_unit,
                                                          concentration=self.concentration_maps.maps[name],
                                                          concentration_unit=self.concentration_unit)
            metabolic_property_maps_dict[name] = metabolic_property_map

        return metabolic_property_maps_dict


class MetabolicPropertyMap:
    """
    Takes 3D volumes (maps) of the respective metabolite. This includes T1, T2 and concentration so far.
    """

    def __init__(self,
                 chemical_compound_name: str,
                 block_size: tuple,
                 t1: np.ndarray,
                 t1_unit: pint.Unit,
                 t2: np.ndarray,
                 t2_unit: pint.Unit,
                 concentration: np.ndarray,
                 concentration_unit: pint.Unit,
                 t1_metadata: dict = None,
                 t2_metadata: dict = None,
                 concentration_metadata: dict = None):
        self.chemical_compound_name = chemical_compound_name
        self.t1 = CustomArray(dask_array=da.from_array(t1, chunks=block_size),
                              unit=t1_unit,
                              meta=t1_metadata)

        self.t2 = CustomArray(dask_array=da.from_array(t2, chunks=block_size),
                              unit=t2_unit,
                              meta=t2_metadata)

        self.concentration = CustomArray(dask_array=da.from_array(concentration, chunks=block_size),
                                         unit=concentration_unit,
                                         meta=concentration_metadata)

        self.block_size = block_size

    def __str__(self):
        """
        String representation of an object of this class.

        :return: string for printing to the console
        """
        text = (f"MetabolicPropertyMap of : {self.chemical_compound_name} \n"
                f" with block size: {self.block_size} \n"
                f" t1: {self.t1} \n"
                f" t2: {self.t2} \n"
                f" concentration: {self.concentration}")
        return text


class Simulator:
    # TODO

    def __init__(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def transform_metabolic_atlas_to_t1(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def create_masks(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")


if __name__ == "__main__":
    pass
