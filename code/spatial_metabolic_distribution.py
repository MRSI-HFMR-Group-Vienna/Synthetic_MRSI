from dataclasses import dataclass, asdict
import spectral_spatial_simulation
from dataclasses import dataclass
from tools import CustomArray
from printer import Console
import dask.array as da
import numpy as np
import file
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


class MetabolicPropertyMapsAssembler:
    # TODO Docstring

    def __init__(self,
                 fid: spectral_spatial_simulation.FID,
                 concentration_maps: file.Maps, # TODO change to Maps that is in spectral_spatial_simulation!
                 T1_maps: file.Maps,
                 T2_maps: file.Maps,
                 concentration_unit, # TODO define data type
                 T1_unit,
                 T2_unit):

        self.fid = fid
        self.concentration_maps = concentration_maps
        self.T1_maps = T1_maps
        self.T2_maps = T2_maps
        self.concentration_uni = concentration_unit
        self.T1_unit = T1_unit
        self.T2_unit = T2_unit

    def assemble(self):
        # TODO cerate metabolic property map --> i guess i need dictionary --> see main file
        for fid_one_signal in self.fid:
            print(fid_one_signal)

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
