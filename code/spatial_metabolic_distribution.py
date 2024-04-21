from dataclasses import dataclass, asdict
from dataclasses import dataclass
from tools import CustomArray
from printer import Console
import dask.array as da
import numpy as np
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
