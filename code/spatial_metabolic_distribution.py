from dataclasses import dataclass, asdict
from tools import SubVolumeDataset
from dataclasses import dataclass
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


# class MetabolicMap:
#    """
#    This includes the data of the metabolic map as numpy.array, thus the concentration at each spatial position.
#    """
#    def __init__(self):
#        self.concentration: np.ndarray = None
#        self.name: str = None
#        self.unit: str = None


# @dataclass
class MetabolicPropertyMap:
    """
    Contains for a chemical compound maps regarding concentration [mmol], t1 [ms] and t2 [ms]. It is possible to multiply two
    objects of this class. Further, units are added automatically.
    TODO: other operations than * necessary?
    """

    def __init__(self,
                 chemical_compound_name: str,
                 concentration: np.ndarray | da.core.Array | SubVolumeDataset,  # for creating subsets the dask Array
                 t2: np.ndarray | da.core.Array | SubVolumeDataset,
                 t1: np.ndarray | da.core.Array | SubVolumeDataset,
                 #main_volume_shape: tuple = None,
                 blocks_shape_xyz: tuple) -> None:

        # For using units
        # u = pint.UnitRegistry() # TODO

        # Shape of the main volume and sub-volumes that will be created afterward
        # self.main_volume_shape = main_volume_shape  # TODO ==> necessary to know main volume if create sub volume! ==> for index!
        self.blocks_shape_xyz = blocks_shape_xyz

        #self.main_volume_shape = main_volume_shape

        # Index of the sub volume
        self.sub_volume_index = None  # TODO

        # Name of the respective chemical compound to which all this maps belong
        self.chemical_compound_name = chemical_compound_name

        # TODO compute/read it!!!!!
        # TODO compute/read it!!!!!
        # TODO compute/read it!!!!!
        # TODO compute/read it!!!!! Do I need it?
        #self.main_volume_shape = -1

        # Check if isinstance of already SubVolumeDataset -> necessary if object1 + object2 in __mul__
        if isinstance(concentration, SubVolumeDataset):
            self.concentration = concentration
        else:
            self.concentration = SubVolumeDataset(main_volume=concentration, blocks_shape_xyz=self.blocks_shape_xyz)  # [mmol] --> TODO: add pint --> also interpolation
        if isinstance(t2, SubVolumeDataset):
            self.t2 = t2
        else:
            self.t2 = SubVolumeDataset(main_volume=t2, blocks_shape_xyz=self.blocks_shape_xyz)  # [u.ms] --> TODO: add pint --> also interpolation
        if isinstance(t1, SubVolumeDataset):
            self.t1 = t1
        else:
            self.t1 = SubVolumeDataset(main_volume=t1, blocks_shape_xyz=self.blocks_shape_xyz)  # [u.ms] --> TODO: add pint --> also interpolation

        # Iterator for accessing sub-volumes (so-called blocks) of the main volume
        self.blocks_iterator: int = 0  # initial value, start at block 0

        # Get the number of sub-volumes created (=blocks). Has to be the same number for each SubVolumeDataset in the MetabolicPropertyMap.
        self.number_of_partial_volumes = self.concentration.number_of_blocks

    #    def get_partial_map(self, indices: tuple) -> None:
    #        """
    #        To get only a sub-volumes of the main volumes
    #
    #        :param indices: x,y,z indices of the sub-volume in the main volume!
    #        :return: Nothing
    #        """
    #        self.concentration.get_by_index(*indices)
    #        self.t2.get_by_index(*indices)
    #        self.t1.get_by_index(*indices)
    #
    #        # TODO: either return the real indices or calculate it!
    #        # TODO
    #        # TODO
    #        # TODO
    #        # TODO
    #        #concentration_partly_volume = self.concentration.blocks_xyz
    #        #t2_partly_volume = SubVolumeDataset(volume=self.t2, blocks_xyz=self.blocks_xyz).volume
    #        #t1_partly_volume = SubVolumeDataset(volume=self.t1, blocks_xyz=self.blocks_xyz).volume
    #        #return MetabolicPropertyMap(chemical_compound_name=self.chemical_compound_name,
    #        #                            concentration=concentration_partly_volume,
    #        #                            t2=t2_partly_volume,
    #        #                            t1=t1_partly_volume,
    #        #                            main_volume_shape=self.main_volume_shape,
    #        #                            blocks_xyz=self.blocks_xyz)

    ##@property
    ##def concentration(self):
    ##    """
    ##    Getter method. Returning the block(s) as dask array in a list. Possible to compute(), which yields a numpy.array.
    ##    """
    ##    return self.concentration.blocks
    ##
    ##@property
    ##def t2(self):
    ##    """
    ##    Getter method. Returning the block(s) as dask arrays in a list. Possible to compute(), which yields a numpy.array.
    ##    """
    ##    return self.t2.blocks
    ##
    ##@property
    ##def t1(self):
    ##    """
    ##    Getter method. Returning the block(s) dask arrays in a list. Possible to compute(), which yields a numpy.array.
    ##    """
    ##    return self.t1.blocks

    def __getitem__(self, blocks_number):
        """
        Blocks are cut out of the main volume (=forming the sub volume aka partial volume).
        """
        concentration_blocks = self.concentration[blocks_number]  # just partially volume (=sub volume) of main volume
        t2_blocks = self.t2[blocks_number]  # just partially volume (=sub volume) of main volume
        t1_blocks = self.t1[blocks_number]  # just partially volume (=sub volume) of main volume
        concentration_partial_volume, t2_partial_volume, t1_partial_volume = concentration_blocks, t2_blocks, t1_blocks

        return MetabolicPropertyMap(chemical_compound_name=self.chemical_compound_name,
                                    concentration=concentration_partial_volume,
                                    t2=t2_partial_volume,
                                    t1=t1_partial_volume,
                                    blocks_shape_xyz=self.blocks_shape_xyz
                                    #main_volume_shape=self.main_volume_shape
                                    )

    def __iter__(self):
        """
        Return the object itself as the iterator
        """
        return self

    def __next__(self):
        """
        To implement it in a for each loop, for example, or to use next() on the object! Note: note the object itself is
        returned, but a delayed object 'dask.delayed.Delayed'. Thus, it needs to be computed first!
        """
        if self.blocks_iterator >= self.number_of_partial_volumes:
            raise StopIteration  # Stop iterator when the last block was reached!
        else:
            self.blocks_iterator += 1
            concentration_blocks = self.concentration[self.blocks_iterator - 1]  # just partially volume (=sub volume) of main volume
            t2_blocks = self.t2[self.blocks_iterator - 1]  # just partially volume (=sub volume) of main volume
            t1_blocks = self.t1[self.blocks_iterator - 1]  # just partially volume (=sub volume) of main volume

            # metabolic_property_map = MetabolicPropertyMap(chemical_compound_name=self.chemical_compound_name,
            #                            concentration=concentration_blocks,
            #                            t2=t2_blocks,
            #                            t1=t1_blocks,
            #                            blocks_xyz=self.blocks_xyz)

            metabolic_property_map = MetabolicPropertyMap(chemical_compound_name=self.chemical_compound_name,
                                                          concentration=concentration_blocks,
                                                          t2=t2_blocks,
                                                          t1=t1_blocks,
                                                          blocks_shape_xyz=self.blocks_shape_xyz,
                                                          #main_volume_shape=self.main_volume_shape
                                                          )

            return metabolic_property_map

    def __mul__(self, other):
        """
        Implements to multiply objects of these class by simply apply: object1 * object2. This returns a new
        object containing the product of all maps, and the name is a string of "compound 1 + compound 2"
        """
        # TODO: handle mismatching shapes?
        ##if self.main_volume_shape != other.main_volume_shape:
        ##    Console.printf("error", f"Could not create product of the two objects. "
        ##                            f"Shape of object 1 is '{self.main_volume_shape}' "
        ##                            f"while for object 2 it is '{other.main_volume_shape}' !")
        ##    sys.exit()

        chemical_compound_name: str = f"{self.chemical_compound_name} * {other.chemical_compound_name}"
        # The following attributes are transformed from a numpy ndarray to pint.Quantity, which contains
        # a numpy ndarray and the unit.
        #  -> get only numpy ndarray with variable.magnitude
        #  -> get only unit via variable.units
        concentration = self.concentration * other.concentration
        t2 = self.t2 * other.t2
        t1 = self.t1 * other.t1

        metabolic_property_map = MetabolicPropertyMap(chemical_compound_name=chemical_compound_name,
                                                      concentration=concentration,
                                                      t2=t2,
                                                      t1=t1,
                                                      blocks_shape_xyz=self.blocks_shape_xyz
                                                      #main_volume_shape=self.main_volume_shape
                                                      )

        return metabolic_property_map

    def __len__(self):
        # The other volumes need to have the same number of blocks!
        return self.number_of_partial_volumes

    # def __iter__(self):
    #    """
    #    Implements to be able to iterate through all attributes.
    #    """
    #    obj_dict = asdict(self)
    #    return iter(obj_dict.items())

    #def get_combined_maps(self, chunk_number: int = 0):
    #    """
    #    Multiply all maps. So far considered: concentration, T1, T2. By default, the first chunk is selected. If
    #    over an object of the MetabolicPropertyMap is iterated, one chunk per iteration is given. However, it is
    #    still in a list; thus by default, the chunk number is set to 0 to easy access the product of all chunks.
    #    """
    #    return self.concentration[chunk_number] * self.t2[chunk_number] * self.t1[chunk_number]



#
## TODO: Also return only subpart! --> Synchronise with FIDs?
##       ---> use from sampling.SubVolumeDataset
##           ---> need index of subvolume, and each voxel in it --> calculate it instead of storing it!
##                                                                     --> index pixel(dimension) = main volume index(dimension) * sub volume index(dimension) * size of sub volume(dimension)


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
