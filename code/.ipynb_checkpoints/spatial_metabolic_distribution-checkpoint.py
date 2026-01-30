import time
#
from cupyx.scipy.ndimage import zoom as zoom_gpu
#
from scipy.ndimage import zoom as zoom_cpu
from tools import CustomArray, GPUTools, SortingTools, DaskTools
from dataclasses import dataclass
from printer import Console
import dask.array as da
from tqdm import tqdm
import numpy as np
import cupy as cp
import pint
import sys


class Model:
    # TODO
    def __init__(self):
        # TODO
        raise NotImplementedError("This method is not yet implemented")

    def add_mask(self):
        # TODO. Create masks in the simulator. Right place there?
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


# TODO: Maybe create small Map object
#  -> field name
#  -> data
#  -> unit
#
@dataclass(frozen=True)
class MetabolicPropertyMap:
    name: str
    data: str
    unit: pint.Unit

### ==> von file.maps ---> bekomme dict mit keys() als metabolites und values als maps ===> jedes hat (name, data, unit)
### [ ] TODO: generiere: in file.maps etwas das enthält: {keys, values, units}

class MetabolicPropertyMaps:

    def __init__(self):
        name: list[str] = None,
        maps: list[MetabolicPropertyMap] = None

    def __iter__(self):
        # to return one?
        pass

    def __next__(self):
        pass

    def __add__(self, other):
        pass

    def plot(self):
        #cross-sectional plot of center of each metabolite?
        pass


#   See idea below...:
#@dataclass(frozen=True)
#class Map:
#    name: str
#    data: numpy array
#    unit: pint unit = None
#
#   and define get
#    def __init__(self, maps: Optional[dict[str, ArrayLike]] = None):
#        self._maps: dict[str, Map] = {}
#        if maps:
#            for name, data in maps.items():
#                self._maps[name] = Map(name=name, data=data, unit=None)
#
#    # For backward compatibility -> dict-like behavior (Mapping):
#    def __getitem__(self, name: str) -> ArrayLike:
#        return self._maps[name].data
#
#    def __iter__(self) -> Iterator[str]:
#        return iter(self._maps)
#
#    def __len__(self) -> int:
#        return len(self._maps)


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


    def interpolate_to_target_size(self, target_size: tuple, order: int = 3, target_device: str = 'cpu', target_gpu: int = 0):
        """
        To interpolate all maps that the Maps object contains to a desired target size. The order of interpolation
        can also be set. For more details see zoom of scipy.ndimage (CPU) or cupyx.scipy.ndimage (CUDA).

        It is further possible to perform the interpolation of CPU or CUDA.

        :param target_size: Interpolation to desired size. Insert dimensions as tuple.
        :param order: Desired interpolation (e.g., bilinear). Thus set according number.
        :param target_device: CPU (cpu) or CUDA (cuda)
        :param target_gpu: Desired GPU device
        :return: the Maps object
        """

        Console.printf("info", f"Start interpolating metabolic maps on {target_device}")
        Console.add_lines("Interpolate maps: ")

        # Select method based on the desired device
        if target_device == 'cpu':
            zoom = zoom_cpu
        elif target_device == 'cuda':
            zoom = zoom_gpu
            Console.printf("info", f"Selected GPU: {target_gpu}")
        else:
            Console.printf("error", f"Invalid target device: {target_device}. it must be either 'cpu' or 'cuda'.")
            sys.exit()

        # Interpolate each map in the dictionary
        for i, (working_name, loaded_map) in tqdm(enumerate(self.maps.items()), total=len(self.maps)):
            # Calculate the zoom factor required by the interpolation method
            zoom_factor = np.divide(target_size, loaded_map.shape)
            # When cuda is selected, convert numpy array to a cupy array

            if target_device == 'cuda':
                # Compute on desired GPU, and then bring back to CPU
                interpolated = GPUTools.run_cupy_local_pool(working_function=lambda: zoom(input=cp.asarray(loaded_map), zoom=zoom_factor, order=order), device=target_gpu)
            else:
                # Interpolate with selected method
                interpolated = zoom(input=loaded_map, zoom=zoom_factor, order=order)


            self.maps[working_name] = interpolated


            Console.add_lines(f"  {(i)}: {working_name:.<10}: {loaded_map.shape} --> {self.maps[working_name].shape}")
        Console.printf_collected_lines("success")

        return self


class MetabolicPropertyMapsAssembler:
    """
    This class handles Maps objects. Each Maps object contains one map for each Metabolite.

    The functionality can also be seen als re-sort. The Maps object contains one type of map for each metabolite, e.g.
    all concentration maps of each metabolite, and then will be re-sorted that one MetabolicPropertyMap contains all
    Maps only associated to this one metabolite.

    Example transformation (multiple Maps are required):

    Concentration: [Glucose, NAA, Cholin] ==> Glucose: [Concentration, T1, T2]  ==> dict[Glucose]: MetabolicPropertyMap

    """

    def __init__(self,
                 #block_size: tuple,
                 concentration_maps: Maps,
                 t1_maps:            Maps,
                 t2_maps:            Maps,
                 concentration_unit,       # TODO define data type
                 t1_unit,
                 t2_unit):

        #self.block_size = block_size

        self.concentration_maps = concentration_maps
        self.t1_maps = t1_maps
        self.t2_maps = t2_maps
        self.concentration_unit = concentration_unit
        self.t1_unit = t1_unit
        self.t2_unit = t2_unit


    def assemble_volume(self, block_size:tuple[int, ...]|str = "optimised") -> (list[str, ...], dict[da.Array, ...]):
        """
        This assembles the metabolic maps in a way that for each category one 4D array is created for an accelerated
        computation via dask. So the output shape will be (Metabolites, X, Y, Z). However, a dict is used for a more
        interpretable output: dict(map type, (Metabolites, X, Y, Z), where map type is for example concentration, t1, t2.
        Additionally, the metabolite order of each 4D array of the dictionary is given additionally as output. Otherwise,
        this information would probably be lost.

        :return: a dictionary containing the metabolites order in the 4D array, and another dictionary of the 4D arrays
        of each metabolic map type (e.g., concentration, t1, t2)
        """

        # 1) Assemble a dict of all types of maps
        unsorted_metabolic_maps = {
            "concentration": self.concentration_maps.maps,
            "t1": self.t1_maps.maps,
            "t2": self.t2_maps.maps}

        # 2) Ensure same order inside all dictionaries for later interpretability (+SortingTools checks also if all have same keys)
        sorted_metabolic_maps = SortingTools.sort_dict_reference(*unsorted_metabolic_maps.values())

        # 3) Obtain order of the metabolite axis M in the 4D array (M, X, Y, Z)
        metabolites_order = list(self.concentration_maps.maps.keys())

        # 4) Create a dict with metabolic map types (e.g., concentration, t1, t2) and stack them via dask
        all_metabolic_maps_dict: dict = {}
        for i, metabolic_maps in enumerate(sorted_metabolic_maps):
            one_metabolic_map_type = [] # all metabolites of one type
            for _, metabolic_map in metabolic_maps.items():
                one_metabolic_map_type.append(metabolic_map)

            # 5) Stack the metabolites to one 4D array and re-chunk it so that metabolites axis represents one chunk (for reducing worker to worker transfer)
            one_metabolic_type_dask = da.stack(one_metabolic_map_type, axis=0)

            if block_size == "optimised":
                block_size = (len(metabolites_order),*self.block_size)

            one_metabolic_type_dask = DaskTools.rechunk(one_metabolic_type_dask, chunksize=block_size)
            #DaskTools.rechunk(one_metabolic_type_dask, chunksize=(len(metabolites_order), *self.block_size)) # TODO: maybe here more flexibility regarding the chunksize although maybe slower?
            #DaskTools.rechunk(one_metabolic_type_dask, chunksize=(1,*self.block_size))  # TODO: maybe here more flexibility regarding the chunksize although maybe slower?

            # 6) To create a dictionary of the metabolic map types (concentration, t1, t2, ...) of the 4D arrays
            all_metabolic_maps_dict[list(unsorted_metabolic_maps.keys())[i]] = one_metabolic_type_dask #da.stack(one_metabolic_map_type, axis=0) #stack to (metabolite, X, Y, Z) for each dict entry

        return {"metabolites_order": metabolites_order, "metabolic_maps_volumes": all_metabolic_maps_dict}


    def assemble_dict(self, block_size:tuple[int, ...]|str = "auto") -> dict:
        """
        To take maps, each of one type (T1, T2, concentration) with a volume for each metabolite, and create
        MetabolicPropertyMaps, each for one metabolite containing all types (T1, T2, concentration). Finally
        creates dict with key as name of metabolite and value the corresponding MetabolicPropertyMap.

        :return: Dictionary of MetabolicPropertyMaps. One for each metabolite, containing associated volumes.
        """

        if block_size == "optimised":
            Console.printf("error", "Block size 'optimised' not available for the dictionary version.")
            sys.exit()

        # The dict object will contain:
        #    key:    name of the chemical compound
        #    value:  maps (in MetabolicPropertyMap)
        metabolic_property_maps_dict: dict[str, MetabolicPropertyMap] = {}

        # Assemble dictionary with name (chemical compound / metabolite name) and corresponding MetabolicPropertyMap
        for name, _ in self.concentration_maps.maps.items():

            metabolic_property_map = MetabolicPropertyMap(chemical_compound_name=name,
                                                          block_size=block_size,
                                                          t1=self.t1_maps.maps[name],
                                                          t1_unit=self.t1_unit,
                                                          t2=self.t2_maps.maps[name],
                                                          t2_unit=self.t1_unit,
                                                          concentration=self.concentration_maps.maps[name],
                                                          concentration_unit=self.concentration_unit)
            metabolic_property_maps_dict[name] = metabolic_property_map

        return metabolic_property_maps_dict


    class MetabolicPropertyMaps:

        def __init__(self):
            # The name of the metabolite associated with this map
            pass


        def __add__(self):
            # To add of type Maps
            pass

###class MetabolicPropertyMap:
###    """
###    This is class is to pack together the different maps (so far T1, T2, concentration) of one metabolite (e.g. Glucose).
###    (!) It also transforms the numpy maps to a dask-array extension called CustomArray and further defines the block size for computations.
###
###    The MetabolicPropertyMap It is mainly used in the class MetabolicPropertyMapsAssembler.
###    """
###    def __init__(self,
###                 chemical_compound_name: str,
###                 block_size: tuple,
###                 t1: np.ndarray,
###                 t1_unit: pint.Unit,
###                 t2: np.ndarray,
###                 t2_unit: pint.Unit,
###                 concentration: np.ndarray,
###                 concentration_unit: pint.Unit,
###                 t1_metadata: dict = None,
###                 t2_metadata: dict = None,
###                 concentration_metadata: dict = None):
###
###        # The name of the metabolite of one object
###        self.chemical_compound_name = chemical_compound_name
###
###        # The associates Maps with the metabolite:
###        self.t1 = CustomArray(dask_array=da.from_array(t1, chunks=block_size),
###                              unit=t1_unit,
###                              meta=t1_metadata)
###
###        self.t2 = CustomArray(dask_array=da.from_array(t2, chunks=block_size),
###                              unit=t2_unit,
###                              meta=t2_metadata)
###
###        self.concentration = CustomArray(dask_array=da.from_array(concentration, chunks=block_size),
###                                         unit=concentration_unit,
###                                         meta=concentration_metadata)
###
###        self.block_size = block_size

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
