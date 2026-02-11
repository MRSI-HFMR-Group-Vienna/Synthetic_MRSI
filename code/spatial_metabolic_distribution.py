import time

from tools import JupyterPlotManager
from tools import CustomArray, GPUTools, SortingTools, DaskTools, InterpolationTools
from cupyx.scipy.ndimage import zoom as zoom_gpu
from scipy.ndimage import zoom as zoom_cpu
from dataclasses import dataclass
from tools import UnitTools, SpaceEstimator, ArrayTools
from printer import Console
import dask.array as da
from tqdm import tqdm
import numpy as np
import cupy as cp
import pint
import sys

from interfaces import Interpolation


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


class ParameterMap(Interpolation):
    """
    This holds a map of a certain type (e.g., T1) of a certain metabolite (e.g., NAA). It's possible to interpolate
    the map to a certain target shape and to push it to the respective device (GPU <-> CPU). Ist also possible to
    change the unit and the data type.
    """
    def __init__(self, map_type: str, metabolite_name: str, values: np.ndarray | np.memmap, unit: pint.Unit, affine: np.ndarray=None):
        self.map_type: str = map_type                 # e.g., T1
        self.metabolite_name: str = metabolite_name   # e.g., NAA
        self.values: np.ndarray = values              # a numpy array
        self.unit: pint.Unit = unit                   # the unit
        self.affine: np.ndarray = affine              # TODO: Not yet used. E.g., if volumes obtain a different orientation.

    def to_device(self, device, verbose=False):
        """
        To bring the values of the maps object to the desired device.

        :param device: "cpu" or "cuda"/"gpu"
        :param verbose: if True then prints to the console.
        :return: Nothing
        """
        self.values = GPUTools.to_device(self.values, device=device)
        Console.printf("success", f"Brought Maps values to device: {device}", mute=not verbose)

    def interpolate(self, target_size: tuple, order: int, device: str, target_gpu: int = 0, verbose: bool=False):
        """
        To interpolate a volume to a certain size with a certain order.

        :param device: on which device it should be computed
        :param target_size: the desired shape
        :param order: the interpolation order
        :param target_gpu: When gpu is used then the index of the gpu. By default, the first one.
        :param verbose: If true then prints to console.
        :return: Nothing
        """

        self.values = InterpolationTools.interpolate(array=self.values,
                                                     target_size=target_size,
                                                     order=order,
                                                     compute_on_device=device,
                                                     return_on_device="cpu",
                                                     target_gpu=target_gpu,
                                                     verbose=verbose)
        return self


    def change_unit(self, target_unit: str, verbose: bool=True):
        """
        To change to the current unit (e.g., mmol -> mol)
        :param target_unit: the desired unit as string
        :param verbose: if True then prints to the console.
        :return: Nothing
        """
        self.values, self.unit = UnitTools.to_unit(values=self.values,current_units=self.unit, target_units=target_unit, return_separate=True, verbose=verbose)


    def change_data_type(self, data_type, verbose: bool=True):
        """
        To change the values to a desired data type. Possible for numpy and cupy.

        :param data_type: data type as string (e.g. float32)
        :param verbose: if True, then print to console
        :return: Nothing
        """
        data_type_before = self.values.dtype
        data_type_after = data_type
        self.values = self.values.astype(data_type_after)
        Console.printf("success", f"Changed data type from '{data_type_before}' => '{data_type_after}'", mute=not verbose)

    def __str__(self):
        """
        Print to the console the name(s) of the chemical compounds in the FID and the signal shape.
        """
        display_unit = "MiB"

        Console.add_lines(f"The parameter map contains the following properties:")
        Console.add_lines(f" => {'map type: ':.<20} {self.map_type}")
        Console.add_lines(f" => {'metabolite: ':.<20} {self.metabolite_name}")
        Console.add_lines(f" => {'unit: ':.<20} {self.unit}")
        Console.add_lines(f" => {'values shape: ':.<20} {self.values.shape}")
        Console.add_lines(f" => {'values data type: ':.<20} {self.values.dtype}")
        Console.add_lines(f" => {'required space: ':.<20} {SpaceEstimator.for_array(self.values, unit=display_unit, verbose=False):.5g} ({display_unit})")
        Console.printf_collected_lines("info")
        return "\n"


class ParameterVolume(Interpolation):
    """
    Can only be of one type of map (e.g., T1, or T2, ...), but it should be with multiple metabolites.
    """

    def __init__(self, maps_type):
        self.maps_type: str = maps_type # e.g, T1
        self.maps: list[ParameterMap, ...] = []

        # Relevant fields after the 4D volume is created:
        self.volume: np.ndarray = None
        self.metabolites: list[str] = [] # list of names of the metabolites, corresponds to the 4D array (1st dimension)
        self.unit: list = []
        self.data_type = None

    def add_map(self, map: ParameterMap, verbose: bool = False):
        """
        To add a ParameterMap to the current object.

        :param map: Parameter Map of class ParameterMap
        :param verbose: if True then print to console.
        :return: Nothing
        """

        if not map.map_type == self.maps_type:
            Console.printf("error", f"Cannot add Parameter Map of type {map.map_type} to {self.maps_type}. Create new object instead.")
            sys.exit()

        self.maps.append(map)
        Console.printf("success", f"Added to Parameter Maps of type {self.maps_type} the metabolite: {map.metabolite_name}.", mute=not verbose)


    def interpolate_maps(self, target_size: tuple, order: int = 3, device: str = "cpu", target_gpu: int = 0, verbose: bool = False):
        """
        To interpolate to the desired target size with desired order. Note, that this is done for each map but not the volume.

        :param target_size: the desired target size as tuple
        :param order: the interpolation order
        :param device: the device. Either "cpu", "gpu"/"cuda"
        :param target_gpu: if "gpu"/"cuda" is selected then the index of the gpu
        :param verbose: print to console if true
        :return:  Nothing
        """
        for i, m in enumerate(self.maps):
            self.maps[i] = m.interpolate(target_size=target_size, order=order, device=device, target_gpu=target_gpu, verbose=verbose)

        self.to_volume(verbose=verbose)
        Console.printf("warning", "The individual maps are interpolated, and then the 4D volume is created again. This might slow down the process.")

    # called previously interpolate_volume to distinguish between interpolate_maps but sice inherits from Interpolation it is not possible anymore....
    def interpolate(self, target_size: tuple, order: int = 3, device: str = "cpu", target_gpu: int = 0, verbose: bool = False):
        """
        To interpolate to the desired target size with desired order. This is done for the whole 4D volume.

        Note: target size is only the size of one metabolite map, therefore 3D

        :param target_size: the desired target size as tuple
        :param order: the interpolation order
        :param device: the device. Either "cpu", "gpu"/"cuda"
        :param target_gpu: if "gpu"/"cuda" is selected then the index of the gpu
        :param verbose: print to console if true
        :return:  Nothing
        """

        # When no Parameter Map is in the list
        if not self.maps:
            Console.printf("error", "First add at least one 3D Parameter Map to the object to create a 4D volume.")
            return
        # When at least one Parameter Map is in the list
        else:
            shape_old = self.volume.shape
            target_size = (shape_old[0], *target_size)  # to get 4D tuple
            self.volume = InterpolationTools.interpolate(
                array=self.volume,
                target_size=target_size,
                order=order,
                compute_on_device=device,
                return_on_device="cpu",
                target_gpu=target_gpu,
                verbose=verbose
            )


    def to_volume(self, verbose=True):
        """
        To create 4D volume out of individual 3D volumes. Thus, 3D maps of different metabolites by same map type (e.g. T1)
        are assembled to one 4D array of (metabolite, X,Y,Z).

        :param verbose: if True then prints to the console.
        :return: Nothing
        """

        # When no Parameter Map is in the list
        if not self.maps:
            Console.printf("error", "First add at least one 3D Parameter Map to the object to create a 4D volume.")
        # When at least one Parameter Map is in the list
        else:
            # Some check are required:
            # 1 check all objects same type (e.g., T1) -> already to when call add_map

            metabolites = [] # list of names of the metabolites
            volumes = []     # list of the volumes

            # 2 check all objects same shape
            #shapes = set([m.values.shape for m in self.maps])
            shapes = set([(next(iter(m.values.values())) if isinstance(m.values, dict) else m.values).shape for m in self.maps])

            units = set([m.unit for m in self.maps])
            if len(shapes) > 1:
                Console.printf("error", f"Cannot create 4D array of the 3D arrays since they exhibiting different shapes: {shapes}")

            # 3 check all objects same unit
            elif len(units) > 1:
                Console.printf("error", f"Cannot create 4D array of the 3D arrays since they exhibiting different units: {units} ")
            # 4 All objects are of same map type, exhibit same shape (X,Y,Z) and also h ave the same unit
            else:
                for m in self.maps:
                    metabolites.append(m.metabolite_name)
                    #volumes.append(m.values)  # shape (X,Y,Z)
                    volumes.append(next(iter(m.values.values())) if isinstance(m.values, dict) else m.values)  # shape (X,Y,Z)

                self.metabolites = metabolites
                self.volume = np.stack(volumes, axis=0)       # shape (metabolite, X,Y,Z)
                self.unit = self.maps[0].unit               # tale unit of first object since all must exhibit the in the list at this point (see above)
                Console.printf("success",
                               f"Created 4D volume of metabolite maps: {self.metabolites}. \n"
                               f" {'Map type: ':.<17} {self.maps_type} \n"
                               f" {'Unit: ':.<17} {self.unit} \n"
                               f" {'Shape: ':.<17} {self.volume.shape} \n"
                               f" {'Data type: ':.<17} {self.volume.dtype} \n"
                               f" {'Space: ':.<17} {SpaceEstimator.for_array(self.volume, unit='MiB', verbose=False):.5g}",
                               mute=not verbose)



    def size(self, unit="MiB", verbose=True) -> pint.Quantity:
        """
        To calculate the required size on the disk.

        :param unit: the desired unit (e.g., MiB, GiB, ...)
        :param verbose: prints the calculated size to the console if True
        :return:
        """
        return SpaceEstimator.for_array(self.volume, unit=unit, verbose=verbose)

    def to_data_type(self, data_type: str):
        """
        To just change the data type. E.g., from float46 to float32

        :param data_type: to change to data type
        :return: Nothing
        """
        if self.volume is None:
            Console.printf("error", "First run 'to_volume' to create th 4D volume, then it is possible to change the data type.")
        else:
            data_type_before = self.volume.dtype
            space_required_before = SpaceEstimator.for_array(self.volume, unit="MiB")
            #self.volume = self.volume.astype(data_type)
            self.volume = ArrayTools.to_data_type(self.volume, data_type=data_type, verbose=False)
            data_type_after = self.volume.dtype
            space_required_after = SpaceEstimator.for_array(self.volume, unit="MiB")
            Console.printf("success", f"Changed the data type from {data_type_before} ({space_required_before:.5g}) -> {data_type_after} ({space_required_after:.5g})")


    def reorder_metabolites(self, metabolites: list[str]):
        """
        To reorder the metabolites order of the list of the Parameter Maps. If already a volume out of them is created, then this is also reordered.

        As input simply a list of the names of the metabolite sin the desired order is required.

        :param metabolites: list of metabolites. E.g., ["NAA", "Glu", Glx", ...]
        :return: Nothing
        """
        current_order = list(self.metabolites)  # current metabolite names
        desired_order = list(metabolites)  # desired metabolite names

        # Error if there are not the same metabolite names inside both lists
        if set(current_order) != set(desired_order):
            missing = set(desired_order) - set(current_order)
            extra = set(current_order) - set(desired_order)
            Console.printf("error", f"Cannot sort. Metabolite sets do not match. Missing: {sorted(missing)}; Extra: {sorted(extra)}")
        # Warning if already desired order
        elif current_order == desired_order:
            Console.printf("warning", "No sorting is applied since already desired order is given!")
        # Otherwise, sort maps and value
        else:
            # Map current name -> current index, then build permutation for desired order
            idx = {name: i for i, name in enumerate(current_order)}
            perm = [idx[name] for name in desired_order]

            # (Option A) Reorder the metabolites in the list (of ParameterMaps)
            if self.maps is not None:
                # If maps is a list aligned with self.metabolites
                if len(self.maps) != len(current_order):
                    Console.printf("error", f"Containing {len(self.maps)} Parameter Maps but metabolites length is {len(current_order)}")
                else:
                    self.maps = [self.maps[i] for i in perm]
                    # Update metabolite order names in list
                    self.metabolites = desired_order
            else:
                Console.printf("warning", "The Parameter Maps cannot eb sorted since no elements are available.")

            # (Option B) Reorder the metabolite axis of the volume (axis 0)
            if self.volume is not None:
                if self.volume.shape[0] != len(current_order):
                    Console.printf("error", f"Volume first dimension is {self.volume.shape[0]} but metabolite length is {len(current_order)}")
                else:
                    self.volume = self.volume[perm, ...]
                    # Update metabolite order names in list
                    self.metabolites = desired_order
            else:
                Console.printf("warning", "The metabolite axis of the volume cannot be rearranged since it is empty. Run 'to_volume' first!")



    def __str__(self):
        """
        For using tge print function on the object. A pretty output ;)

        :return: Nothing (Except a new line ;))
        """
        display_unit = "MiB"

        Console.add_lines(f"The parameter map contains the following properties:")
        Console.add_lines(f" => {'maps type: ':.<20} {self.maps_type}")
        Console.add_lines(f" => {'metabolites: ':.<20} {self.metabolites}")
        Console.add_lines(f" => {'unit: ':.<20} {self.unit}")
        Console.add_lines(f" => {'values shape: ':.<20} {self.volume.shape}")
        Console.add_lines(f" => {'values data type: ':.<20} {self.volume.dtype}")
        Console.add_lines(f" => {'required space: ':.<20} {SpaceEstimator.for_array(self.volume, unit=display_unit, verbose=False):.5g} ({display_unit})")
        Console.printf_collected_lines("info")
        return "\n"


    def display_jupyter(self, display: str = "maps"):
        """
        To display this volume interactive. Either the maps of the transformed volume is displayed. To distinguish only necessary
        if manually a change was made at maps or volume.

        One use case for example: Crating this object, holding volumes in dict, transforming to 4D volume. If change in volume
        and if only then maps were displayed, the change is not plotted. Therefore, this two ways are available.

        :param display:
        :return:
        """

        if not display in ("maps", "volume"):
            Console.printf("error", f"Only possible arguments for 'display' are 'volume' and 'maps'. But was given: {display}")
        else:
            maps_volumes_list = None
            maps_names_list = None

            if display == "maps":
                maps_volumes_list = [m.values for m in self.maps]
                maps_names_list = [m.metabolite_name for m in self.maps]
            elif display == "volume":
                if self.volume is None:
                    Console.printf("error", "Cannot display the volume. Run to_volume first to create the 4D volume.")
                else:
                    maps_volumes_list = list(self.volume)
                    maps_names_list = self.metabolites
            else:
                # I assume this cas will never happen ;)
                Console.printf("error", "An error occurred while trying to display the volume.")

            JupyterPlotManager.volume_grid_viewer(vols=maps_volumes_list, rows=1, cols=len(maps_names_list), titles=maps_names_list)


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
                 concentration_maps, ##Maps,
                 t1_maps,            ##Maps,
                 t2_maps,            ##Maps,
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


class MetabolicPropertyMap:
    """
    This is class is to pack together the different maps (so far T1, T2, concentration) of one metabolite (e.g. Glucose).
    (!) It also transforms the numpy maps to a dask-array extension called CustomArray and further defines the block size for computations.

    The MetabolicPropertyMap It is mainly used in the class MetabolicPropertyMapsAssembler.
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

        # The name of the metabolite of one object
        self.chemical_compound_name = chemical_compound_name

        # The associates Maps with the metabolite:
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
