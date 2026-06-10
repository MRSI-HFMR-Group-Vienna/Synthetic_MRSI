import time

from tools import JupyterPlotManager
from tools import CustomArray, GPUTools, SortingTools, DaskTools, InterpolationTools
from cupyx.scipy.ndimage import zoom as zoom_gpu
from scipy.ndimage import zoom as zoom_cpu
from dataclasses import dataclass
from tools import UnitTools, SpaceEstimator, ArrayTools
from prettyconsole import Console
import dask.array as da
from tqdm import tqdm
import numpy as np
import cupy as cp
import pint
import sys

from interface import InterpolationInterface


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


class ParameterMap(InterpolationInterface):
    """
    This holds a map of a certain type (e.g., T1) of a certain metabolite (e.g., NAA). It's possible to interpolate
    the map to a certain target shape and to push it to the respective device (GPU <-> CPU). Ist also possible to
    change the unit and the data type.
    """
    def __init__(self,
                 map_type_name: str,
                 map_name: str,
                 values: np.ndarray | np.memmap=None,
                 unit: pint.Unit=None,
                 affine: np.ndarray=None):

        self.map_type: str = map_type_name            # e.g., T1
        self.map_name: str = map_name                 # e.g., NAA
        self.values: np.ndarray = values              # a numpy array
        self.unit: pint.Unit = unit                   # the unit
        self.affine: np.ndarray = affine              # TODO: Not yet used. E.g., if volumes obtain a different orientation.

        self.applied_methods = [] # Just to collect already applied methods to this object

    def to_device(self, device, verbose=False):
        """
        To bring the values of the maps object to the desired device.

        :param device: "cpu" or "cuda"/"gpu"
        :param verbose: if True then prints to the console.
        :return: the object itself
        """
        self.values = GPUTools.to_device(self.values, device=device)
        Console.printf("success", f"Brought Maps values to device: {device}", mute=not verbose)

        self.applied_methods.append(f"put array on device: {device}")

        return self

    def interpolate(self, target_size: tuple, order: int, compute_on_device: str = "cpu", target_gpu: int = 0, return_on_device: str = "cpu",verbose: bool=False):
        """
        To interpolate a volume to a certain size with a certain order.

        :param compute_on_device: on which device it should be computed
        :param return_on_device: on which device it should be returned (not yet possible to specify the gpu id)
        :param target_size: the desired shape
        :param order: the interpolation order
        :param target_gpu: When gpu is used then the index of the gpu. By default, the first one.
        :param verbose: If true then prints to console.
        :return: Nothing
        """

        self.values = InterpolationTools.interpolate(array=self.values,
                                                     target_size=target_size,
                                                     order=order,
                                                     compute_on_device=compute_on_device,
                                                     return_on_device=return_on_device,
                                                     target_gpu=target_gpu,
                                                     verbose=verbose)

        self.applied_methods.append(f"interpolate with order {order} to shape: {target_size}")

        return self


    def change_unit(self, target_unit: str, verbose: bool=True):
        """
        To change to the current unit (e.g., mmol -> mol)
        :param target_unit: the desired unit as string
        :param verbose: if True then prints to the console.
        :return: the object itself
        """
        unit_before = self.unit  # just for self.applied_methods
        unit_after = target_unit # just for self.applied_methods

        self.values, self.unit = UnitTools.to_unit(values=self.values,current_units=self.unit, target_units=target_unit, return_separate=True, verbose=verbose)

        self.applied_methods.append(f"change unit from {unit_before} to {unit_after}")

        return self

    def change_data_type(self, data_type, verbose: bool=True):
        """
        To change the values to a desired data type. Possible for numpy and cupy.

        :param data_type: data type as string (e.g. float32)
        :param verbose: if True, then print to console
        :return: the object itself
        """
        data_type_before = self.values.dtype
        data_type_after = data_type
        self.values = self.values.astype(data_type_after)
        Console.printf("success", f"Changed data type from '{data_type_before}' => '{data_type_after}'", mute=not verbose)

        self.applied_methods.append(f"change data type from {data_type_before} to {data_type_after}")

        return self

    def check_nan(self):
        """
        Check if Not a Number (NaN) are present in the array of this object. Just prints the result to the console.
        :return: object itself to apply further methods
        """
        nans_present = ArrayTools.check_nan(self.values, verbose=True)

        self.applied_methods.append(f"checked for NaNs: {nans_present}")

        return self

    def enforce_positive_values(self, convert_negative, convert_zeros):
        """
        To convert negative values and/or zeros values to eps value.

        :return: the object itself to concatenate further method calls
        """

        self.values = ArrayTools.enforce_min_eps(array=self.values, convert_negative=convert_negative, convert_zeros=convert_zeros)

        self.applied_methods.append(f"Enforce positive eps. Converting negative values: {convert_negative}; converting zero values: {convert_zeros}")

        return self

    def create_dummy_volume(self,
                            shape: tuple = None,
                            fill_value: float | int | list[int | float] = None,
                            mask: np.ndarray | list[np.ndarray] = None,
                            unit: pint.Unit = None):
        """
        For now this only supports the following:

            Case 1: Shape is given (no mask) and one fill value
                    => create one array uniform with values of desired shape

            Case 2: Shape is NOT given and mask and one fill values:
                    => create one array uniform within the maks area of shape of mask

            case 3: Shape is NOT given but multiple masks and fill values:
                    => create one array with values weighted on the mask
                       For example: weight(mask_WM) * value_WM + weight(mask_GM) * value_GM
                                    The weight(mask_WM) + weight(mask_WM) need to sum up to 1 everywhere


        :param unit: The unit of the created array (as pint.Unit)
        :param shape: shape as tuple (e.g., (X, Y, Z, ...)
        :param fill_value: one scalar or array of values
        :param mask: one scalar or array of values
        :return: The object itself with now created values
        """

        # Create the array and assign to the current object
        self.values = ArrayTools.create_volume(shape=shape, value=fill_value, mask=mask)
        self.unit = unit

        # Return the object itself with now the created array for the self.values
        return self


    def conjugate(self, verbose: bool=False):
        """
        The complex conjugate of a complex number is obtained by changing the sign of its imaginary part.

        :return: Nothing
        """

        if self.values.dtype.kind != "c":
            Console.printf("error", f"Cannot complex conjugate the values of the Parameter Map '{self.map_name}' "
                                    f"with data type: {self.values.dtype}")
        else:
            xp = ArrayTools.get_backend(self.values)
            self.values = xp.conjugate(self.values)
            Console.printf("success", "Complex conjugated the volume of the ParameterMap.", mute=not verbose)


    def __str__(self):
        """
        Print to the console the name(s) of the chemical compounds in the FID and the signal shape.
        """
        display_unit = "MiB"

        Console.add_lines(f"The parameter map contains the following properties:")
        Console.add_lines(f" => {'map type: ':.<20} {self.map_type}")
        Console.add_lines(f" => {'map name: ':.<20} {self.map_name}")
        Console.add_lines(f" => {'unit: ':.<20} {self.unit}")
        Console.add_lines(f" => {'values shape: ':.<20} {self.values.shape}")
        Console.add_lines(f" => {'values data type: ':.<20} {self.values.dtype}")
        Console.add_lines(f" => {'required space: ':.<20} {SpaceEstimator.for_array(self.values, unit=display_unit, verbose=False):.5g} ({display_unit})")
        Console.printf_collected_lines("info")
        return "\n"


class ParameterVolume(InterpolationInterface):
    """
    Can only be of one type of map (e.g., T1, or T2, ...), but it should be with multiple metabolites.
    """

    def __init__(self, maps_type):
        self.maps_type: str = maps_type # e.g, T1
        self.maps: list[ParameterMap, ...] = []

        # Relevant fields after the 4D volume is created:
        self.volume: np.ndarray = None
        self.names: list[str] = [] # list of names of the metabolites or other maps, corresponds to the 4D array (1st dimension)
        self.unit: list = []
        self.data_type = None

    def add_map(self, map_object: ParameterMap, verbose: bool = False):
        """
        To add a ParameterMap to the current object.

        :param map_object: Parameter Map of class ParameterMap
        :param verbose: if True then print to console.
        :return: Nothing
        """

        if not map_object.map_type == self.maps_type:
            Console.printf("error", f"Cannot add Parameter Map of type {map_object.map_type} to {self.maps_type}. Create new object instead.")
            sys.exit()

        self.maps.append(map_object)
        Console.printf("success", f"Added to Parameter Maps of type {self.maps_type} the map: {map_object.map_name}.", mute=not verbose)

    def simulate_dummy_map(self,
                           map_name: str,
                           fill_value: int | float,
                           mask: np.ndarray | list[np.ndarray] = None,
                           unit: pint.Unit = None,
                           device_interpolate: str ="cpu",
                           target_gpu_interpolate: int =0,
                           verbose: bool = False):
        """
        To create a ParameterMap that will be added to this object of the same shape as the other ParameterMaps (based on the first
        entry). The use cases are:

                In any case state the desired metabolite name (also including water, and so on...),
                Then:
                    a) Give the desire fill value with and without mask
                    b) Give the desired fill values and masks


        :param unit: The unit of the ParameterMap (pint.Unit)
        :param mask: One of multiple masks in a list or None
        :param map_name: the name of the metabolite for what the dummy data will be created
        :param fill_value: The fill value or values in list (need the same length then mask)

        :param target_gpu_interpolate: If if created values (array) of the Parameter Map does not fit the other ParameterMaps
                                       due to a missmatch of the provided masks (different shape to already here existing ones)
        :param device_interpolate: If interpolation required then 'cpu' or 'gpu'/'cuda'
        :param verbose:
        :return:
        """

        for m in self.maps:
            if m.map_name == map_name:
                Console.printf("error", f"The map name '{map_name}' already exists! Therefore, cannot be created!")
                return

        # The shape required for the final ParameterMap.
        target_shape = self.maps[0].values.shape

        # (!) Important to note: When mask(s) is/are provided then shape needs to be none for
        #                        ParameterMap.create_dummy_volume
        shape_create_volume = None if mask is None else self.maps[0].values.shape

        # To create the dummy values Parameter Map
        parameter_map = ParameterMap(map_type_name=self.maps_type, map_name=map_name).create_dummy_volume(shape=shape_create_volume, fill_value=fill_value, mask=mask, unit=unit)

        # If the shape of the created volume (in the ParameterMap) does not fit the shape of the
        # shape of the ParameterMaps already collected in this object (of Parameter _Volume).
        # (!) Please note: Interpolation after creating the ParameterMap.values since it is better
        #                  to interpolate the final results than multiple times multiple masks (if
        #                  provided)
        if parameter_map.values.shape != self.maps[0].values.shape:
            parameter_map.interpolate(target_size=target_shape, device=device_interpolate, target_gpu=target_gpu_interpolate)

        self.add_map(parameter_map)

        return self

    def interpolate_maps(self, target_size: tuple, order: int = 3, compute_on_device: str = "cpu", target_gpu: int = 0, return_on_device: str = "cpu", verbose: bool = False):
        """
        To interpolate to the desired target size with desired order. Note, that this is done for each map but not the volume.

        :param target_size: the desired target size as tuple
        :param order: the interpolation order
        :param compute_on_device: interpolation of desired device. Either "cpu", "gpu"/"cuda". GPU is usually much faster!
        :param return_on_device: after interpolation on which device the array should reside. CPU is preferred.
        :param target_gpu: if "gpu"/"cuda" is selected then the index of the gpu
        :param verbose: print to console if true
        :return: the object itself
        """
        for i, m in enumerate(self.maps):
            self.maps[i] = m.interpolate(target_size=target_size, order=order, compute_on_device=compute_on_device, target_gpu=target_gpu, return_on_device=return_on_device, verbose=verbose)

        self.to_volume(verbose=verbose)
        Console.printf("warning", "The individual maps are interpolated, and then the 4D volume is created again. This might slow down the process.")

        return self

    # called previously interpolate_volume to distinguish between interpolate_maps but sice inherits from Interpolation it is not possible anymore....
    def interpolate(self, target_size: tuple, order: int = 3, compute_on_device: str = "cpu", target_gpu: int = 0, return_on_device: str = "cpu", verbose: bool = False):
        """
        To interpolate to the desired target size with desired order. This is done for the whole 4D volume.

        Note: target size is only the size of one metabolite map, therefore 3D

        :param target_size: the desired target size as tuple
        :param order: the interpolation order
        :param compute_on_device: to compute on the desired device. Either "cpu", "gpu"/"cuda". GPU is much faster!
        :param return_on_device: to bring the array to cpu or gpu after computation. Either "cpu", "gpu"/"cuda
        :param target_gpu: if "gpu"/"cuda" is selected then the index of the gpu
        :param verbose: print to console if true
        :return: the object itself
        """

        # When no Parameter Map is in the list
        if not self.maps:
            Console.printf("error", "First add at least one Parameter Map to the object to create a stacked volume.")
            return
        # When at least one Parameter Map is in the list
        else:
            shape_old = self.volume.shape
            target_size = (shape_old[0], *target_size)  # to get 4D tuple
            self.volume = InterpolationTools.interpolate(
                array=self.volume,
                target_size=target_size,
                order=order,
                compute_on_device=compute_on_device,
                return_on_device=return_on_device,
                target_gpu=target_gpu,
                verbose=verbose
            )

        return self

    def drop(self, name: str | list[str]):
        """
        Possible to remove one or multiple maps by name(s). This also remove it from the volume.

        :param name: name of the desired chemical compound or metabolite
        :return: The object itself
        """
        if isinstance(name, str):
            name = [name]

        # Check if all names are valid
        invalid = [n for n in name if n not in self.names]
        if invalid:
            Console.add_lines("cannot drop:")
            for n in invalid:
                Console.add_lines(f" -> {n}")

            Console.add_lines("Only possible:")
            for n in self.names:
                Console.add_lines(f" -> {n}")

            Console.printf_collected_lines("error")

            return self

        Console.add_lines("Removed the following from ParameterVolume:", tag="drop")
        for n in name:
            # Find the index and the array to remove
            name_index = self.names.index(n)

            # Get the backend (cupy or numpy)
            xp = ArrayTools.get_backend(self.volume)
            self.volume = xp.delete(self.volume, name_index, axis=0)

            # Remove it from the list here
            self.names.remove(n)

            # Remove it from also from the list of maps
            self.maps.pop(name_index)

            #print(n)

            Console.add_lines(f" => {n}", tag="drop")
        Console.printf_collected_lines("success", tag="drop")

        return self


    def to_volume(self, verbose=True):
        """
        To create one stacked volume out of individual parameter map arrays. Maps of different metabolites with
        the same map type, e.g. T1, are assembled to one array with shape (metabolite, *map_shape).
        Note: But this class can also be used for other map types, not only metabolites!

        :param verbose: if True then prints to the console.
        :return: the object itself
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
                Console.printf("error", f"Cannot create stacked volume since the parameter map arrays exhibit different shapes: {shapes}")

            # 3 check all objects same unit
            elif len(units) > 1:
                Console.add_lines("Cannot create stacked volume since the parameter maps exhibit different units:")
                for m in self.maps:
                    Console.add_lines(f" > {m.map_name:.<15}: {str(m.unit):<10}")
                Console.printf_collected_lines("error")

            # 4 All objects are of same map type, exhibit same shape (..., X,Y,Z) and also h ave the same unit
            else:
                for m in self.maps:
                    metabolites.append(m.map_name)
                    #volumes.append(m.values)  # shape (..., X,Y,Z)
                    volumes.append(next(iter(m.values.values())) if isinstance(m.values, dict) else m.values)  # shape (X,Y,Z)

                self.names = metabolites
                self.volume = np.stack(volumes, axis=0)       # shape (...W,X,Y,Z)
                self.unit = self.maps[0].unit               # tale unit of first object since all must exhibit the in the list at this point (see above)
                Console.printf("success",
                               f"Created stacked volume of the following maps: {self.names}. \n"
                               f" {'Map type: ':.<17} {self.maps_type} \n"
                               f" {'Unit: ':.<17} {self.unit} \n"
                               f" {'Shape: ':.<17} {self.volume.shape} \n"
                               f" {'Data type: ':.<17} {self.volume.dtype} \n"
                               f" {'Space: ':.<17} {SpaceEstimator.for_array(self.volume, unit='MiB', verbose=False):.5g}",
                               mute=not verbose)

        return self


    def conjugate(self, apply_to_maps: bool = True, verbose: bool = True):
        """
        The complex conjugate of a complex number is obtained by changing the sign of its imaginary part.

        :return: Nothing
        """

        if not self.volume.dtype.kind == 'c':
            Console.printf("error", f"Cannot complex conjugate the volume since data type: {self.volume.dtype}")
            return self

        Console.add_lines(f"Complex conjugated:")

        # Conjugate the whole volume
        xp = ArrayTools.get_backend(self.volume)
        self.volume = xp.conjugate(self.volume)
        Console.add_lines(f" {'> The volume of the Parameter _Volume:':<{50}}: True")

        # Check if the use also wants to complex conjugate the individual maps objects (to be consistent), besides
        # interpolating the whole volume (created out of all maps)
        number_maps = len(self.maps)
        if apply_to_maps:
            for i in range(number_maps):
                self.maps[i] = self.maps[i].conjugate()
            Console.add_lines(f" {f'> Also the {number_maps} individual maps objects:':<{50}}: {apply_to_maps}")

        Console.printf_collected_lines("success", mute=not verbose)

        return self


    def get_maps(self):
         # TODO
         # To get one (by name) or all maps
        raise NotImplementedError


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
        :return: the object itself
        """
        current_order = list(self.names)  # current metabolite names
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
                # If maps is a list aligned with self.names
                if len(self.maps) != len(current_order):
                    Console.printf("error", f"Containing {len(self.maps)} Parameter Maps but metabolites length is {len(current_order)}")
                else:
                    self.maps = [self.maps[i] for i in perm]
                    # Update metabolite order names in list
                    self.names = desired_order
            else:
                Console.printf("warning", "The Parameter Maps cannot eb sorted since no elements are available.")

            # (Option B) Reorder the metabolite axis of the volume (axis 0)
            if self.volume is not None:
                if self.volume.shape[0] != len(current_order):
                    Console.printf("error", f"_Volume first dimension is {self.volume.shape[0]} but metabolite length is {len(current_order)}")
                else:
                    self.volume = self.volume[perm, ...]
                    # Update metabolite order names in list
                    self.names = desired_order
            else:
                Console.printf("warning", "The metabolite axis of the volume cannot be rearranged since it is empty. Run 'to_volume' first!")

        return self


    def __str__(self):
        """
        For using tge print function on the object. A pretty output ;)

        :return: Nothing (Except a new line ;))
        """
        display_unit = "MiB"

        Console.add_lines(f"The parameter map contains the following properties:")
        Console.add_lines(f" => {'maps type: ':.<20} {self.maps_type}")
        Console.add_lines(f" => {'maps names: ':.<20} {self.names}")
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
                maps_names_list = [m.map_name for m in self.maps]
            elif display == "volume":
                if self.volume is None:
                    Console.printf("error", "Cannot display the volume. Run to_volume first to create the 4D volume.")
                else:
                    maps_volumes_list = list(self.volume)
                    maps_names_list = self.names
            else:
                # I assume this cas will never happen ;)
                Console.printf("error", "An error occurred while trying to display the volume.")

            JupyterPlotManager.volume_grid_viewer(vols=maps_volumes_list, rows=1, cols=len(maps_names_list), titles=maps_names_list)



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
