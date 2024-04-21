from printer import Console
import dask.array as da
import numpy as np
import torch
import math
import pint
import sys

u = pint.UnitRegistry()  # for using units


class CustomArray(da.Array):
    """
    Extending the dask Array class with additional functionalities.

    The Custom dask Array obtains the following additional features:
    * block number of the individual block
    * the block index (see how .blocks.reval() crates a list of blocks)
    * the shape of the main volume in each individual block
    * the total number of blocks in each individual block
    * the unit
    * custom metadata

    Further, consider the coordinate system of the axis 0 = x, axis 1 = y, axis 2 = z:

                   origin
                 +----------> (+z)
                /|
              /  |
            /    |
        (+x)     |
                 v (+y)

        This is important when dealing with individual blocks of the whole volume. The block number is
        increased first in z, then y and then x direction.
    """

    def __new__(cls,
                dask_array: da.Array,
                block_number: int = None,
                block_idx: tuple = None,
                main_volume_shape: tuple = None,
                total_number_blocks: int = None,
                unit: pint.UnitRegistry | str = None,
                meta: dict = None):
        """
        To create a new instance of the dask.Array class and add custom attributes.

        Note: The __new__ is called before __init__. It is responsible for creating a new instance of the class. After, the new object is created with
              __new__ the __init__ is called for instantiate the new created object.

        :param dask_array: dask.Array object
        :param block_number: when creating blocks out of an array with the argument in the dask from_array(): chunks=(size_x, size_y, size_z). A number in elements [0:number_blocks]
        :param block_idx: tuple containing the block index for each dimension. E,g., (1, 3, 5) for block position: x=1, y=3, z=5. Check coordinate system in description of the class.
        :param main_volume_shape: the shape of the main volume in x,y,z for a 3D array, for example.
        :param total_number_blocks: the total number of blocks of which the main volume consists.
        :param unit: unit corresponding to the values in the array.
        :param meta: custom data added by the user.
        """
        instance = super().__new__(cls, dask_array.dask, dask_array.name, dask_array.chunks, dtype=dask_array.dtype)
        instance.block_number = block_number
        instance.block_idx = block_idx
        instance.unit = unit
        instance.size_mb = dask_array.dtype.itemsize * dask_array.size / (1024 ** 2)

        # When no shape is given, it will be assumed it is the main volume rather than a block of the main volume. Otherwise, set the shape of the main volume in each block.
        # Important, if only the block is given and information of the main volume is required.
        if main_volume_shape is None:
            instance.main_volume_shape = instance.shape
        else:
            instance.main_volume_shape = main_volume_shape
        # The same as above. If no shape is given, it will be assumed that it is the main volume and not a block of it. Otherwise, assign the number of blocks to each block
        # to be able to know the total number of blocks.
        if total_number_blocks is None:
            instance.total_number_blocks = math.prod(dask_array.blocks.shape) - 1
        else:
            instance.total_number_blocks = total_number_blocks

        instance.meta = meta

        return instance

    def __repr__(self) -> str:
        """
        To extend the __repr__ method from the superclass and to add additional information if a CustomArray gets printed to the console with print().

        Additional information:
        * Start and end indices of the block in the global volume (only for x,y,z dimensions supported yet)
        * The block number (e.g., 4/100)
        * The block coordinates (e.g., (0,0,4))
        * The estimated size of the block in [MB]
        * The shape of the main volume (e.g., 1000, 120,120,120)
        * The set unit
        * The custom metadata

        :return: formatted string.
        """
        arr_repr = super().__repr__()  # Get the standard representation of the Dask array
        x_global_start, y_global_start, z_global_start = self.get_global_index(0, 0, 0)
        x_global_end, y_global_end, z_global_end = self.get_global_index(self.blocks.shape[0] - 1, self.blocks.shape[1] - 1, self.blocks.shape[2] - 1)

        meta_repr = (f"\nBlock number: {self.block_number}/{self.total_number_blocks} "
                     f"\nBlock coordinates: {self.block_idx} "
                     f"\nEstimated size: {round(self.size_mb, 3)} MB "
                     f"\nMain volume shape: {self.main_volume_shape} "
                     f"\nMain volume coordinates: x={x_global_start}:{x_global_end} y={y_global_start}:{y_global_end} z={z_global_start}:{z_global_end} "
                     f"\nUnit: {self.unit} "
                     f"\nMetadata: {self.meta}")

        return arr_repr + meta_repr + "\n"  # Concatenate the two representations

    def __mul__(self, other):
        """
        For preserving information from the left multiplicand. Add it to the result.

        :param other: The right multiplicand. It has to be a dask.Array or a CustomArray
        :return: product of array 1 and array 2
        """
        result = super().__mul__(other)
        result = CustomArray(dask_array=result,
                             block_number=self.block_number,
                             block_idx=self.block_idx,
                             main_volume_shape=self.main_volume_shape,
                             total_number_blocks=self.total_number_blocks,
                             meta=self.meta)

        return result

    def get_global_index(self, x, y, z):
        """
        Useful if a block of a CustomArray is handled individually. To get the global indices (x,y,z) of the local indices in the respective block.
        local(x,y,z) ===> global(x,y,z)

        :param x: local index in x (thus in current block)
        :param y: local index in y (thus in current block)
        :param z: local index in z (thus in current block)
        :return: global indices as tuple (x,y,z)
        """
        # x,y,z is in block, and global in original global volume
        if x >= self.shape[0]:
            print(f"Error: x block shape is only {self.shape[0]}, but index {x} was given")
            return
        if y >= self.shape[1]:
            print(f"Error: y block shape is only {self.shape[1]}, but index {y} was given")
            return
        if z >= self.shape[2]:
            print(f"Error: z block shape is only {self.shape[2]}, but index {z} was given")
            return

        global_x = self.block_idx[0] * self.shape[0] + x
        global_y = self.block_idx[1] * self.shape[1] + y
        global_z = self.block_idx[2] * self.shape[2] + z

        return global_x, global_y, global_z

    @property
    def blocks(self):
        """
        To override the method of the superclass. Create a CustomBLockView instead of the dask BlockView. However, the CustomBlockView inherits from
        the dask BlockView, but extends its functionality!

        :return: CustomBlockView
        """

        block_view_object = CustomBlockView(self, main_volume_shape=self.main_volume_shape, total_number_blocks=self.total_number_blocks)
        return block_view_object


class CustomBlockView(da.core.BlockView):
    """
    Extending the dask class BlockView with additional functionalities. This is required for the CustomArray class that inherits from dask Array.
    Additional functionalities of one block:
    * Shape of the main volume in each block
    * Number of total blocks in each block
    * Unit of the main volume in each block
    """

    def __init__(self, custom_array: CustomArray, main_volume_shape: tuple=None, total_number_blocks: int=None):
        """
        Addition additional the main volume shape, the total number of blocks and the unit. Also, call the super constructor.

        :param custom_array: the CustomArray object
        :param main_volume_shape:
        :param total_number_blocks:
        """
        self.main_volume_shape = main_volume_shape
        self.total_number_blocks = total_number_blocks
        self.block_number = 0
        self.unit = custom_array.unit
        super().__init__(custom_array)

    def __getitem__(self, index) -> CustomArray:
        """
        Override the __getitem__ of the superclass. In the workflow of dask each block is a new dask.Array. Thus, replacing the dask.Array with
        a CustomArray that inherits from dask.Array.

        Be aware that each block has a block number. The block number allows selecting the next block. The blocks are  ordered condescending in
        the following dimensions: first z, then y, then x, with the following coordinate system:

               origin
                 +----------> (+z)
                /|
              /  |
            /    |
        (+x)     |
                 v (+y)

        :param index: index of the block
        :return: a block as CustomArray
        """
        dask_array = super(CustomBlockView, self).__getitem__(index)
        custom_dask_array = CustomArray(dask_array=dask_array,
                                        block_number=self.block_number,
                                        block_idx=index,
                                        main_volume_shape=self.main_volume_shape,
                                        total_number_blocks=self.total_number_blocks,
                                        unit=self.unit)

        self.block_number += 1  # to get block number of in the main volume (first z, then y, then x)

        return custom_dask_array



class ConfiguratorGPU():
    """
    TODO!!! Describe it
    """

    def __init__(self, required_space_gpu: np.ndarray):

        torch.cuda.empty_cache()
        self.available_gpus: int = torch.cuda.device_count()
        self.selected_gpu: int = None
        self.required_space_gpu = required_space_gpu
        self.free_space_selected_gpu: torch.Tensor = None

    def select_least_busy_gpu(self):
        # based on percentage free
        free_space_devices = []
        for device in range(self.available_gpus):
            space_total_cuda = torch.cuda.mem_get_info(device)[1]
            space_free_cuda = torch.cuda.mem_get_info(device)[0]
            percentage_free_space = space_free_cuda / space_total_cuda
            free_space_devices.append(percentage_free_space)

        self.selected_gpu = free_space_devices.index(max(free_space_devices))

        torch.cuda.set_device(self.selected_gpu)
        Console.printf("info", f"Selected GPU {self.selected_gpu} -> most free space at moment!")

        self.free_space_selected_gpu = torch.cuda.mem_get_info(self.selected_gpu)[0]  # free memory on GPU [bytes]

    def print_available_gpus(self):
        Console.add_lines("Available GPU(s) and free space on it:")

        for device in range(self.available_gpus):
            space_total_cuda = torch.cuda.mem_get_info(device)[1]
            space_free_cuda = torch.cuda.mem_get_info(device)[0]
            percentage_free_space = space_free_cuda / space_total_cuda
            Console.add_lines(f" GPU {device} [MB]: {space_free_cuda / (1024 ** 2)} / {space_total_cuda / (1024 ** 2)} ({round(percentage_free_space, 2) * 100}%)")

        Console.printf_collected_lines("info")

    def select_gpu(self, gpu_index: int = 0):
        torch.cuda.set_device(gpu_index)
        self.selected_gpu = gpu_index
        Console.printf("info", f"Selected GPU {gpu_index} -> manually selected by the user!")

    def enough_space_available(self) -> bool:

        if self.selected_gpu is not None:
            self.free_space_selected_gpu = torch.cuda.mem_get_info(self.selected_gpu)[0]  # free memory on GPU [bytes]

            if self.required_space_gpu <= self.free_space_selected_gpu:
                Console.printf("info",
                               f"Possible to put whole tensor of size {self.required_space_gpu / (1024 ** 2)} [MB] on GPU. Available: {self.free_space_selected_gpu / (1024 ** 2)} [MB]")
            else:
                Console.printf("info",
                               f"Not possible to put whole tensor of size {self.required_space_gpu / (1024 ** 2)} [MB] on GPU. Available: {self.free_space_selected_gpu / (1024 ** 2)} [MB]")

        else:
            Console.printf("error", f"No GPU is selected. Number of available GPUs: {self.available_gpus}")
            sys.exit()

        # TODO: Check return type!
        return self.required_space_gpu <= self.free_space_selected_gpu


# put estimate space here

class SpaceEstimator:
    """
    To calculate the required space on the disk for a numpy array of a given shape and data type. Usefully, if a big array
    should be created and to check easily if it does not exceed a certain required space.
    """

    @staticmethod
    def for_numpy(data_shape: tuple, data_type: np.dtype, unit: str = "MB") -> np.ndarray:
        """
        For estimating the required space of a numpy array with a desired shape and data type.

        :param data_shape: desired shape of the numpy array
        :param data_type: desired data type of the numpy array (e.g., np.int64)
        :param unit: desired unit the numpy array
        :return: required space on disk as a numpy array (with defined unit: [bytes], [KB], [MB], [GB]). Standard unit is [MB].
        """

        # np.prod...  To get total numpy array size
        # itemsize... bytes required by each element in array
        space_required_bytes = np.prod(data_shape) * data_type.itemsize

        if unit is "byte":
            return space_required_bytes

        elif unit is "KB":
            return space_required_bytes * 1 / 1024

        elif unit is "MB":
            return space_required_bytes * 1 / (1024 ** 2)

        elif unit is "GB":
            return space_required_bytes * 1 / (1024 ** 3)

    @staticmethod
    def for_torch(torch_array: torch.Tensor, unit: str = "MB"):
        """
        TODO: Implement it.
        """
        raise NotImplementedError("This method is not yet implemented")
